"""Descriptive-comparative analysis of copresence dimensions in Valais.

The restricted workbook is read-only. Every exported artifact is aggregate and
contains no respondent identifier, free text, timestamp, or municipality-level
table. The active design compares retired secondary-home respondents domiciled
outside Valais with retired primary residents who do not declare a former R2
status in the municipality.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import textwrap
import unicodedata
import warnings
from pathlib import Path
from typing import Iterable

import matplotlib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2, fisher_exact, mannwhitneyu, rankdata
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.sm_exceptions import ConvergenceWarning

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


DATA_SHEET = "SHE-vivreensemle"
INFO_SHEET = "information"
SURVEY_YEAR = 2026
RANDOM_SEED = 20260814
BOOTSTRAP_REPS = 5000
PERMUTATION_REPS = 5000

CORE_COLUMNS = {
    "status2": "Domicile_2_nivaux",
    "status3": "Domicile_3_nivaux",
    "progress": "PROGRESSION",
    "professional_status": "Quelle_est_votre_situation_professionnel",
    "age": "Groupe_dage",
    "gender": "Quel_est_votre_sexe",
    "key": "CLE",
}

SOCIODEMOGRAPHIC_COLUMNS = {
    "nationality": "Quelle_est_votre_nationalite",
    "education": "Quel_est_votre_plus_haut_niveau_de_forma",
    "retirement_year": "Depuis_quelle_annee_etes-vous_a_la_retra",
    "former_profession": "Quelle_est_votre_profession_ou_la_dernie",
    "household_income": "Quel_est_le_revenu_mensuel_brut_de_votre",
}

RESIDENTIAL_COLUMNS = {
    "r2_seasons": "A_quelles_periodes_y_sejournez-vous_prin",
    "r1_always_lived": "Avez-vous_toujours_vecu_dans_cette_commu",
    "r1_former_secondary": "Etiez-vous_auparavant_resident_secondair",
    "r1_current_secondary": "Combien_de_residences_secondaires_possed",
}

TERRITORY_COMPOSITION_SPECS = {
    "organisations": {
        "label": "Types d'organisations locales",
        "source": "De_quelles_associations_de_votre_commune",
        "mode": "multiple",
        "categories": [
            ("residents_property", "Association de résidents/propriétaires"),
            ("cultural", "Association culturelle (fanfare, chant, etc.),"),
            ("sports", "Club sportif (ski-club, pétanque, club de marche, etc.)"),
            ("political", "Parti politique"),
            ("foundation", "Conseil de fondation"),
            ("other", "Autre"),
            ("none", "Aucune"),
        ],
    },
    "people_first": {
        "label": "Personnes fréquentées au premier rang",
        "source": "Quand_vous_etes_dans_cette_commune_quel1",
        "mode": "first",
        "categories": [
            ("family", "Famille"),
            ("friends", "Amies ou amis"),
            ("secondary_residents", "Personnes en résidence secondaire"),
            ("primary_residents", "Personnes en résidence principale"),
            ("compatriots", "Compatriotes"),
            ("expatriates", "Personnes expatriées"),
            ("other", "Autre"),
        ],
    },
    "places_first": {
        "label": "Lieux fréquentés au premier rang",
        "source": "Quels_sont_les_lieux_que_vous_frequente1",
        "mode": "first",
        "categories": [
            ("shops", "Commerces (alimentation, boutiques)"),
            ("sports_facilities", "Installations sportives (salle de sport, terrain, piscine)"),
            ("restaurants", "Restaurants, cafés, bars"),
            ("public_spaces", "Espaces publics (exemple\u00a0: place de jeux, place du village)"),
            ("administrative_buildings", "Bâtiments administratifs (commune, poste)"),
            ("leisure_spaces", "Espaces de loisirs (cinéma, bibliothèque, centre culturel)"),
            ("worship", "Lieux de culte"),
            ("associational", "Lieux associatifs"),
            ("other", "Autre"),
        ],
    },
    "activities_first": {
        "label": "Activités au premier rang",
        "source": "Quels_sont_les_activites_que_vous_pratiq",
        "mode": "first",
        "categories": [
            ("sports", "Activités sportives (exemple\u00a0: gym, marche)"),
            ("home", "Activités à domicile (exemple: lecture, musique, peinture, etc.)"),
            ("cultural", "Activités culturelles (concerts, théâtre, expositions)"),
            ("local_events", "Participation à des manifestations locales (exemple : loto)"),
            ("religious", "Activités religieuses"),
            ("other", "Autre"),
        ],
    },
}

LIKERT = {
    "pas du tout d accord": 1,
    "plutot pas d accord": 2,
    "ni d accord ni pas d accord": 3,
    "plutot d accord": 4,
    "tout a fait d accord": 5,
}

LIKERT_LABELS = {
    1: "Pas du tout d'accord",
    2: "Plutôt pas d'accord",
    3: "Ni d'accord ni pas d'accord",
    4: "Plutôt d'accord",
    5: "Tout à fait d'accord",
}

FREQUENCY = {
    "quelques jours par annee": 1,
    "plusieurs sejours par annee": 2,
    "chaque mois": 3,
    "chaque semaine": 4,
}

FREQUENCY_LABELS = {
    1: "Quelques jours par année",
    2: "Plusieurs séjours par année",
    3: "Chaque mois",
    4: "Chaque semaine",
}

GROUP_LABELS = {
    "r2_retired_outside": "R2 retraités hors Valais",
    "r1_retired": "R1 retraités sans ancien statut R2 déclaré dans la commune",
}

LIKERT_PROFILE_IDS = [
    "TR05",
    "TR06",
    "RV01",
    "RV02",
    "RV03",
    "VT01",
    "VT02",
    "VT03",
    "LG01",
    "ER01",
    "ER03",
]

PRIMARY_COMPARISON_ID = "ER03"

# Files emitted by superseded versions of this descriptive-comparative workflow.
# They are removed explicitly so that a fresh run cannot leave obsolete models,
# the former trajectory wording or superseded conceptual figures beside active outputs.
DEPRECATED_TABLE_OUTPUTS = (
    "tab_02_sample_characteristics.csv",
    "tab_03_copresence_distribution.csv",
    "tab_03_residential_trajectory.csv",
    "tab_04_primary_binary_model.csv",
    "tab_04a_unadjusted_binary_models.csv",
    "tab_07_reciprocal_cohabitation.csv",
    "tab_s01_expanded_social_model.csv",
    "tab_s01b_demographic_sensitivity.csv",
    "tab_s01c_commune_covariance_sensitivity.csv",
    "tab_s02_home_feeling_model.csv",
    "tab_s03_ordinal_sensitivity.csv",
    "tab_s04_attachment_linear_trend.csv",
    "tab_s05_adjusted_attachment_probabilities.csv",
    "tab_08_ordinal_dimension_models.csv",
    "tab_s08_ordinal_model_coefficients.csv",
    "tab_s09_proportional_odds_diagnostics.csv",
    "tab_s10_outcome_category_counts.csv",
    "tab_s03_all_r1_sensitivity.csv",
    "tab_s05_contact_indicator_sensitivity.csv",
    "tab_s06_owner_association_sensitivity.csv",
    "tab_s07_former_secondary_r1_sensitivity.csv",
    "tab_s10_standardized_predictions.csv",
)

DEPRECATED_FIGURE_OUTPUTS = (
    "fig_01_copresence_distribution.png",
    "fig_01_copresence_distribution.pdf",
    "fig_02_copresence_primary_model.png",
    "fig_02_copresence_primary_model.pdf",
    "fig_03_adjusted_attachment_probabilities.png",
    "fig_03_adjusted_attachment_probabilities.pdf",
    "fig_02_r2_dimension_profile.png",
    "fig_02_r2_dimension_profile.pdf",
    "fig_03_group_agreement_differences.png",
    "fig_03_group_agreement_differences.pdf",
    "fig_04_reciprocal_cohabitation.png",
    "fig_04_reciprocal_cohabitation.pdf",
)

# The five dimensions are analytical blocks, not reflective scales.  The item
# sets are fixed here and exposed in diagnostics so that the omnibus tests and
# ordinal models cannot silently drift away from the manuscript.
DIMENSION_SPECS: dict[str, dict[str, object]] = {
    "D1": {
        "label": "Usages et sociabilités territoriales",
        "hypothesis": "H1",
        "indicator_ids": ["TU01", "TU02", "TU03", "TU04"],
    },
    "D2": {
        "label": "Rôle perçu dans le vivre-ensemble",
        "hypothesis": "H2",
        "indicator_ids": ["RV01", "RV02", "RV03"],
    },
    "D3": {
        "label": "Vision du territoire",
        "hypothesis": "H3",
        "indicator_ids": ["VT01", "VT02", "VT03"],
    },
    "D4": {
        "label": "Ancrage et légitimité territoriale",
        "hypothesis": "H4",
        "indicator_ids": ["TR05", "TR06", "LG01"],
    },
    "D5": {
        "label": "Perception des relations intergroupes",
        "hypothesis": "H5",
        "indicator_ids": ["ER01", "ER03"],
    },
}

OUTCOME_SPECS: dict[str, dict[str, str]] = {
    "mefiance_intergroupe": {
        "indicator_id": "ER01",
        "label": "Perception de la méfiance intergroupe",
    },
    "cohabitation_en_miroir": {
        "indicator_id": "ER03",
        "label": "Évaluation en miroir de la cohabitation avec l’autre groupe",
    },
}

PROFILE_MODEL_PREDICTORS = [
    "group_r2",
    "female",
    "origin_valais",
    "retirement_duration5",
    "tertiary_education",
    "income_category",
]

MODEL_LABELS = {
    "group_r2": "R2 retraité hors Valais (vs R1 sans ancien statut R2 déclaré dans la commune)",
    "female": "Femme (vs homme)",
    "origin_valais": "Originaire du Valais",
    "retirement_duration5": "Durée depuis la retraite (5 ans)",
    "tertiary_education": "Formation tertiaire",
    "income_category": "Revenu du ménage (1 catégorie)",
}


def normalize_text(value: object) -> str | None:
    """Normalize accents, apostrophes, punctuation, and whitespace."""
    if pd.isna(value):
        return None
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^a-zA-Z0-9+]+", " ", text).strip().lower()


def recode_likert(value: object) -> float:
    normalized = normalize_text(value)
    return float(LIKERT[normalized]) if normalized in LIKERT else np.nan


def split_choices(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [part.strip() for part in str(value).split(";") if part.strip()]


def first_choice(value: object) -> str | None:
    """Return the first-ranked response while preserving the questionnaire label."""
    choices = split_choices(value)
    return choices[0] if choices else None


def selected_choice(value: object, expected: str) -> float:
    if pd.isna(value):
        return np.nan
    target = normalize_text(expected)
    choices = {normalize_text(part) for part in split_choices(value)}
    return float(target in choices)


def selection_count(value: object) -> float:
    if pd.isna(value):
        return np.nan
    return float(len(split_choices(value)))


def organization_member(value: object) -> float:
    if pd.isna(value):
        return np.nan
    choices = {normalize_text(part) for part in split_choices(value)}
    nonempty = {choice for choice in choices if choice and choice != "aucune"}
    return float(bool(nonempty))


def yes_no(value: object) -> float:
    normalized = normalize_text(value)
    if normalized == "oui":
        return 1.0
    if normalized == "non":
        return 0.0
    return np.nan


def retirement_duration(value: object, age_group: object) -> float:
    """Return years since retirement after a conservative plausibility audit.

    The age bands do not reveal exact age. A year is retained only if retirement
    could have occurred at age 18 or older for at least one age in the reported
    band. The open 81+ band receives only the global lower bound.
    """
    year = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(year) or not float(year).is_integer():
        return np.nan
    year = int(year)
    if year > SURVEY_YEAR or year < 1900:
        return np.nan
    normalized_age = normalize_text(age_group)
    upper_bounds = {"45 64": 64, "65 80": 80}
    if normalized_age in upper_bounds:
        earliest_year = SURVEY_YEAR - upper_bounds[normalized_age] + 18
        if year < earliest_year:
            return np.nan
    return float(SURVEY_YEAR - year)


def household_income_score(value: object) -> float:
    """Map the five ordered income bands to 1--5; refusals remain missing."""
    normalized = normalize_text(value)
    mapping = {
        "moins de 4 000 chf": 1.0,
        "4 000 5 999 chf": 2.0,
        "6 000 7 999 chf": 3.0,
        "8 000 11 999 chf": 4.0,
        "12 000 chf ou plus": 5.0,
    }
    return mapping.get(normalized, np.nan)


def load_export(path: Path) -> tuple[pd.DataFrame, dict[str, str], pd.DataFrame]:
    raw = pd.read_excel(path, sheet_name=DATA_SHEET, header=0)
    labels = raw.iloc[0].fillna("").astype(str).to_dict()
    data = raw.iloc[1:].reset_index(drop=True)
    info = pd.read_excel(path, sheet_name=INFO_SHEET, header=None)
    return data, labels, info


def load_item_map(path: Path) -> pd.DataFrame:
    items = pd.read_csv(path, dtype=str).fillna("")
    required = {
        "domain",
        "indicator_id",
        "indicator_label",
        "r2_source",
        "r1_source",
        "scale",
        "comparability",
        "analysis_role",
        "missing_policy",
        "interpretive_limit",
    }
    missing = sorted(required - set(items.columns))
    if missing:
        raise ValueError(f"Item map columns missing: {missing}")
    if items["indicator_id"].duplicated().any():
        duplicates = items.loc[items["indicator_id"].duplicated(), "indicator_id"].tolist()
        raise ValueError(f"Duplicate indicator identifiers: {duplicates}")
    return items


def verify_schema(data: pd.DataFrame, items: pd.DataFrame) -> None:
    mapped = set(items["r2_source"]).union(items["r1_source"])
    composition_sources = {
        str(spec["source"]) for spec in TERRITORY_COMPOSITION_SPECS.values()
    }
    required = (
        set(CORE_COLUMNS.values())
        .union(SOCIODEMOGRAPHIC_COLUMNS.values())
        .union(RESIDENTIAL_COLUMNS.values())
        .union(composition_sources)
        .union(source for source in mapped if source)
    )
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Required source columns missing: {missing}")


def group_masks(data: pd.DataFrame) -> dict[str, pd.Series]:
    complete = data[CORE_COLUMNS["progress"]].map(
        lambda value: (normalize_text(value) or "").startswith("termine")
    )
    retired = data[CORE_COLUMNS["professional_status"]].eq("Retraité")
    r2_outside = data[CORE_COLUMNS["status3"]].eq("Secondaire_hors_Valais")
    r1 = data[CORE_COLUMNS["status2"]].eq("Principale")
    former_r2 = data[RESIDENTIAL_COLUMNS["r1_former_secondary"]].map(yes_no)
    return {
        "r2_retired_outside": complete & retired & r2_outside,
        # A verified non-R2 history requires an explicit "Non" response. The
        # 51 former R2 respondents and three missing history responses are both
        # excluded from the principal comparator and reported in the flow.
        "r1_retired": complete & retired & r1 & former_r2.eq(0),
    }


def source_for(items: pd.DataFrame, indicator_id: str, group_id: str) -> str:
    row = items.loc[items["indicator_id"].eq(indicator_id)]
    if len(row) != 1:
        raise KeyError(indicator_id)
    key = "r2_source" if group_id == "r2_retired_outside" else "r1_source"
    return str(row.iloc[0][key])


def indicator_row(items: pd.DataFrame, indicator_id: str) -> pd.Series:
    row = items.loc[items["indicator_id"].eq(indicator_id)]
    if len(row) != 1:
        raise KeyError(indicator_id)
    return row.iloc[0]


def harmonized_series(
    data: pd.DataFrame,
    mask: pd.Series,
    items: pd.DataFrame,
    indicator_id: str,
    group_id: str,
) -> pd.Series:
    """Return a harmonized respondent series for one indicator and group."""
    source = source_for(items, indicator_id, group_id)
    if not source:
        return pd.Series(dtype=float)
    raw = data.loc[mask, source]

    if indicator_id == "TR01":
        return raw.map(yes_no)
    if indicator_id == "TR02":
        year = pd.to_numeric(raw, errors="coerce")
        duration = SURVEY_YEAR - year
        return duration.where(duration.between(0, 126))
    if indicator_id == "TR03":
        return raw.map(lambda value: FREQUENCY.get(normalize_text(value), np.nan))
    if indicator_id == "TR04":
        return raw.copy()
    if indicator_id == "TR07":
        return raw.copy()
    if indicator_id in {"TR08", "TR09", "TR10"}:
        return raw.map(yes_no)
    if indicator_id == "TU01":
        return raw.map(organization_member)
    if indicator_id == "TU02":
        target = (
            "Personnes en résidence principale"
            if group_id == "r2_retired_outside"
            else "Personnes en résidence secondaire"
        )
        return raw.map(lambda value: selected_choice(value, target))
    if indicator_id in {"TU03", "TU04"}:
        return raw.map(selection_count)
    if indicator_id == "TU05":
        return raw.map(
            lambda value: selected_choice(
                value,
                "Participation à des manifestations locales (exemple : loto)",
            )
        )
    if str(indicator_row(items, indicator_id)["scale"]).startswith("likert"):
        return raw.map(recode_likert)
    return raw.copy()


def likert_states(raw: pd.Series) -> dict[str, int]:
    normalized = raw.map(normalize_text)
    valid = raw.map(recode_likert).notna()
    sans = normalized.eq("sans avis")
    blank = raw.isna()
    other = ~(valid | sans | blank)
    return {
        "valid_n": int(valid.sum()),
        "sans_avis_n": int(sans.sum()),
        "blank_n": int(blank.sum()),
        "other_invalid_n": int(other.sum()),
    }


def quantile_or_nan(values: pd.Series, q: float) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return float(clean.quantile(q)) if len(clean) else math.nan


def wilson_interval(count: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n <= 0:
        return math.nan, math.nan
    z = 1.959963984540054
    p = count / n
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    return center - half, center + half


def newcombe_difference_interval(
    count1: int,
    n1: int,
    count2: int,
    n2: int,
) -> tuple[float, float]:
    if n1 <= 0 or n2 <= 0:
        return math.nan, math.nan
    p1, p2 = count1 / n1, count2 / n2
    low1, high1 = wilson_interval(count1, n1)
    low2, high2 = wilson_interval(count2, n2)
    difference = p1 - p2
    lower = difference - math.sqrt((p1 - low1) ** 2 + (high2 - p2) ** 2)
    upper = difference + math.sqrt((high1 - p1) ** 2 + (p2 - low2) ** 2)
    return lower, upper


def dominance_delta(x: Iterable[float], y: Iterable[float]) -> float:
    """Return Cliff's delta, P(X > Y) - P(X < Y), with ties contributing zero."""
    x_values = np.asarray(list(x), dtype=float)
    y_values = np.asarray(list(y), dtype=float)
    x_values = x_values[np.isfinite(x_values)]
    y_values = y_values[np.isfinite(y_values)]
    if len(x_values) == 0 or len(y_values) == 0:
        return math.nan
    x_unique, x_counts = np.unique(x_values, return_counts=True)
    y_unique, y_counts = np.unique(y_values, return_counts=True)
    signs = np.sign(x_unique[:, None] - y_unique[None, :])
    pair_counts = x_counts[:, None] * y_counts[None, :]
    return float((signs * pair_counts).sum() / (len(x_values) * len(y_values)))


def bootstrap_dominance_interval(
    x: Iterable[float],
    y: Iterable[float],
    *,
    seed: int,
    reps: int = BOOTSTRAP_REPS,
) -> tuple[float, float]:
    x_values = np.asarray(list(x), dtype=float)
    y_values = np.asarray(list(y), dtype=float)
    x_values = x_values[np.isfinite(x_values)]
    y_values = y_values[np.isfinite(y_values)]
    if len(x_values) == 0 or len(y_values) == 0:
        return math.nan, math.nan
    support = np.union1d(x_values, y_values)
    x_counts = np.array([(x_values == value).sum() for value in support])
    y_counts = np.array([(y_values == value).sum() for value in support])
    x_prob = x_counts / x_counts.sum()
    y_prob = y_counts / y_counts.sum()
    signs = np.sign(support[:, None] - support[None, :])
    rng = np.random.default_rng(seed)
    x_boot = rng.multinomial(len(x_values), x_prob, size=reps)
    y_boot = rng.multinomial(len(y_values), y_prob, size=reps)
    numerators = np.einsum("bi,ij,bj->b", x_boot, signs, y_boot)
    deltas = numerators / (len(x_values) * len(y_values))
    return float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))


def benjamini_hochberg(p_values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(p_values), dtype=float)
    adjusted = np.full(values.shape, np.nan)
    finite = np.isfinite(values)
    if not finite.any():
        return adjusted
    p = values[finite]
    order = np.argsort(p)
    ranked = p[order]
    m = len(ranked)
    raw = ranked * m / np.arange(1, m + 1)
    monotone = np.minimum.accumulate(raw[::-1])[::-1]
    restored = np.empty_like(monotone)
    restored[order] = np.minimum(monotone, 1.0)
    adjusted[finite] = restored
    return adjusted


def likert_summary(
    data: pd.DataFrame,
    mask: pd.Series,
    items: pd.DataFrame,
    indicator_id: str,
    group_id: str,
) -> dict[str, object]:
    row = indicator_row(items, indicator_id)
    source = source_for(items, indicator_id, group_id)
    raw = data.loc[mask, source]
    values = raw.map(recode_likert)
    states = likert_states(raw)
    valid = values.dropna().astype(int)
    agree_n = int(valid.isin([4, 5]).sum())
    disagree_n = int(valid.isin([1, 2]).sum())
    neutral_n = int(valid.eq(3).sum())
    low, high = wilson_interval(agree_n, len(valid))
    result: dict[str, object] = {
        "domain": row["domain"],
        "indicator_id": indicator_id,
        "indicator_label": row["indicator_label"],
        "group_id": group_id,
        "group_label": GROUP_LABELS[group_id],
        "eligible_n": int(mask.sum()),
        **states,
        "disagree_n": disagree_n,
        "disagree_pct": 100 * disagree_n / len(valid) if len(valid) else math.nan,
        "neutral_n": neutral_n,
        "neutral_pct": 100 * neutral_n / len(valid) if len(valid) else math.nan,
        "agree_n": agree_n,
        "agree_pct": 100 * agree_n / len(valid) if len(valid) else math.nan,
        "agree_ci_low_pct": 100 * low,
        "agree_ci_high_pct": 100 * high,
        "median": quantile_or_nan(valid, 0.5),
        "q1": quantile_or_nan(valid, 0.25),
        "q3": quantile_or_nan(valid, 0.75),
        "interpretive_limit": row["interpretive_limit"],
    }
    for level in range(1, 6):
        count = int(valid.eq(level).sum())
        result[f"level_{level}_n"] = count
        result[f"level_{level}_pct"] = 100 * count / len(valid) if len(valid) else math.nan
    return result


def full_likert_distributions(
    data: pd.DataFrame,
    masks: dict[str, pd.Series],
    items: pd.DataFrame,
    group_ids: Iterable[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_id in group_ids:
        for indicator_id in LIKERT_PROFILE_IDS:
            row = indicator_row(items, indicator_id)
            source = source_for(items, indicator_id, group_id)
            raw = data.loc[masks[group_id], source]
            coded = raw.map(recode_likert)
            valid_n = int(coded.notna().sum())
            total_n = int(masks[group_id].sum())
            for level, label in LIKERT_LABELS.items():
                count = int(coded.eq(level).sum())
                rows.append(
                    {
                        "domain": row["domain"],
                        "indicator_id": indicator_id,
                        "indicator_label": row["indicator_label"],
                        "group_id": group_id,
                        "group_label": GROUP_LABELS[group_id],
                        "response": label,
                        "response_order": level,
                        "count": count,
                        "pct_valid": 100 * count / valid_n if valid_n else math.nan,
                        "pct_eligible": 100 * count / total_n if total_n else math.nan,
                        "valid_n": valid_n,
                        "eligible_n": total_n,
                    }
                )
            normalized = raw.map(normalize_text)
            states = [
                ("Sans avis", int(normalized.eq("sans avis").sum()), 6),
                ("Blanc", int(raw.isna().sum()), 7),
                (
                    "Autre non valide",
                    int((~(coded.notna() | normalized.eq("sans avis") | raw.isna())).sum()),
                    8,
                ),
            ]
            for label, count, order in states:
                rows.append(
                    {
                        "domain": row["domain"],
                        "indicator_id": indicator_id,
                        "indicator_label": row["indicator_label"],
                        "group_id": group_id,
                        "group_label": GROUP_LABELS[group_id],
                        "response": label,
                        "response_order": order,
                        "count": count,
                        "pct_valid": math.nan,
                        "pct_eligible": 100 * count / total_n if total_n else math.nan,
                        "valid_n": valid_n,
                        "eligible_n": total_n,
                    }
                )
    return pd.DataFrame(rows)


def relational_response_summary(distributions: pd.DataFrame) -> pd.DataFrame:
    """Collapse the two H5 items into readable low/neutral/high distributions.

    A tension signal is agreement (4--5) for ER01, which asserts mistrust, and
    disagreement (1--2) for ER03, which evaluates cohabitation unfavourably.
    This is descriptive and adds no inferential test.
    """
    rows: list[dict[str, object]] = []
    for outcome_id, spec in OUTCOME_SPECS.items():
        indicator_id = str(spec["indicator_id"])
        for group_id in ("r2_retired_outside", "r1_retired"):
            selected = distributions.loc[
                distributions["indicator_id"].eq(indicator_id)
                & distributions["group_id"].eq(group_id)
                & distributions["response_order"].between(1, 5)
            ].copy()
            counts = {
                int(row.response_order): int(row.count)
                for row in selected.itertuples(index=False)
            }
            valid_n = int(selected["valid_n"].iloc[0])
            low_n = counts.get(1, 0) + counts.get(2, 0)
            neutral_n = counts.get(3, 0)
            high_n = counts.get(4, 0) + counts.get(5, 0)
            tension_n = high_n if indicator_id == "ER01" else low_n
            favourable_n = low_n if indicator_id == "ER01" else high_n
            tension_low, tension_high = wilson_interval(tension_n, valid_n)
            rows.append(
                {
                    "outcome_id": outcome_id,
                    "indicator_id": indicator_id,
                    "outcome_label": str(spec["label"]),
                    "group_id": group_id,
                    "group_label": GROUP_LABELS[group_id],
                    "valid_n": valid_n,
                    "low_n": low_n,
                    "low_pct": 100 * low_n / valid_n,
                    "neutral_n": neutral_n,
                    "neutral_pct": 100 * neutral_n / valid_n,
                    "high_n": high_n,
                    "high_pct": 100 * high_n / valid_n,
                    "tension_n": tension_n,
                    "tension_pct": 100 * tension_n / valid_n,
                    "tension_ci_low_pct": 100 * tension_low,
                    "tension_ci_high_pct": 100 * tension_high,
                    "favourable_n": favourable_n,
                    "favourable_pct": 100 * favourable_n / valid_n,
                    "tension_definition": (
                        "accord avec l'existence d'une méfiance (4–5)"
                        if indicator_id == "ER01"
                        else "cohabitation défavorable (1–2)"
                    ),
                }
            )
    return pd.DataFrame(rows)


def comparison_indicator_ids(items: pd.DataFrame) -> list[str]:
    """Return the 15 indicators assigned to H1--H5, in model order."""
    available = set(items["indicator_id"].astype(str))
    ordered = [
        str(indicator_id)
        for spec in DIMENSION_SPECS.values()
        for indicator_id in spec["indicator_ids"]
    ]
    missing = [indicator_id for indicator_id in ordered if indicator_id not in available]
    if missing:
        raise KeyError(f"Dimension indicators absent from item map: {missing}")
    return ordered


def compare_groups(
    data: pd.DataFrame,
    masks: dict[str, pd.Series],
    items: pd.DataFrame,
    comparator_id: str,
    indicator_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    selected_ids = list(indicator_ids) if indicator_ids is not None else comparison_indicator_ids(items)
    for index, indicator_id in enumerate(selected_ids):
        item = indicator_row(items, indicator_id)
        x = harmonized_series(
            data, masks["r2_retired_outside"], items, indicator_id, "r2_retired_outside"
        )
        y = harmonized_series(data, masks[comparator_id], items, indicator_id, comparator_id)
        x_numeric = pd.to_numeric(x, errors="coerce").dropna()
        y_numeric = pd.to_numeric(y, errors="coerce").dropna()
        common: dict[str, object] = {
            "domain": item["domain"],
            "indicator_id": indicator_id,
            "indicator_label": item["indicator_label"],
            "comparability": item["comparability"],
            "comparison": f"R2 retraités hors Valais – {GROUP_LABELS[comparator_id]}",
            "n_r2": len(x_numeric),
            "n_comparator": len(y_numeric),
            "primary_comparison": indicator_id == PRIMARY_COMPARISON_ID,
            "interpretive_limit": item["interpretive_limit"],
            "r2_median": quantile_or_nan(x_numeric, 0.5),
            "comparator_median": quantile_or_nan(y_numeric, 0.5),
            "r2_q1": quantile_or_nan(x_numeric, 0.25),
            "r2_q3": quantile_or_nan(x_numeric, 0.75),
            "comparator_q1": quantile_or_nan(y_numeric, 0.25),
            "comparator_q3": quantile_or_nan(y_numeric, 0.75),
        }
        if item["scale"] == "binary":
            success_x = int(x_numeric.eq(1).sum())
            success_y = int(y_numeric.eq(1).sum())
            p_x = success_x / len(x_numeric) if len(x_numeric) else math.nan
            p_y = success_y / len(y_numeric) if len(y_numeric) else math.nan
            low, high = newcombe_difference_interval(
                success_x, len(x_numeric), success_y, len(y_numeric)
            )
            contingency = [
                [success_x, len(x_numeric) - success_x],
                [success_y, len(y_numeric) - success_y],
            ]
            p_value = (
                float(fisher_exact(contingency, alternative="two-sided").pvalue)
                if len(x_numeric) and len(y_numeric)
                else math.nan
            )
            common.update(
                {
                    "scale_family": "binary",
                    "effect_metric": "différence de proportions (points de pourcentage)",
                    "effect_estimate": 100 * (p_x - p_y),
                    "effect_ci_low": 100 * low,
                    "effect_ci_high": 100 * high,
                    "p_value": p_value,
                    "r2_positive_pct": 100 * p_x,
                    "comparator_positive_pct": 100 * p_y,
                    "r2_agree_pct": math.nan,
                    "comparator_agree_pct": math.nan,
                    "agree_difference_pp": math.nan,
                    "agree_difference_ci_low_pp": math.nan,
                    "agree_difference_ci_high_pp": math.nan,
                }
            )
        else:
            if len(x_numeric) and len(y_numeric):
                p_value = float(
                    mannwhitneyu(
                        x_numeric,
                        y_numeric,
                        alternative="two-sided",
                        method="asymptotic",
                    ).pvalue
                )
            else:
                p_value = math.nan
            delta = dominance_delta(x_numeric, y_numeric)
            low, high = bootstrap_dominance_interval(
                x_numeric,
                y_numeric,
                seed=RANDOM_SEED + 101 * index,
            )
            agree_x = int(x_numeric.isin([4, 5]).sum()) if item["scale"] == "likert_5" else 0
            agree_y = int(y_numeric.isin([4, 5]).sum()) if item["scale"] == "likert_5" else 0
            if item["scale"] == "likert_5":
                ax = agree_x / len(x_numeric) if len(x_numeric) else math.nan
                ay = agree_y / len(y_numeric) if len(y_numeric) else math.nan
                alow, ahigh = newcombe_difference_interval(
                    agree_x, len(x_numeric), agree_y, len(y_numeric)
                )
            else:
                ax = ay = alow = ahigh = math.nan
            common.update(
                {
                    "scale_family": "likert_ordinal"
                    if item["scale"] == "likert_5"
                    else "selection_count",
                    "effect_metric": "delta de Cliff",
                    "effect_estimate": delta,
                    "effect_ci_low": low,
                    "effect_ci_high": high,
                    "p_value": p_value,
                    "r2_positive_pct": math.nan,
                    "comparator_positive_pct": math.nan,
                    "r2_agree_pct": 100 * ax,
                    "comparator_agree_pct": 100 * ay,
                    "agree_difference_pp": 100 * (ax - ay),
                    "agree_difference_ci_low_pp": 100 * alow,
                    "agree_difference_ci_high_pp": 100 * ahigh,
                }
            )
        rows.append(common)
    result = pd.DataFrame(rows)
    result["q_bh"] = benjamini_hochberg(result["p_value"])
    return result


def pooled_indicator_series(
    data: pd.DataFrame,
    masks: dict[str, pd.Series],
    items: pd.DataFrame,
    indicator_id: str,
) -> pd.Series:
    """Harmonize one indicator over the focal R2 and retired-R1 groups."""
    pooled_mask = masks["r2_retired_outside"] | masks["r1_retired"]
    result = pd.Series(np.nan, index=data.index[pooled_mask], dtype=float)
    for group_id in ("r2_retired_outside", "r1_retired"):
        values = harmonized_series(data, masks[group_id], items, indicator_id, group_id)
        result.loc[values.index] = pd.to_numeric(values, errors="coerce")
    return result


def dimension_global_tests(
    data: pd.DataFrame,
    masks: dict[str, pd.Series],
    items: pd.DataFrame,
    *,
    repetitions: int = PERMUTATION_REPS,
) -> pd.DataFrame:
    """Test five global no-difference hypotheses without constructing scores.

    For each item, the effect is Cliff's delta.  The dimension statistic is the
    mean squared item delta.  A synchronized permutation of the residential
    group label preserves dependence among a dimension's items and calibrates
    the omnibus p-value.  This is a conditional working-exchangeability test,
    not design-based population inference.
    """
    pooled_index = data.index[
        masks["r2_retired_outside"] | masks["r1_retired"]
    ]
    group = masks["r2_retired_outside"].loc[pooled_index].to_numpy(dtype=bool)
    rng = np.random.default_rng(RANDOM_SEED + 7001)
    permutations = np.empty((repetitions, len(pooled_index)), dtype=bool)
    for repetition in range(repetitions):
        permutations[repetition] = rng.permutation(group)

    rows: list[dict[str, object]] = []
    for dimension_id, spec in DIMENSION_SPECS.items():
        indicator_ids = list(spec["indicator_ids"])
        observed_deltas: list[float] = []
        permuted_squared: list[np.ndarray] = []
        valid_counts: list[str] = []
        for indicator_id in indicator_ids:
            values = pooled_indicator_series(data, masks, items, indicator_id).reindex(
                pooled_index
            )
            valid = values.notna().to_numpy()
            numeric = values.to_numpy(dtype=float)[valid]
            observed_group = group[valid]
            x = numeric[observed_group]
            y = numeric[~observed_group]
            observed_deltas.append(dominance_delta(x, y))
            valid_counts.append(f"{indicator_id}:{len(x)}/{len(y)}")

            ranks = rankdata(numeric, method="average")
            permuted_group = permutations[:, valid]
            n_r2 = permuted_group.sum(axis=1).astype(float)
            n_r1 = len(numeric) - n_r2
            rank_sum_r2 = permuted_group @ ranks
            u_r2 = rank_sum_r2 - n_r2 * (n_r2 + 1) / 2
            denominator = n_r2 * n_r1
            delta = np.divide(
                2 * u_r2,
                denominator,
                out=np.full(repetitions, np.nan, dtype=float),
                where=denominator > 0,
            ) - 1
            permuted_squared.append(delta**2)

        observed_statistic = float(np.nanmean(np.square(observed_deltas)))
        permutation_statistic = np.nanmean(np.vstack(permuted_squared), axis=0)
        p_value = float(
            (1 + np.sum(permutation_statistic >= observed_statistic))
            / (repetitions + 1)
        )
        rows.append(
            {
                "dimension_id": dimension_id,
                "dimension_label": spec["label"],
                "hypothesis": spec["hypothesis"],
                "indicator_ids": "; ".join(indicator_ids),
                "item_count": len(indicator_ids),
                "valid_n_r2_r1_by_item": "; ".join(valid_counts),
                "global_statistic_mean_squared_cliffs_delta": observed_statistic,
                "root_mean_squared_cliffs_delta": math.sqrt(observed_statistic),
                "p_value": p_value,
                "permutation_repetitions": repetitions,
                "method": "permutation synchronisée de la moyenne des deltas de Cliff au carré",
            }
        )
    result = pd.DataFrame(rows)
    result["q_bh_five_dimensions"] = benjamini_hochberg(result["p_value"])
    result["decision_alpha_0_05"] = np.where(
        result["q_bh_five_dimensions"].lt(0.05),
        "Différence détectée",
        "Aucune différence détectée",
    )
    return result


def pooled_model_frame(
    data: pd.DataFrame,
    masks: dict[str, pd.Series],
    items: pd.DataFrame,
) -> pd.DataFrame:
    """Build a non-exported individual model frame for the two retired groups."""
    pooled_mask = masks["r2_retired_outside"] | masks["r1_retired"]
    index = data.index[pooled_mask]
    raw = data.loc[index]
    frame = pd.DataFrame(index=index)
    frame["group_r2"] = masks["r2_retired_outside"].loc[index].astype(float)

    age = raw[CORE_COLUMNS["age"]]
    gender = raw[CORE_COLUMNS["gender"]]
    frame["female"] = gender.eq("Femme").astype(float).where(
        gender.isin(["Femme", "Homme"])
    )
    frame["origin_valais"] = raw["Etes-vous_originaire_du_Valais"].map(yes_no)
    frame["retirement_duration5"] = pd.Series(
        [
            retirement_duration(value, age_value) / 5
            if math.isfinite(retirement_duration(value, age_value))
            else np.nan
            for value, age_value in zip(
                raw[SOCIODEMOGRAPHIC_COLUMNS["retirement_year"]],
                age,
                strict=True,
            )
        ],
        index=index,
        dtype=float,
    )
    education = raw[SOCIODEMOGRAPHIC_COLUMNS["education"]]
    frame["tertiary_education"] = education.eq(
        "Tertiaire (bachelor, master, doctorat)"
    ).astype(float).where(education.notna())

    frame["income_category"] = raw[
        SOCIODEMOGRAPHIC_COLUMNS["household_income"]
    ].map(household_income_score)

    for dimension in DIMENSION_SPECS.values():
        for indicator_id in dimension["indicator_ids"]:
            frame[str(indicator_id)] = pooled_indicator_series(
                data, masks, items, str(indicator_id)
            ).reindex(index)

    for outcome_id, spec in OUTCOME_SPECS.items():
        frame[outcome_id] = frame[str(spec["indicator_id"])]

    return frame


def _fit_ordered_model(sample: pd.DataFrame, outcome: str, predictors: list[str]):
    model = OrderedModel(
        sample[outcome].astype(int),
        sample[predictors].astype(float),
        distr="logit",
    )
    result = model.fit(
        method="bfgs",
        disp=False,
        maxiter=1200,
        cov_type="HC3",
    )
    if not bool(result.mle_retvals.get("converged", False)):
        raise RuntimeError(f"Ordered model did not converge for {outcome}: {predictors}")
    return result


def _robust_wald_block(fit, terms: list[str]) -> tuple[float, int, float]:
    """Joint Wald test using the model's HC3 covariance matrix."""
    if not terms:
        return math.nan, 0, math.nan
    beta = fit.params.loc[terms].to_numpy(dtype=float)
    covariance = fit.cov_params().loc[terms, terms].to_numpy(dtype=float)
    df = int(np.linalg.matrix_rank(covariance))
    if df == 0:
        return math.nan, 0, math.nan
    statistic = max(0.0, float(beta.T @ np.linalg.pinv(covariance) @ beta))
    return statistic, df, float(chi2.sf(statistic, df))


def _threshold_term_diagnostics(
    sample: pd.DataFrame,
    outcome: str,
    predictors: list[str],
    model_id: str,
    terms: list[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    exog = sm.add_constant(sample[predictors].astype(float), has_constant="add")
    for term in terms:
        if term not in predictors:
            continue
        for threshold in range(1, 5):
            binary = sample[outcome].astype(int).gt(threshold).astype(int)
            n_above = int(binary.sum())
            n_at_or_below = int(len(binary) - n_above)
            row: dict[str, object] = {
                "model_id": model_id,
                "outcome_id": outcome,
                "term": term,
                "term_label": MODEL_LABELS.get(term, term),
                "threshold": f"> {threshold}",
                "n_above": n_above,
                "n_at_or_below": n_at_or_below,
                "odds_ratio": math.nan,
                "ci_low": math.nan,
                "ci_high": math.nan,
                "converged": False,
                "reason": "",
            }
            if min(n_above, n_at_or_below) < 10:
                row["reason"] = "moins de 10 observations d'un côté du seuil"
                rows.append(row)
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=ConvergenceWarning)
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    fit = sm.Logit(binary, exog).fit(
                        disp=False, maxiter=300, cov_type="HC3"
                    )
                confidence = fit.conf_int().loc[term]
                converged = bool(fit.mle_retvals.get("converged", False))
                row.update(
                    {
                        "odds_ratio": math.exp(float(fit.params[term])),
                        "ci_low": math.exp(float(confidence.iloc[0])),
                        "ci_high": math.exp(float(confidence.iloc[1])),
                        "converged": converged,
                        "reason": "" if converged else "échec de convergence",
                    }
                )
            except Exception as error:  # pragma: no cover - data-dependent diagnostic
                row["reason"] = type(error).__name__
            rows.append(row)
    return rows


def _maximum_vif(sample: pd.DataFrame, predictors: list[str]) -> float:
    if len(predictors) <= 1:
        return 1.0
    exog = sm.add_constant(sample[predictors].astype(float), has_constant="add")
    values: list[float] = []
    for position in range(1, exog.shape[1]):
        try:
            values.append(
                float(variance_inflation_factor(exog.to_numpy(dtype=float), position))
            )
        except Exception:  # pragma: no cover - singularity is surfaced as infinity
            values.append(math.inf)
    return max(values)


def _three_category_predictions(
    fit,
    sample: pd.DataFrame,
    predictors: list[str],
    *,
    outcome_id: str,
    model_id: str,
    analysis_set: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for group_value, group_label in ((0.0, "R1 retraités"), (1.0, "R2 retraités hors Valais")):
        counterfactual = sample[predictors].astype(float).copy()
        counterfactual["group_r2"] = group_value
        probabilities = np.asarray(
            fit.model.predict(fit.params, exog=counterfactual), dtype=float
        )
        rows.append(
            {
                "analysis_set": analysis_set,
                "model_id": model_id,
                "outcome_id": outcome_id,
                "group_id": "r2_retired_outside" if group_value else "r1_retired",
                "group_label": group_label,
                "predicted_low_pct": 100 * probabilities[:, :2].sum(axis=1).mean(),
                "predicted_neutral_pct": 100 * probabilities[:, 2].mean(),
                "predicted_high_pct": 100 * probabilities[:, 3:].sum(axis=1).mean(),
                "n_standardization_sample": len(sample),
            }
        )
    return rows


def profile_outcome_models(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit one full ordinal profile model per D5 indicator.

    The main specification contains residential status, gender, Valais origin,
    retirement duration, education and income. Age classes and all H1--H4
    indicators are deliberately absent. A second specification only removes
    income to show the impact of its substantial non-response. No stepwise
    selection, interaction or dimension score is used.
    """
    summaries: list[dict[str, object]] = []
    coefficients: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    no_income_predictors = [
        predictor
        for predictor in PROFILE_MODEL_PREDICTORS
        if predictor != "income_category"
    ]
    specifications = [
        (
            "principal_avec_revenu",
            PROFILE_MODEL_PREDICTORS,
            "Profil observé avec revenu",
        ),
        (
            "sensibilite_sans_revenu",
            no_income_predictors,
            "Même profil sans revenu",
        ),
    ]

    for outcome_id, outcome_spec in OUTCOME_SPECS.items():
        for analysis_set, predictors, model_label in specifications:
            sample = frame[[outcome_id, *predictors]].dropna().copy()
            fit = _fit_ordered_model(sample, outcome_id, predictors)
            model_id = f"{outcome_id}_{analysis_set}"
            confidence = fit.conf_int()
            profile_terms = list(predictors)
            profile_wald, profile_df, profile_p = _robust_wald_block(
                fit, profile_terms
            )
            summaries.append(
                {
                    "analysis_set": analysis_set,
                    "model_id": model_id,
                    "outcome_id": outcome_id,
                    "outcome_label": outcome_spec["label"],
                    "model_label": model_label,
                    "predictors": "; ".join(predictors),
                    "n_complete": len(sample),
                    "profile_wald_chi2_hc3": profile_wald,
                    "profile_wald_df": profile_df,
                    "profile_wald_p_hc3": profile_p,
                    "aic": float(fit.aic),
                    "max_vif": _maximum_vif(sample, predictors),
                    "converged": bool(fit.mle_retvals.get("converged", False)),
                }
            )
            for term in predictors:
                term_ci = confidence.loc[term]
                coefficients.append(
                    {
                        "analysis_set": analysis_set,
                        "model_id": model_id,
                        "outcome_id": outcome_id,
                        "term": term,
                        "term_label": MODEL_LABELS.get(term, term),
                        "odds_ratio": math.exp(float(fit.params[term])),
                        "ci95_low": math.exp(float(term_ci.iloc[0])),
                        "ci95_high": math.exp(float(term_ci.iloc[1])),
                        "p_value": float(fit.pvalues[term]),
                        "n_complete": len(sample),
                    }
                )
            if analysis_set == "principal_avec_revenu":
                diagnostics.extend(
                    _threshold_term_diagnostics(
                        sample,
                        outcome_id,
                        predictors,
                        model_id,
                        list(PROFILE_MODEL_PREDICTORS),
                    )
                )

    summary_frame = pd.DataFrame(summaries)
    summary_frame["profile_wald_q_bh_two_outcomes"] = np.nan
    for analysis_set in summary_frame["analysis_set"].unique():
        selected = summary_frame["analysis_set"].eq(analysis_set)
        summary_frame.loc[
            selected, "profile_wald_q_bh_two_outcomes"
        ] = benjamini_hochberg(summary_frame.loc[selected, "profile_wald_p_hc3"])

    coefficient_frame = pd.DataFrame(coefficients)
    coefficient_frame["coefficient_q_bh_within_model"] = np.nan
    coefficient_frame["multiplicity_family"] = ""
    for (analysis_set, outcome_id), selected_rows in coefficient_frame.groupby(
        ["analysis_set", "outcome_id"], sort=False
    ):
        selected = coefficient_frame.index.isin(selected_rows.index)
        coefficient_frame.loc[
            selected, "coefficient_q_bh_within_model"
        ] = benjamini_hochberg(coefficient_frame.loc[selected, "p_value"])
        coefficient_frame.loc[selected, "multiplicity_family"] = (
            f"{len(selected_rows)} coefficients du modèle {outcome_id} ({analysis_set})"
        )
    return summary_frame, coefficient_frame, pd.DataFrame(diagnostics)


def sample_flow(data: pd.DataFrame, masks: dict[str, pd.Series], items: pd.DataFrame) -> pd.DataFrame:
    complete = data[CORE_COLUMNS["progress"]].map(
        lambda value: (normalize_text(value) or "").startswith("termine")
    )
    retired = data[CORE_COLUMNS["professional_status"]].eq("Retraité")
    r2_outside = data[CORE_COLUMNS["status3"]].eq("Secondaire_hors_Valais")
    r1 = data[CORE_COLUMNS["status2"]].eq("Principale")
    former_r2 = data[RESIDENTIAL_COLUMNS["r1_former_secondary"]].map(yes_no)
    r1_retired_before_history = complete & retired & r1
    r2_outcome = harmonized_series(
        data, masks["r2_retired_outside"], items, "ER03", "r2_retired_outside"
    )
    r1_outcome = harmonized_series(data, masks["r1_retired"], items, "ER03", "r1_retired")
    rows = [
        ("Export nettoyé", "Tous", len(data), "Questionnaires disponibles après nettoyage Sphinx"),
        ("Questionnaires complets", "Tous", int(complete.sum()), "PROGRESSION commence par Terminé"),
        ("Statut R2 hors Valais", "R2", int(r2_outside.sum()), "Avant complétude et retraite"),
        ("R2 hors Valais complets", "R2", int((r2_outside & complete).sum()), ""),
        (
            "R2 hors Valais complets et retraités",
            "R2 focal",
            int(masks["r2_retired_outside"].sum()),
            "Population focale",
        ),
        (
            "R2 avec évaluation intergroupe valide",
            "R2 focal",
            int(r2_outcome.notna().sum()),
            "Énoncé de cohabitation avec les résidents principaux, catégories 1–5",
        ),
        ("Statut R1", "R1", int(r1.sum()), "Avant complétude et retraite"),
        ("R1 complets", "R1", int((r1 & complete).sum()), "Avant restriction à la retraite"),
        (
            "R1 complets et retraités",
            "R1",
            int(r1_retired_before_history.sum()),
            "Avant contrôle de l'ancien statut R2",
        ),
        (
            "R1 retraités déclarant un ancien statut R2",
            "Exclusion R1",
            int((r1_retired_before_history & former_r2.eq(1)).sum()),
            "Exclus du comparateur principal",
        ),
        (
            "R1 retraités sans réponse sur l'ancien statut R2",
            "Exclusion R1",
            int((r1_retired_before_history & former_r2.isna()).sum()),
            "Exclus: absence de preuve d'un historique non-R2",
        ),
        (
            "R1 retraités sans ancien statut R2 déclaré dans la commune",
            "R1 comparateur",
            int(masks["r1_retired"].sum()),
            "Réponse explicite « Non »; comparateur principal",
        ),
        (
            "R1 retraités avec évaluation intergroupe valide",
            "R1 comparateur",
            int(r1_outcome.notna().sum()),
            "Énoncé de cohabitation avec les résidents secondaires, catégories 1–5",
        ),
        (
            "Retraités dans l'export nettoyé",
            "Tous",
            int(retired.sum()),
            "Avant restriction résidentielle et complétude",
        ),
    ]
    return pd.DataFrame(rows, columns=["stage", "group", "n", "definition"])


def sample_profile(data: pd.DataFrame, masks: dict[str, pd.Series]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_id in ["r2_retired_outside", "r1_retired"]:
        subset = data.loc[masks[group_id]]
        total = len(subset)
        variables: list[
            tuple[str, pd.Series, list[str], pd.Series, set[str]]
        ] = []
        gender_raw = subset[CORE_COLUMNS["gender"]]
        gender = gender_raw.map(
            lambda value: value
            if value in {"Homme", "Femme"}
            else "Autre ou non-réponse"
        )
        variables.append(
            (
                "Genre déclaré",
                gender,
                ["Femme", "Homme", "Autre ou non-réponse"],
                gender_raw.notna(),
                {"Autre ou non-réponse"},
            )
        )
        origin_source = "Etes-vous_originaire_du_Valais"
        origin = subset[origin_source].fillna("Non-réponse").astype(str)
        variables.append(
            (
                "Originaire du Valais",
                origin,
                ["Oui", "Non", "Non-réponse"],
                origin.isin(["Oui", "Non"]),
                {"Non-réponse"},
            )
        )

        education = (
            subset[SOCIODEMOGRAPHIC_COLUMNS["education"]]
            .fillna("Non-réponse")
            .astype(str)
        )
        variables.append(
            (
                "Niveau de formation",
                education,
                [
                    "Obligatoire",
                    "Secondaire (lycée, maturité, formation professionnelle)",
                    "Tertiaire (bachelor, master, doctorat)",
                    "Non-réponse",
                ],
                education.ne("Non-réponse"),
                {"Non-réponse"},
            )
        )
        nationality = (
            subset[SOCIODEMOGRAPHIC_COLUMNS["nationality"]]
            .fillna("Non-réponse")
            .astype(str)
        )
        variables.append(
            (
                "Nationalité déclarée",
                nationality,
                ["Suisse", "Autre", "Non-réponse"],
                nationality.ne("Non-réponse"),
                {"Non-réponse"},
            )
        )
        profession = (
            subset[SOCIODEMOGRAPHIC_COLUMNS["former_profession"]]
            .fillna("Non-réponse")
            .astype(str)
        )
        variables.append(
            (
                "Dernière profession",
                profession,
                [
                    "Cadre supérieur / Haut fonctionnaire",
                    "Cadre moyen",
                    "Collaborateur / Fonctionnaire",
                    "Indépendant / Professions libérales",
                    "Personne au foyer",
                    "Sans emploi",
                    "Autre",
                    "Non-réponse",
                ],
                profession.ne("Non-réponse"),
                {"Non-réponse"},
            )
        )
        income = (
            subset[SOCIODEMOGRAPHIC_COLUMNS["household_income"]]
            .fillna("Non-réponse")
            .astype(str)
        )
        income_unavailable = {"Ne souhaite pas répondre", "Non-réponse"}
        variables.append(
            (
                "Revenu mensuel brut du ménage",
                income,
                [
                    "Moins de 4’000 CHF",
                    "4’000–5’999 CHF",
                    "6’000–7’999 CHF",
                    "8’000–11’999 CHF",
                    "12’000 CHF ou plus",
                    "Ne souhaite pas répondre",
                    "Non-réponse",
                ],
                ~income.isin(income_unavailable),
                income_unavailable,
            )
        )

        for variable, series, categories, valid_mask, unavailable in variables:
            valid_n = int(valid_mask.sum())
            for category in categories:
                count = int(series.eq(category).sum())
                rows.append(
                    {
                        "group_id": group_id,
                        "group_label": GROUP_LABELS[group_id],
                        "eligible_n": total,
                        "variable": variable,
                        "category": category,
                        "count": count,
                        "percent": 100 * count / total if total else math.nan,
                        "valid_n": valid_n,
                        "percent_valid": (
                            100 * count / valid_n
                            if valid_n and category not in unavailable
                            else math.nan
                        ),
                    }
                )
    return pd.DataFrame(rows)


def nationality_summary(profile: pd.DataFrame) -> pd.DataFrame:
    """Return an aggregate nationality table for each article group and pooled."""
    nationality = profile.loc[
        profile["variable"].eq("Nationalité déclarée"),
        [
            "group_id",
            "group_label",
            "eligible_n",
            "category",
            "count",
            "percent",
            "valid_n",
            "percent_valid",
        ],
    ].copy()
    nationality = nationality.rename(columns={"percent": "percent_eligible"})

    pooled_counts = nationality.groupby("category", sort=False)["count"].sum()
    pooled_eligible_n = int(
        nationality.groupby("group_id", sort=False)["eligible_n"].first().sum()
    )
    pooled_valid_n = int(
        nationality.groupby("group_id", sort=False)["valid_n"].first().sum()
    )
    pooled_rows: list[dict[str, object]] = []
    for category in ["Suisse", "Autre", "Non-réponse"]:
        count = int(pooled_counts.get(category, 0))
        pooled_rows.append(
            {
                "group_id": "pooled_article_sample",
                "group_label": "Ensemble de l’échantillon de l’article",
                "eligible_n": pooled_eligible_n,
                "category": category,
                "count": count,
                "percent_eligible": 100 * count / pooled_eligible_n,
                "valid_n": pooled_valid_n,
                "percent_valid": (
                    100 * count / pooled_valid_n
                    if category != "Non-réponse"
                    else np.nan
                ),
            }
        )
    return pd.concat(
        [nationality, pd.DataFrame(pooled_rows)], ignore_index=True
    )


def nationality_figure(summary: pd.DataFrame, output_dir: Path) -> None:
    """Plot valid nationality percentages and disclose missing responses."""
    group_order = [
        "r2_retired_outside",
        "r1_retired",
        "pooled_article_sample",
    ]
    short_labels = {
        "r2_retired_outside": "R2 retraités hors Valais",
        "r1_retired": "R1 retraités sans ancien statut R2",
        "pooled_article_sample": "Ensemble de l’échantillon",
    }
    valid = summary.loc[summary["category"].isin(["Suisse", "Autre"])].copy()
    pivot = (
        valid.pivot(index="group_id", columns="category", values="percent_valid")
        .reindex(group_order)
        .fillna(0.0)
    )
    details = (
        summary.drop_duplicates("group_id")
        .set_index("group_id")
        .reindex(group_order)
    )
    missing = (
        summary.loc[summary["category"].eq("Non-réponse")]
        .set_index("group_id")["count"]
        .reindex(group_order)
        .fillna(0)
        .astype(int)
    )

    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    y = np.arange(len(group_order))
    left = np.zeros(len(group_order))
    colors = {"Suisse": "#2f6688", "Autre": "#c9783a"}
    for category in ["Suisse", "Autre"]:
        values = pivot[category].to_numpy(float)
        bars = ax.barh(
            y,
            values,
            left=left,
            height=0.58,
            color=colors[category],
            edgecolor="white",
            linewidth=1.5,
            label=category,
        )
        for idx, (bar, value) in enumerate(zip(bars, values, strict=True)):
            ax.text(
                left[idx] + value / 2,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.1f} %".replace(".", ","),
                ha="center",
                va="center",
                color="white",
                fontsize=12,
                fontweight="bold",
            )
        left += values

    ax.set_yticks(y, [short_labels[group] for group in group_order])
    ax.invert_yaxis()
    ax.set_xlim(0, 118)
    ax.set_xlabel("Pourcentage parmi les réponses valides")
    ax.set_title(
        "Nationalité déclarée des répondants de l’article\n"
        "Les données distinguent uniquement « Suisse » et « Autre ».",
        loc="left",
        fontweight="bold",
        fontsize=15,
        pad=16,
    )
    for idx, group_id in enumerate(group_order):
        ax.text(
            101.0,
            idx,
            f"n valide = {int(details.loc[group_id, 'valid_n'])}\n"
            f"non-réponse = {int(missing.loc[group_id])}",
            va="center",
            fontsize=10.5,
            color="#333333",
        )
    ax.legend(
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.42, -0.30),
        frameon=False,
    )
    ax.grid(axis="x", alpha=0.18)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    for suffix in ["png", "pdf"]:
        fig.savefig(
            output_dir / f"fig_nationality_distribution.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def retirement_duration_summary(
    data: pd.DataFrame,
    masks: dict[str, pd.Series],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_id in ["r2_retired_outside", "r1_retired"]:
        subset = data.loc[masks[group_id]]
        raw = subset[SOCIODEMOGRAPHIC_COLUMNS["retirement_year"]]
        duration = pd.Series(
            [
                retirement_duration(value, age)
                for value, age in zip(raw, subset[CORE_COLUMNS["age"]], strict=True)
            ],
            index=subset.index,
            dtype=float,
        )
        valid = duration.dropna()
        raw_nonmissing_n = int(pd.to_numeric(raw, errors="coerce").notna().sum())
        rows.append(
            {
                "group_id": group_id,
                "group_label": GROUP_LABELS[group_id],
                "eligible_n": len(subset),
                "raw_nonmissing_n": raw_nonmissing_n,
                "valid_n": len(valid),
                "invalid_n": raw_nonmissing_n - len(valid),
                "median_years": quantile_or_nan(valid, 0.5),
                "q1_years": quantile_or_nan(valid, 0.25),
                "q3_years": quantile_or_nan(valid, 0.75),
            }
        )
    return pd.DataFrame(rows)


def residential_markers_summary(
    data: pd.DataFrame,
    masks: dict[str, pd.Series],
    items: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_id in ["r2_retired_outside", "r1_retired"]:
        total = int(masks[group_id].sum())
        for indicator_id in ["TR01", "TR05", "TR06"]:
            values = harmonized_series(data, masks[group_id], items, indicator_id, group_id)
            valid = pd.to_numeric(values, errors="coerce").dropna()
            if indicator_id == "TR01":
                count = int(valid.eq(1).sum())
                low, high = wilson_interval(count, len(valid))
                statistic = "Oui"
                value = 100 * count / len(valid) if len(valid) else math.nan
                ci_low, ci_high, unit = 100 * low, 100 * high, "%"
            else:
                count = int(valid.isin([4, 5]).sum())
                low, high = wilson_interval(count, len(valid))
                statistic = "Accord (4–5)"
                value = 100 * count / len(valid) if len(valid) else math.nan
                ci_low, ci_high, unit = 100 * low, 100 * high, "%"
            rows.append(
                {
                    "group_id": group_id,
                    "group_label": GROUP_LABELS[group_id],
                    "indicator_id": indicator_id,
                    "indicator_label": indicator_row(items, indicator_id)["indicator_label"],
                    "statistic": statistic,
                    "category": "",
                    "valid_n": len(valid),
                    "eligible_n": total,
                    "value": value,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "unit": unit,
                }
            )
        duration = harmonized_series(data, masks[group_id], items, "TR02", group_id)
        duration = pd.to_numeric(duration, errors="coerce").dropna()
        rows.append(
            {
                "group_id": group_id,
                "group_label": GROUP_LABELS[group_id],
                "indicator_id": "TR02",
                "indicator_label": indicator_row(items, "TR02")["indicator_label"],
                "statistic": "Médiane [Q1–Q3]",
                "category": "Acquisition R2" if group_id == "r2_retired_outside" else "Installation R1",
                "valid_n": len(duration),
                "eligible_n": total,
                "value": quantile_or_nan(duration, 0.5),
                "ci_low": quantile_or_nan(duration, 0.25),
                "ci_high": quantile_or_nan(duration, 0.75),
                "unit": "années écoulées",
            }
        )
    for indicator_id, labels in [("TR03", FREQUENCY_LABELS)]:
        values = harmonized_series(
            data,
            masks["r2_retired_outside"],
            items,
            indicator_id,
            "r2_retired_outside",
        )
        valid = pd.to_numeric(values, errors="coerce").dropna()
        for level, label in labels.items():
            count = int(valid.eq(level).sum())
            rows.append(
                {
                    "group_id": "r2_retired_outside",
                    "group_label": GROUP_LABELS["r2_retired_outside"],
                    "indicator_id": indicator_id,
                    "indicator_label": indicator_row(items, indicator_id)["indicator_label"],
                    "statistic": "Distribution",
                    "category": label,
                    "valid_n": len(valid),
                    "eligible_n": int(masks["r2_retired_outside"].sum()),
                    "value": 100 * count / len(valid) if len(valid) else math.nan,
                    "ci_low": math.nan,
                    "ci_high": math.nan,
                    "unit": "%",
                }
            )
    for group_id in ["r2_retired_outside", "r1_retired"]:
        tenure = harmonized_series(data, masks[group_id], items, "TR04", group_id)
        valid_tenure = tenure.dropna().astype(str)
        for category in ["Propriétaire", "Locataire", "Autre"]:
            count = int(valid_tenure.eq(category).sum())
            low, high = wilson_interval(count, len(valid_tenure))
            rows.append(
                {
                    "group_id": group_id,
                    "group_label": GROUP_LABELS[group_id],
                    "indicator_id": "TR04",
                    "indicator_label": indicator_row(items, "TR04")["indicator_label"],
                    "statistic": "Distribution",
                    "category": category,
                    "valid_n": len(valid_tenure),
                    "eligible_n": int(masks[group_id].sum()),
                    "value": 100 * count / len(valid_tenure) if len(valid_tenure) else math.nan,
                    "ci_low": 100 * low,
                    "ci_high": 100 * high,
                    "unit": "%",
                }
            )

    season_raw = data.loc[
        masks["r2_retired_outside"], source_for(items, "TR07", "r2_retired_outside")
    ]
    season_valid = season_raw.dropna()
    season_targets = ["Hiver", "Printemps", "Été", "Automne"]
    for category in [*season_targets, "Quatre saisons"]:
        if category == "Quatre saisons":
            count = int(
                season_valid.map(
                    lambda value: all(
                        normalize_text(target)
                        in {normalize_text(choice) for choice in split_choices(value)}
                        for target in season_targets
                    )
                ).sum()
            )
            statistic = "Toutes les saisons"
        else:
            count = int(
                season_valid.map(lambda value: selected_choice(value, category)).sum()
            )
            statistic = "Saison mentionnée"
        low, high = wilson_interval(count, len(season_valid))
        rows.append(
            {
                "group_id": "r2_retired_outside",
                "group_label": GROUP_LABELS["r2_retired_outside"],
                "indicator_id": "TR07",
                "indicator_label": indicator_row(items, "TR07")["indicator_label"],
                "statistic": statistic,
                "category": category,
                "valid_n": len(season_valid),
                "eligible_n": int(masks["r2_retired_outside"].sum()),
                "value": 100 * count / len(season_valid) if len(season_valid) else math.nan,
                "ci_low": 100 * low,
                "ci_high": 100 * high,
                "unit": "%",
            }
        )

    for indicator_id in ["TR08", "TR10"]:
        values = harmonized_series(
            data, masks["r1_retired"], items, indicator_id, "r1_retired"
        )
        valid = pd.to_numeric(values, errors="coerce").dropna()
        count = int(valid.eq(1).sum())
        low, high = wilson_interval(count, len(valid))
        rows.append(
            {
                "group_id": "r1_retired",
                "group_label": GROUP_LABELS["r1_retired"],
                "indicator_id": indicator_id,
                "indicator_label": indicator_row(items, indicator_id)["indicator_label"],
                "statistic": "Oui",
                "category": "",
                "valid_n": len(valid),
                "eligible_n": int(masks["r1_retired"].sum()),
                "value": 100 * count / len(valid) if len(valid) else math.nan,
                "ci_low": 100 * low,
                "ci_high": 100 * high,
                "unit": "%",
            }
        )
    return pd.DataFrame(rows)


def territory_use_summary(
    data: pd.DataFrame,
    masks: dict[str, pd.Series],
    items: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_id in ["r2_retired_outside", "r1_retired"]:
        total = int(masks[group_id].sum())
        for indicator_id in ["TU01", "TU02", "TU05"]:
            values = harmonized_series(data, masks[group_id], items, indicator_id, group_id)
            valid = pd.to_numeric(values, errors="coerce").dropna()
            count = int(valid.eq(1).sum())
            low, high = wilson_interval(count, len(valid))
            rows.append(
                {
                    "group_id": group_id,
                    "group_label": GROUP_LABELS[group_id],
                    "indicator_id": indicator_id,
                    "indicator_label": indicator_row(items, indicator_id)["indicator_label"],
                    "statistic": "Oui",
                    "valid_n": len(valid),
                    "eligible_n": total,
                    "value": 100 * count / len(valid) if len(valid) else math.nan,
                    "q1": 100 * low,
                    "q3": 100 * high,
                    "unit": "% [IC95 %]",
                }
            )
        for indicator_id in ["TU03", "TU04"]:
            values = harmonized_series(data, masks[group_id], items, indicator_id, group_id)
            valid = pd.to_numeric(values, errors="coerce").dropna()
            rows.append(
                {
                    "group_id": group_id,
                    "group_label": GROUP_LABELS[group_id],
                    "indicator_id": indicator_id,
                    "indicator_label": indicator_row(items, indicator_id)["indicator_label"],
                    "statistic": "Médiane [Q1–Q3]",
                    "valid_n": len(valid),
                    "eligible_n": total,
                    "value": quantile_or_nan(valid, 0.5),
                    "q1": quantile_or_nan(valid, 0.25),
                    "q3": quantile_or_nan(valid, 0.75),
                    "unit": "modalités sélectionnées",
                }
            )
    return pd.DataFrame(rows)


def territory_composition_summary(
    data: pd.DataFrame,
    masks: dict[str, pd.Series],
) -> pd.DataFrame:
    """Describe organization types and the first-ranked ordered responses."""
    rows: list[dict[str, object]] = []
    for group_id in ["r2_retired_outside", "r1_retired"]:
        eligible_n = int(masks[group_id].sum())
        for dimension_id, spec in TERRITORY_COMPOSITION_SPECS.items():
            raw = data.loc[masks[group_id], str(spec["source"])]
            valid = raw.dropna()
            first_ranked = valid.map(first_choice) if spec["mode"] == "first" else None
            for category_id, category_label in spec["categories"]:
                if spec["mode"] == "multiple":
                    count = int(
                        valid.map(
                            lambda value: selected_choice(value, str(category_label))
                        ).sum()
                    )
                    statistic = "Mention"
                else:
                    assert first_ranked is not None
                    target = normalize_text(category_label)
                    count = int(first_ranked.map(normalize_text).eq(target).sum())
                    statistic = "Premier rang"
                low, high = wilson_interval(count, len(valid))
                rows.append(
                    {
                        "group_id": group_id,
                        "group_label": GROUP_LABELS[group_id],
                        "dimension_id": dimension_id,
                        "dimension_label": spec["label"],
                        "statistic": statistic,
                        "category_id": category_id,
                        "category_label": category_label,
                        "count": count,
                        "valid_n": len(valid),
                        "eligible_n": eligible_n,
                        "percent": 100 * count / len(valid) if len(valid) else math.nan,
                        "ci_low": 100 * low,
                        "ci_high": 100 * high,
                    }
                )
    return pd.DataFrame(rows)


def missingness_table(
    data: pd.DataFrame,
    masks: dict[str, pd.Series],
    items: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_id in ["r2_retired_outside", "r1_retired"]:
        total = int(masks[group_id].sum())
        for item in items.itertuples(index=False):
            source = item.r2_source if group_id == "r2_retired_outside" else item.r1_source
            if not source:
                continue
            raw = data.loc[masks[group_id], source]
            if str(item.scale).startswith("likert"):
                states = likert_states(raw)
                valid_n = states["valid_n"]
                sans_n = states["sans_avis_n"]
                blank_n = states["blank_n"]
                other_n = states["other_invalid_n"]
            else:
                values = harmonized_series(
                    data, masks[group_id], items, item.indicator_id, group_id
                )
                valid_n = int(values.notna().sum())
                sans_n = 0
                blank_n = total - valid_n
                other_n = 0
            rows.append(
                {
                    "domain": item.domain,
                    "indicator_id": item.indicator_id,
                    "indicator_label": item.indicator_label,
                    "group_id": group_id,
                    "group_label": GROUP_LABELS[group_id],
                    "eligible_n": total,
                    "valid_n": valid_n,
                    "sans_avis_n": sans_n,
                    "blank_or_unavailable_n": blank_n,
                    "other_invalid_n": other_n,
                    "valid_pct": 100 * valid_n / total if total else math.nan,
                }
            )
    return pd.DataFrame(rows)


def public_measurement_matrix(
    items: pd.DataFrame, labels: dict[str, str]
) -> pd.DataFrame:
    """Expose complete questionnaire wording without technical source names."""
    rows: list[dict[str, object]] = []
    for item in items.itertuples(index=False):
        r2_wording = str(labels.get(item.r2_source, "")).strip() if item.r2_source else ""
        r1_wording = str(labels.get(item.r1_source, "")).strip() if item.r1_source else ""
        if r2_wording and r1_wording and r2_wording != r1_wording:
            wording = f"R2 : {r2_wording} | R1 : {r1_wording}"
        else:
            wording = r2_wording or r1_wording
        if item.indicator_id == "TU04":
            wording = (
                f"{wording} [indicateur dérivé : nombre de modalités d’activités sélectionnées]"
            )
        elif item.indicator_id == "TU05":
            wording = (
                f"{wording} [indicateur dérivé : sélection de la modalité « Participation à des manifestations locales (exemple : loto) »]"
            )
        rows.append(
            {
                "domain": item.domain,
                "indicator_id": item.indicator_id,
                "indicator_label": item.indicator_label,
                "question_wording": wording,
                "scale": item.scale,
                "comparability": item.comparability,
                "analysis_role": item.analysis_role,
                "missing_policy": item.missing_policy,
                "interpretive_limit": item.interpretive_limit,
            }
        )
    return pd.DataFrame(rows)


def conceptual_figure(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 7.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def box(
        x: float,
        y: float,
        w: float,
        h: float,
        text: str,
        face: str,
        edge: str = "#24364b",
        fontsize: float = 10.5,
    ) -> None:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.015",
            linewidth=1.4,
            edgecolor=edge,
            facecolor=face,
        )
        ax.add_patch(patch)
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            wrap=True,
        )

    box(
        0.045,
        0.39,
        0.25,
        0.20,
        "STATUT RÉSIDENTIEL\n\nR2 retraités hors Valais\nR1 retraités sans ancien statut R2\ndéclaré dans la commune",
        "#eef3f7",
        fontsize=8.5,
    )
    dimensions = [
        ("D1 — Usages et sociabilités territoriales", 0.41, 0.76, "#edf6ed", "H1"),
        ("D2 — Rôle perçu dans le vivre-ensemble", 0.41, 0.62, "#edf6ed", "H2"),
        ("D3 — Vision du territoire", 0.41, 0.48, "#edf6ed", "H3"),
        ("D4 — Ancrage et légitimité territoriale\nattachement · chez-soi · poids décisionnel", 0.41, 0.32, "#edf6ed", "H4"),
        ("D5 — Perception des relations intergroupes\nméfiance · cohabitation avec l’autre groupe", 0.41, 0.14, "#f8eadf", "H5"),
    ]
    for label, x, y, face, hypothesis in dimensions:
        box(x, y, 0.53, 0.105, label, face, "#456b46", fontsize=10.5)
        ax.annotate(
            "",
            xy=(x, y + 0.052),
            xytext=(0.295, 0.49),
            arrowprops={"arrowstyle": "->", "color": "#77838f", "linewidth": 0.9},
        )
        ax.text(
            0.355,
            y + 0.052,
            hypothesis,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
        )
    ax.text(
        0.5,
        0.97,
        "Modèle comparatif des cinq dimensions de la coprésence territoriale",
        ha="center",
        va="top",
        fontsize=15,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.015,
        "H1 à H5 testent une hypothèse de non-différence entre statuts. Les flèches représentent des comparaisons, non des relations causales.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#4d4d4d",
    )
    fig.tight_layout()
    for suffix in ["png", "pdf"]:
        fig.savefig(
            output_dir / f"fig_01_conceptual_framework.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def r2_profile_figure(profile: pd.DataFrame, output_dir: Path) -> None:
    plot = profile.copy()
    plot = plot.iloc[::-1].reset_index(drop=True)
    domain_colors = {
        "Configuration résidentielle et marqueurs biographiques": "#4c78a8",
        "Rôle perçu dans le vivre-ensemble": "#59a14f",
        "Vision du territoire": "#f28e2b",
        "Ancrage et légitimité territoriale": "#b279a2",
        "Perception des relations intergroupes": "#e15759",
    }
    colors = [domain_colors.get(value, "#777777") for value in plot["domain"]]
    y = np.arange(len(plot))
    estimate = plot["agree_pct"].to_numpy(float)
    lower = estimate - plot["agree_ci_low_pct"].to_numpy(float)
    upper = plot["agree_ci_high_pct"].to_numpy(float) - estimate
    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    for idx in range(len(plot)):
        ax.hlines(y[idx], 0, estimate[idx], color=colors[idx], linewidth=2, alpha=0.8)
        ax.scatter(
            estimate[idx],
            y[idx],
            xerr=np.array([[lower[idx]], [upper[idx]]]),
            fmt="o",
            color=colors[idx],
            ecolor=colors[idx],
            capsize=3,
            markersize=6,
        )
    labels = [textwrap.fill(value, 42) for value in plot["indicator_label"]]
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Part d'accord (catégories 4–5), % et IC95 % de Wilson")
    ax.set_title("Profil descriptif des R2 retraités hors Valais")
    ax.grid(axis="x", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(
        0,
        -0.14,
        "Les items de vision et de légitimité ne sont pas orientés comme favorables ou défavorables.",
        transform=ax.transAxes,
        fontsize=9,
    )
    fig.tight_layout()
    for suffix in ["png", "pdf"]:
        fig.savefig(output_dir / f"fig_02_r2_dimension_profile.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def group_difference_figure(dimension_tests: pd.DataFrame, output_dir: Path) -> None:
    """Deprecated wrapper retained for callers from intermediate revisions."""
    dimension_difference_figure(dimension_tests, output_dir)
    return

    plot = dimension_tests.iloc[::-1].reset_index(drop=True).copy()
    y = np.arange(len(plot))
    estimate = plot["root_mean_squared_cliffs_delta"].to_numpy(float)
    lower = np.zeros_like(estimate)
    upper = np.zeros_like(estimate)
    plot["primary_comparison"] = plot["dimension_id"].eq("D5")
    plot["indicator_label"] = plot["dimension_label"]
    colors = [
        "#c56a2d" if value == "D5" else "#3c6e8f"
        for value in plot["dimension_id"]
    ]
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for idx in range(len(plot)):
        ax.errorbar(
            estimate[idx],
            y[idx],
            xerr=np.array([[lower[idx]], [upper[idx]]]),
            fmt="D" if plot.loc[idx, "primary_comparison"] else "o",
            color=colors[idx],
            ecolor=colors[idx],
            capsize=3,
            markersize=6,
        )
    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(y, [textwrap.fill(value, 42) for value in plot["indicator_label"]])
    ax.set_xlabel("Différence R2 – R1 retraités dans la part d'accord, points de % (IC95 %)")
    ax.set_title("Contrastes descriptifs entre groupes résidentiels")
    ax.grid(axis="x", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(
        0,
        -0.17,
        "Le losange rouge indique la comparaison principale en miroir de la cohabitation intergroupe.\n"
        "Les tests principaux utilisent les cinq catégories ordinales; cette figure montre les parts d'accord.",
        transform=ax.transAxes,
        fontsize=9,
    )
    fig.tight_layout()
    for suffix in ["png", "pdf"]:
        fig.savefig(output_dir / f"fig_02_group_dimension_contrasts.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def dimension_difference_figure(
    dimension_tests: pd.DataFrame, output_dir: Path
) -> None:
    """Plot the five global contrasts, emphasizing D1--D4 versus D5."""
    plot = dimension_tests.iloc[::-1].reset_index(drop=True).copy()
    y = np.arange(len(plot))
    estimate = plot["root_mean_squared_cliffs_delta"].to_numpy(float)
    colors = [
        "#c56a2d" if value == "D5" else "#3c6e8f"
        for value in plot["dimension_id"]
    ]
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for idx in range(len(plot)):
        ax.hlines(
            y[idx], 0, estimate[idx], color=colors[idx], linewidth=2, alpha=0.8
        )
        ax.scatter(
            estimate[idx],
            y[idx],
            s=85 if plot.loc[idx, "dimension_id"] == "D5" else 65,
            marker="D" if plot.loc[idx, "dimension_id"] == "D5" else "o",
            color=colors[idx],
            zorder=3,
        )
        p_value = float(plot.loc[idx, "p_value"])
        p_label = (
            "p < 0,001"
            if p_value < 0.001
            else f"p = {p_value:.3f}".replace(".", ",")
        )
        decision = (
            "différence détectée"
            if p_value < 0.05
            else "absence de différence détectable"
        )
        ax.text(
            estimate[idx] + 0.012,
            y[idx],
            f"{p_label} · {decision}",
            va="center",
            fontsize=9,
            color="#333333",
        )
    labels = [
        textwrap.fill(f"{row.dimension_id} — {row.dimension_label}", 38)
        for row in plot.itertuples(index=False)
    ]
    ax.set_yticks(y, labels)
    ax.set_xlim(0, max(0.42, float(estimate.max()) + 0.23))
    ax.set_xlabel("Amplitude globale du contraste (RMS des |δ de Cliff|)")
    ax.set_title("Contrastes globaux entre les deux profils résidentiels retraités")
    ax.grid(axis="x", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(
        0,
        -0.20,
        "La statistique agrège les indicateurs de chaque dimension sans créer de score individuel.\n"
        "Le losange orange repère D5; un non-rejet ne démontre pas l'équivalence.",
        transform=ax.transAxes,
        fontsize=9,
    )
    fig.tight_layout()
    for suffix in ["png", "pdf"]:
        fig.savefig(
            output_dir / f"fig_02_group_dimension_contrasts.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def cohabitation_figure(distributions: pd.DataFrame, output_dir: Path) -> None:
    combinations = [
        ("ER01", "r2_retired_outside", "Méfiance perçue — R2"),
        ("ER01", "r1_retired", "Méfiance perçue — R1"),
        ("ER03", "r2_retired_outside", "Cohabitation — R2 évalue R1"),
        ("ER03", "r1_retired", "Cohabitation — R1 évalue R2"),
    ]
    rows: list[dict[str, object]] = []
    critical_rows: list[dict[str, object]] = []
    for indicator_id, group_id, label in combinations:
        selected = distributions.loc[
            distributions["indicator_id"].eq(indicator_id)
            & distributions["group_id"].eq(group_id)
            & distributions["response_order"].between(1, 5)
        ]
        for row in selected.itertuples(index=False):
            rows.append(
                {
                    "label": label,
                    "response_order": int(row.response_order),
                    "pct_valid": float(row.pct_valid),
                }
            )
        counts = {
            int(row.response_order): int(row.count)
            for row in selected.itertuples(index=False)
        }
        valid_n = int(selected["valid_n"].iloc[0])
        critical_n = (
            counts.get(4, 0) + counts.get(5, 0)
            if indicator_id == "ER01"
            else counts.get(1, 0) + counts.get(2, 0)
        )
        ci_low, ci_high = wilson_interval(critical_n, valid_n)
        critical_rows.append(
            {
                "label": label,
                "estimate": 100 * critical_n / valid_n,
                "ci_low": 100 * ci_low,
                "ci_high": 100 * ci_high,
                "n": critical_n,
                "valid_n": valid_n,
            }
        )
    pivot = pd.DataFrame(rows).pivot(
        index="label", columns="response_order", values="pct_valid"
    ).reindex([value[2] for value in combinations])
    labels = pivot.index.tolist()
    colors = ["#6b6b6b", "#9b9b9b", "#d9d9d9", "#7aa6c2", "#2f6688"]
    hatches = ["///", "xx", "..", "\\\\", ""]
    fig, (ax, ax_critical) = plt.subplots(
        2,
        1,
        figsize=(13.5, 9.8),
        gridspec_kw={"height_ratios": [2.25, 1.0], "hspace": 0.50},
    )
    left = np.zeros(len(pivot))
    for level in range(1, 6):
        values = pivot[level].to_numpy(float)
        bars = ax.barh(
            labels,
            values,
            left=left,
            height=0.62,
            color=colors[level - 1],
            edgecolor="white",
            linewidth=1.4,
            label=LIKERT_LABELS[level],
        )
        for bar in bars:
            bar.set_hatch(hatches[level - 1])
        for idx, value in enumerate(values):
            label = f"{value:.1f} %".replace(".", ",")
            if value >= 6.5:
                text_color = "white" if level in {1, 2, 5} else "#111111"
                ax.text(
                    left[idx] + value / 2,
                    idx,
                    label,
                    ha="center",
                    va="center",
                    fontsize=11,
                    fontweight="bold",
                    color=text_color,
                )
            else:
                ax.annotate(
                    label,
                    xy=(left[idx] + value / 2, idx + 0.31),
                    xytext=(0, 8 + 8 * ((level + idx) % 2)),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                    arrowprops={
                        "arrowstyle": "-",
                        "color": "#555555",
                        "lw": 0.8,
                    },
                    clip_on=False,
                )
        left += values
    ax.set_xlim(0, 100)
    ax.set_xlabel("Pourcentage des réponses valides")
    ax.set_title("A. Distribution complète des perceptions")
    ax.legend(
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        frameon=False,
        fontsize=10,
    )
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", alpha=0.2)

    critical = pd.DataFrame(critical_rows).iloc[::-1].reset_index(drop=True)
    y = np.arange(len(critical))
    estimate = critical["estimate"].to_numpy(float)
    lower = estimate - critical["ci_low"].to_numpy(float)
    upper = critical["ci_high"].to_numpy(float) - estimate
    ax_critical.errorbar(
        estimate,
        y,
        xerr=np.vstack([lower, upper]),
        fmt="o",
        color="#a24d2c",
        ecolor="#a24d2c",
        capsize=4,
        markersize=6,
    )
    for idx, row in critical.iterrows():
        ax_critical.text(
            min(49.0, float(row["ci_high"]) + 1.0),
            idx,
            f"{float(row['estimate']):.1f} %  "
            f"({int(row['n'])}/{int(row['valid_n'])})".replace(".", ","),
            va="center",
            fontsize=10.5,
            fontweight="bold",
        )
    ax_critical.set_yticks(y, critical["label"])
    ax_critical.set_xlim(0, 50)
    ax_critical.set_xlabel("Part de réponses critiques, % et IC95 % de Wilson")
    ax_critical.set_title(
        "B. Minorités critiques: méfiance déclarée ou cohabitation défavorable"
    )
    ax_critical.spines[["top", "right"]].set_visible(False)
    ax_critical.grid(axis="x", alpha=0.2)
    fig.subplots_adjust(left=0.27, right=0.98, top=0.94, bottom=0.08)
    for suffix in ["png", "pdf"]:
        fig.savefig(output_dir / f"fig_03_intergroup_response_distributions.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.6f")


def nationality_markdown_report(path: Path, summary: pd.DataFrame) -> None:
    """Write a concise reproducible narrative and display table in French."""
    valid = summary.loc[summary["category"].isin(["Suisse", "Autre"])].copy()
    missing = summary.loc[summary["category"].eq("Non-réponse")].set_index("group_id")

    def row(group_id: str, category: str) -> pd.Series:
        return valid.loc[
            valid["group_id"].eq(group_id) & valid["category"].eq(category)
        ].iloc[0]

    display_groups = [
        ("r2_retired_outside", "R2 retraités hors Valais"),
        ("r1_retired", "R1 retraités sans ancien statut R2 déclaré"),
        ("pooled_article_sample", "Ensemble de l’échantillon de l’article"),
    ]
    lines = [
        "# Nationalité déclarée des répondants de l’article",
        "",
        "La variable disponible distingue uniquement les catégories « Suisse » et « Autre »; "
        "elle ne permet pas d’identifier les pays de nationalité.",
        "",
        "Les pourcentages principaux sont calculés parmi les réponses valides. "
        "Les non-réponses sont indiquées séparément.",
        "",
        "| Groupe | Suisse, n (%) | Autre, n (%) | Réponses valides | Non-réponses |",
        "|---|---:|---:|---:|---:|",
    ]
    for group_id, label in display_groups:
        swiss = row(group_id, "Suisse")
        other = row(group_id, "Autre")
        missing_n = int(missing.loc[group_id, "count"])
        lines.append(
            f"| {label} | {int(swiss['count'])} "
            f"({float(swiss['percent_valid']):.1f} %) | {int(other['count'])} "
            f"({float(other['percent_valid']):.1f} %) | {int(swiss['valid_n'])} | "
            f"{missing_n} |".replace(".", ",")
        )
    lines.extend(
        [
            "",
            "Dans l’ensemble de l’échantillon analytique, 381 des 439 répondants ayant "
            "renseigné leur nationalité se déclarent suisses (86,8 %), contre 58 dans "
            "la catégorie « Autre » (13,2 %). La proportion de nationalité suisse est "
            "de 84,2 % chez les R2 retraités domiciliés hors Valais et de 93,0 % chez "
            "les R1 retraités sans ancien statut R2 déclaré. Ces résultats sont "
            "descriptifs et concernent l’échantillon observé.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def markdown_report(
    path: Path,
    masks: dict[str, pd.Series],
    profile: pd.DataFrame,
    comparisons: pd.DataFrame,
) -> None:
    primary = comparisons.loc[comparisons["indicator_id"].eq(PRIMARY_COMPARISON_ID)].iloc[0]
    r2_primary = profile.loc[profile["indicator_id"].eq(PRIMARY_COMPARISON_ID)].iloc[0]
    strongest = comparisons.reindex(
        comparisons["effect_estimate"].abs().sort_values(ascending=False).index
    ).head(5)
    lines = [
        "# Rapport agrégé — dimensions de la coprésence",
        "",
        "## Populations",
        "",
        f"- R2 retraités hors Valais : n = {int(masks['r2_retired_outside'].sum())}.",
        f"- R1 retraités sans ancien statut R2 déclaré dans la commune : n = {int(masks['r1_retired'].sum())}.",
        "",
        "## Évaluation principale en miroir",
        "",
        f"Chez les R2, {r2_primary['agree_pct']:.1f} % des {int(r2_primary['valid_n'])} réponses valides "
        "se situent dans les catégories d'accord. "
        f"La différence R2–R1 retraités est de {primary['agree_difference_pp']:.1f} points de pourcentage "
        f"(IC95 % {primary['agree_difference_ci_low_pp']:.1f} à {primary['agree_difference_ci_high_pp']:.1f}).",
        f"Le delta de Cliff est {primary['effect_estimate']:.3f} "
        f"(IC95 % {primary['effect_ci_low']:.3f} à {primary['effect_ci_high']:.3f}); "
        f"p = {primary['p_value']:.4f}, q-BH = {primary['q_bh']:.4f}.",
        "",
        "## Contrastes les plus amples selon leur métrique propre",
        "",
    ]
    for row in strongest.itertuples(index=False):
        lines.append(
            f"- {row.indicator_id} — {row.indicator_label}: {row.effect_metric} = "
            f"{row.effect_estimate:.3f}; IC95 % {row.effect_ci_low:.3f} à {row.effect_ci_high:.3f}; "
            f"q-BH = {row.q_bh:.4f}."
        )
    lines.extend(
        [
            "",
            "## Limites de mesure",
            "",
            "- Aucun score global ou score de dimension n'est construit.",
            "- Aucun item ne demande comment les répondants pensent être perçus par l'autre groupe.",
            "- L'année d'acquisition R2 et l'année d'installation R1 sont décrites séparément.",
            "- Toutes les sorties sont agrégées et ne contiennent aucune donnée individuelle.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(input_path: Path, output_root: Path, item_map_path: Path) -> dict[str, object]:
    data, labels, info = load_export(input_path)
    items = load_item_map(item_map_path)
    verify_schema(data, items)
    masks = group_masks(data)

    tables_dir = output_root / "tables"
    figures_dir = output_root / "figures"
    diagnostics_dir = output_root / "diagnostics"
    internal_dir = output_root / "reports" / "internal"
    for directory in [tables_dir, figures_dir, diagnostics_dir, internal_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    for filename in DEPRECATED_TABLE_OUTPUTS:
        (tables_dir / filename).unlink(missing_ok=True)
    for filename in DEPRECATED_FIGURE_OUTPUTS:
        (figures_dir / filename).unlink(missing_ok=True)

    flow = sample_flow(data, masks, items)
    profile = sample_profile(data, masks)
    nationality = nationality_summary(profile)
    retirement = retirement_duration_summary(data, masks)
    residential_markers = residential_markers_summary(data, masks, items)
    territory = territory_use_summary(data, masks, items)
    territory_composition = territory_composition_summary(data, masks)
    r2_likert = pd.DataFrame(
        [
            likert_summary(data, masks["r2_retired_outside"], items, item_id, "r2_retired_outside")
            for item_id in LIKERT_PROFILE_IDS
        ]
    )
    comparisons = compare_groups(data, masks, items, "r1_retired")
    profile_comparisons = compare_groups(
        data,
        masks,
        items,
        "r1_retired",
        indicator_ids=["TR01"],
    )
    outcome_comparisons = compare_groups(
        data,
        masks,
        items,
        "r1_retired",
        indicator_ids=[spec["indicator_id"] for spec in OUTCOME_SPECS.values()],
    ).rename(columns={"q_bh": "q_bh_two_relational_indicators"})
    dimension_tests = dimension_global_tests(data, masks, items)
    model_frame = pooled_model_frame(data, masks, items)
    (
        profile_models,
        profile_coefficients,
        proportional_odds,
    ) = profile_outcome_models(model_frame)
    distributions = full_likert_distributions(
        data,
        masks,
        items,
        ["r2_retired_outside", "r1_retired"],
    )
    relational_summary = relational_response_summary(distributions)
    missingness = missingness_table(data, masks, items)
    measurement_public = public_measurement_matrix(items, labels)

    cohab_distribution = distributions.loc[
        distributions["indicator_id"].isin(["ER01", "ER03"])
        & distributions["group_id"].isin(["r2_retired_outside", "r1_retired"])
    ].copy()

    table_map = {
        "tab_01_analysis_flow.csv": flow,
        "tab_02_sample_profile.csv": profile,
        "tab_02d_nationality.csv": nationality,
        "tab_02b_retirement_duration.csv": retirement,
        "tab_02c_profile_comparisons.csv": profile_comparisons,
        "tab_03_residential_markers.csv": residential_markers,
        "tab_04_territory_use.csv": territory,
        "tab_04b_territory_composition.csv": territory_composition,
        "tab_05_r2_dimension_profile.csv": r2_likert,
        "tab_06_group_comparisons.csv": comparisons,
        "tab_06a_dimension_global_tests.csv": dimension_tests,
        "tab_07_relational_outcome_comparisons.csv": outcome_comparisons,
        "tab_07b_relational_outcome_distributions.csv": cohab_distribution,
        "tab_07c_relational_response_summary.csv": relational_summary,
        "tab_08_profile_outcome_models.csv": profile_models,
        "tab_s01_full_likert_distributions.csv": distributions,
        "tab_s02_item_missingness.csv": missingness,
        "tab_s04_measurement_matrix.csv": measurement_public,
        "tab_s08_profile_model_coefficients.csv": profile_coefficients,
        "tab_s09_profile_model_diagnostics.csv": proportional_odds,
    }
    for filename, frame in table_map.items():
        write_csv(frame, tables_dir / filename)

    conceptual_figure(figures_dir)
    dimension_difference_figure(dimension_tests, figures_dir)
    cohabitation_figure(distributions, figures_dir)
    nationality_figure(nationality, figures_dir)
    nationality_markdown_report(
        output_root / "reports" / "nationalite_repondants_article_FR.md",
        nationality,
    )

    primary = comparisons.loc[comparisons["indicator_id"].eq(PRIMARY_COMPARISON_ID)].iloc[0]
    r1_dual = harmonized_series(
        data, masks["r1_retired"], items, "TR10", "r1_retired"
    ).dropna()
    retirement_invalid = {
        str(row.group_id): int(row.invalid_n)
        for row in retirement.itertuples(index=False)
    }
    income_availability = {
        group_id: {
            "eligible_n": int(mask.sum()),
            "valid_n": int(
                data.loc[mask, SOCIODEMOGRAPHIC_COLUMNS["household_income"]]
                .map(household_income_score)
                .notna()
                .sum()
            ),
        }
        for group_id, mask in (
            ("r2_retired_outside", masks["r2_retired_outside"]),
            ("r1_retired", masks["r1_retired"]),
        )
    }
    complete = data[CORE_COLUMNS["progress"]].map(
        lambda value: (normalize_text(value) or "").startswith("termine")
    )
    retired = data[CORE_COLUMNS["professional_status"]].eq("Retraité")
    r1_status = data[CORE_COLUMNS["status2"]].eq("Principale")
    r1_retired_before_history = complete & retired & r1_status
    former_r2 = data[RESIDENTIAL_COLUMNS["r1_former_secondary"]].map(yes_no)
    metadata: dict[str, object] = {
        "analysis_status": "post-data exploratory descriptive-comparative",
        "survey_year": SURVEY_YEAR,
        "random_seed": RANDOM_SEED,
        "bootstrap_repetitions": BOOTSTRAP_REPS,
        "populations": {key: int(mask.sum()) for key, mask in masks.items()},
        "r1_history_filter": {
            "retired_before_history_filter_n": int(r1_retired_before_history.sum()),
            "former_r2_excluded_n": int(
                (r1_retired_before_history & former_r2.eq(1)).sum()
            ),
            "history_missing_excluded_n": int(
                (r1_retired_before_history & former_r2.isna()).sum()
            ),
            "explicit_non_r2_history_included_n": int(masks["r1_retired"].sum()),
        },
        "primary_comparison": {
            "indicator_id": PRIMARY_COMPARISON_ID,
            "r2_source_concept": "R2 evaluation of cohabitation with primary residents",
            "r1_source_concept": "R1 evaluation of cohabitation with secondary residents",
            "n_r2_valid": int(primary["n_r2"]),
            "n_r1_valid": int(primary["n_comparator"]),
            "agreement_difference_percentage_points": float(primary["agree_difference_pp"]),
            "agreement_difference_ci95": [
                float(primary["agree_difference_ci_low_pp"]),
                float(primary["agree_difference_ci_high_pp"]),
            ],
            "cliffs_delta": float(primary["effect_estimate"]),
            "cliffs_delta_ci95": [
                float(primary["effect_ci_low"]),
                float(primary["effect_ci_high"]),
            ],
            "p_value": float(primary["p_value"]),
            "q_bh": float(primary["q_bh"]),
        },
        "comparison_family_size": int(len(comparisons)),
        "comparison_family_definition": "post-data exploratory after data collection",
        "comparison_family_indicator_ids": comparisons["indicator_id"].astype(str).tolist(),
        "multiplicity": f"Benjamini-Hochberg across {len(comparisons)} post-data dimension-indicator contrasts",
        "dimension_hypotheses": {
            dimension_id: {
                "hypothesis": str(spec["hypothesis"]),
                "label": str(spec["label"]),
                "indicator_ids": list(spec["indicator_ids"]),
            }
            for dimension_id, spec in DIMENSION_SPECS.items()
        },
        "dimension_global_test": {
            "method": "synchronized label permutation of mean squared item-level Cliff's deltas",
            "repetitions": PERMUTATION_REPS,
            "multiplicity": "Benjamini-Hochberg across five global dimension hypotheses",
            "score_constructed": False,
        },
        "profile_comparison_indicator_ids": profile_comparisons[
            "indicator_id"
        ].astype(str).tolist(),
        "sociodemographic_profile_models": True,
        "income_availability": income_availability,
        "recruitment_design": "nonprobability multichannel volunteer sample",
        "accessible_population_enumerated": False,
        "sampling_frame_available": False,
        "cluster_sampling_design": False,
        "design_weights": False,
        "inference_scope": "conditional on observed respondents under working independence",
        "ordered_response_rank_used": True,
        "retirement_year_invalid_counts": retirement_invalid,
        "r1_current_secondary_residence_check": {
            "valid_n": int(len(r1_dual)),
            "yes_n": int(r1_dual.eq(1).sum()),
            "missing_n": int(masks["r1_retired"].sum() - len(r1_dual)),
        },
        "global_score_constructed": False,
        "dimension_score_constructed": False,
        "active_profile_outcome_models": {
            "outcomes": {
                outcome_id: dict(spec) for outcome_id, spec in OUTCOME_SPECS.items()
            },
            "strategy": "one supplementary cumulative-logit model per outcome with status, gender, origin, retirement duration, education and income; one no-income sensitivity per outcome",
            "model_count": int(len(profile_models)),
            "primary_with_income_model_count": int(
                profile_models["analysis_set"].eq("principal_avec_revenu").sum()
            ),
            "sensitivity_without_income_model_count": int(
                profile_models["analysis_set"].eq("sensibilite_sans_revenu").sum()
            ),
            "h1_h4_indicators_excluded_from_models": True,
            "age_classes_in_model": False,
            "coefficient_covariance": "HC3",
            "profile_block_test": "joint Wald of all six predictors using HC3 covariance; Benjamini-Hochberg across the two modeled H5 outcomes",
            "coefficient_multiplicity": "Benjamini-Hochberg within each outcome-specific model",
            "threshold_diagnostic_terms": list(PROFILE_MODEL_PREDICTORS),
            "association_not_causation": True,
        },
        "relational_dimension": {
            "indicator_ids": outcome_comparisons["indicator_id"].astype(str).tolist(),
            "outcomes": list(OUTCOME_SPECS),
            "multiplicity": "Benjamini-Hochberg across the two modeled D5 outcomes",
            "score_constructed": False,
        },
        "direct_intergroup_perceptions_measured": True,
        "municipality_comparisons": False,
        "respondent_level_outputs": False,
        "source_rows": int(len(data)),
        "source_columns": int(len(data.columns)),
        "information_sheet_rows": int(len(info)),
        "item_map_rows": int(len(items)),
        "privacy": "aggregate outputs only; no identifiers, free text, timestamps, or municipality tables",
    }
    (diagnostics_dir / "descriptive_comparative_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    quality = pd.DataFrame(
        [
            {
                "check": "unique_technical_key",
                "value": bool(not data[CORE_COLUMNS["key"]].duplicated().any()),
                "detail": "CLE unique in cleaned export",
            },
            {
                "check": "unexpected_likert_categories",
                "value": int(missingness["other_invalid_n"].sum()),
                "detail": "sum across group-item audits; repeated groups are intentional",
            },
            {
                "check": "respondent_level_output",
                "value": False,
                "detail": "only aggregate tables and figures exported",
            },
            {
                "check": "municipality_table_output",
                "value": False,
                "detail": "municipalities are outside the estimand",
            },
            {
                "check": "retirement_year_invalid_r2",
                "value": retirement_invalid["r2_retired_outside"],
                "detail": "nonmissing years failing survey-year and age-band plausibility rules",
            },
            {
                "check": "retirement_year_invalid_r1",
                "value": retirement_invalid["r1_retired"],
                "detail": "nonmissing years failing survey-year and age-band plausibility rules",
            },
            {
                "check": "r1_current_secondary_residence_yes",
                "value": int(r1_dual.eq(1).sum()),
                "detail": f"{len(r1_dual)} valid responses; comparator contamination check",
            },
            {
                "check": "ordered_response_first_rank_retained",
                "value": True,
                "detail": "Q41-Q43 order preserved through first-ranked category profiles",
            },
            {
                "check": "five_global_dimension_tests",
                "value": int(len(dimension_tests)),
                "detail": "five post-data global null hypotheses; BH correction across dimensions",
            },
            {
                "check": "profile_outcome_models",
                "value": int(len(profile_models)),
                "detail": "one supplementary model per H5 item, plus one no-income sensitivity per item",
            },
        ]
    )
    write_csv(quality, diagnostics_dir / "descriptive_data_quality.csv")
    markdown_report(
        internal_dir / "descriptive_comparative_report.md",
        masks,
        r2_likert,
        comparisons,
    )
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--item-map",
        type=Path,
        default=Path("config/descriptive_comparative_items.csv"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metadata = run_analysis(args.input, args.output_root, args.item_map)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
