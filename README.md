# Retired residential profiles and territorial co-presence in Valais

This repository contains the disclosure-controlled aggregate data, figures,
diagnostics and analytical code supporting the article on retired primary (R1)
and secondary-home (R2) residents in the canton of Valais.

## Public-data scope

- Unit represented in the original study: individual survey respondent.
- Public files: aggregate tables only.
- Analytical populations: 348 retired R2 respondents domiciled outside Valais
  and 137 retired R1 respondents who explicitly reported no former R2 status in
  the municipality.
- Excluded from this repository: respondent-level rows, technical identifiers,
  municipalities, free text, timestamps, exact dates and the restricted source
  workbook.
- Disclosure control: participant counts from 1 to 4 are shown as `<5`; linked
  percentages are blanked where necessary.

The repository does **not** contain anonymised microdata. Available project
documentation does not establish permission to redistribute individual survey
responses, and combinations of residential and sociodemographic attributes may
permit re-identification.

## Contents

- `data/`: disclosure-controlled analytical tables used in the article and its
  supplement.
- `figures/`: article figures and the nationality profile requested for the
  sample description.
- `diagnostics/`: aggregate analysis metadata and data-quality checks.
- `code/`: the analytical implementation and configuration for transparency.
- `PRIVACY.md`: disclosure-control rationale and limitations.
- `DATA_DICTIONARY.md`: public-file definitions.
- `checksums.sha256`: integrity hashes for every released file.

## Data availability statement

The disclosure-controlled aggregate data and analytical materials that support
the findings of this study are openly available in the GitHub repository
`theOldfranz86/valais-retired-copresence-data` at
https://github.com/theOldfranz86/valais-retired-copresence-data. Respondent-level
data are not publicly shared because redistribution permissions and consent for
open individual-level release are not documented.

## Reproducibility limitation

The public release supports inspection of reported estimates, distributions,
models and diagnostics. Exact re-estimation from respondent-level records is not
possible from this public package. Access to the restricted source data remains
subject to confirmation by the data controller and the research team.

## Licence

No reuse licence is assigned pending confirmation from the data controller and
all authors. Public visibility does not by itself grant permission for reuse.
