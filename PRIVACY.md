# Privacy and disclosure control

## Release decision

Only aggregate evidence is released. The source workbook is classified inside
the research project as restricted respondent-level data. The original consent
language, data controller and redistribution permissions remain undocumented in
the materials available to this workflow.

## Safeguards

1. No respondent-level row is copied.
2. No direct identifier, technical key, municipality, timestamp, exact date or
   free-text response is included.
3. Participant counts between 1 and 4 are suppressed as `<5`.
4. The release is generated from an explicit allowlist of aggregate outputs.
5. Every CSV is checked for forbidden column-name tokens before export.

## Residual limitations

Aggregate data reduce but do not eliminate disclosure risk. Users must not
attempt to identify respondents or combine these outputs with external data for
that purpose. A respondent-level open-data release requires documented consent,
data-controller authorization and a separate disclosure-risk assessment.
