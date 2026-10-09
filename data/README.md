# Input data

Raw inputs are intentionally excluded from the anonymous repository. Provide the following files locally or pass equivalent paths to `run_analysis.py`:

1. `gbd_stomach_incidence_2022_both.csv`: GBD 2023 Results Tool export for China, 2022, stomach cancer, incidence, number, both sexes, age-specific groups from 20–24 through 95+.
2. `globocan_stomach_age_groups_2022.csv`: GLOBOCAN 2022/NCC–IARC-aligned age-group cases for China, including 20–24 through 70–74 and a 75+ residual. The script can derive 75+ when all-age and four under-20 groups are present.
3. `target_article.pdf`: the published article PDF containing Figure 4b.

The GLOBOCAN/NCC-aligned series is not an independent validation dataset. The manuscript explains the provenance relationship and the harmonisation of the age range.
