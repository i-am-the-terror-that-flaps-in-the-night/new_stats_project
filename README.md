# new_stats_project

Teen liver fat in NHANES 2017–2018 (cycle J): every table and figure in the
write-up, built from one merged dataset.

## Layout

```
analysis/                  Python package (installed into .venv, so `import analysis` works anywhere)
  paths.py                 All folder paths (data/, output/ and its subfolders)
  utilities/               Data pipeline and shared stats helpers
    convert_xpt.py         1. Convert downloaded .xpt files → parquet/excel/csv
    merge.py               2. Merge per-file parquets → data/merged/nhanes_J.parquet
    cleaning.py            Clean the merged data and cut it into study samples (data_cutting())
    survey.py              Survey-weighted means, CIs, regressions
  engine/                  One script per table/figure, named by its label
    main.py                Write the cleaned study samples to output/cleaned/
    tables/table_B*.py     Tables → output/tables/
    figures/figure_*.py    Figures → output/figures/
    all_results.ipynb      Every table and figure, one per cell

data/                      (not in git)
  archive/xpt/, archive/csv/   Original NHANES downloads and csv copies
  parquet/, excel/         Converted per-file data
  merged/nhanes_J.parquet  The merged dataset all analyses use
  reference/               CDC BMI-for-age reference

output/                    (not in git)
  tables/                  table_*.csv
  figures/                 figure_*.svg
  cleaned/                 Study samples (xlsx + one csv per sample)
```

## Running

Run any script as a module from the project root, for example:

```sh
.venv/bin/python -m analysis.engine.tables.table_B1
.venv/bin/python -m analysis.engine.figures.figure_A1
.venv/bin/python -m analysis.engine.main        # write the cleaned samples
```

To rebuild the data from new downloads, put the `.xpt` files directly in `data/`
and run `analysis.utilities.convert_xpt`, then `analysis.utilities.merge`.
