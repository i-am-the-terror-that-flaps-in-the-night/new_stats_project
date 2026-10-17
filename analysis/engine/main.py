from analysis.utilities.cleaning_and_cutting import data_cutting

# All analyses use data/merged/nhanes_J.parquet, cleaned and cut into study samples
samples = data_cutting()

scan = samples["scan"]
screening = samples["screening"]
diet = samples["diet"]
fasting = samples["fasting"]
