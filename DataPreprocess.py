"""
This file is used to preprocess the clinical.tsv file.
It sorts the data by case_id and then by days_to_treatment_start
"""

import pandas as pd


def preprocess_clinical_data(clinical_data_path, output_path):
    clinical_df = pd.read_csv(clinical_data_path, sep="\t")
    # Sort the data by case_id and then by days_to_treatment_start
    # days_to_treatment_start is the number of days from the date of diagnosis to the date of treatment start
    # days_to_treatment_start is "'--" if the treatment has not started yet
    # days_to_treatment_start is a string so we need to convert it to a number
    clinical_df["days_to_treatment_start"] = clinical_df["days_to_treatment_start"].apply(
        lambda x: float(x) if x != "\'--" else float("inf"))
    clinical_df = clinical_df.sort_values(by=["case_id", "days_to_treatment_start"])
    clinical_df.to_csv(output_path, index=False)



if __name__ == '__main__':
    preprocess_clinical_data("clinical.tsv", "clinical_sorted.csv")

