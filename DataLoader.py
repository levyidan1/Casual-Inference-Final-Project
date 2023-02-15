""""
This file is used to load the data from the follow_ups_data.csv file and the clinical.tsv file.
It creates a dataloader object that can be used to get the data for each patient.
# clinical coloumns:
case_id	case_submitter_id	age_at_index	cause_of_death	days_to_birth	days_to_death	ethnicity	gender	race	vital_status	age_at_diagnosis	days_to_last_follow_up	days_to_last_known_disease_status	iss_stage	days_to_treatment_end	days_to_treatment_start	regimen_or_line_of_therapy	therapeutic_agents	treatment_or_therapy	treatment_type
# follow_ups_data coloumns:
Case ID	Follow-Up	Days to Follow-Up	Laboratory Test	Test Value	Test Units	Patient Height	Patient Weight
"""

from typing import Tuple
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from Patient import Patient


class ClinicalDataset(Dataset):
    def __init__(self, clinical_data_path: str, follow_ups_data_path: str):
        self.clinical_df = pd.read_csv(clinical_data_path)
        self.follow_ups_df = pd.read_csv(follow_ups_data_path)
        self.cases = self.clinical_df["case_id"].unique()
        self.follow_ups = self.follow_ups_df["Follow-Up"].unique()
        self.markers = self.follow_ups_df["Laboratory Test"].unique()
        self.marker_units_dict = {
            marker: self.follow_ups_df[self.follow_ups_df["Laboratory Test"] == marker]["Test Units"].unique()[0]
            for marker in self.markers
        }
        self.cases = [case for case in self.cases if case in self.follow_ups_df["Case ID"].unique()]
        self.clinical_df.set_index("case_id", inplace=True)
        self.follow_ups_df.set_index("Case ID", inplace=True)

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, idx: int) -> Patient:
        case_id = self.cases[idx]
        clinical_data = self.clinical_df.loc[case_id]
        follow_ups_data = self.follow_ups_df.loc[case_id]
        return Patient(clinical_data, follow_ups_data, case_id)

    def get_dataloader(self, batch_size: int = 1, shuffle: bool = True) -> DataLoader:
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)

    def get_train_test_dataloaders(self, train_size: float = 0.8, batch_size: int = 1, shuffle: bool = True) -> Tuple[
            DataLoader, DataLoader]:
        train_size = int(train_size * len(self))
        test_size = len(self) - train_size
        train_dataset, test_dataset = random_split(self, [train_size, test_size])
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
        return train_dataloader, test_dataloader

    def get_train_test_datasets(self, train_size: float = 0.8) -> Tuple[Dataset, Dataset]:
        train_size = int(train_size * len(self))
        test_size = len(self) - train_size
        train_dataset, test_dataset = random_split(self, [train_size, test_size])
        return train_dataset, test_dataset

