import unittest
import pandas as pd

from DataLoader import ClinicalDataset


class MyTestCase(unittest.TestCase):
    def test_number_of_markers(self):
        df = pd.read_csv("follow_ups_data.csv")
        markers = df["Laboratory Test"].unique()
        self.assertEqual(len(markers), 19)

    def test_number_of_cases(self):
        follow_ups_df = pd.read_csv("follow_ups_data.csv")
        cases = follow_ups_df["Case ID"].unique()
        self.assertEqual(len(cases), 992)
        clinical_df = pd.read_csv("clinical.tsv", sep="\t")
        cases = clinical_df["case_id"].unique()
        self.assertEqual(len(cases), 995)
        print(f'Number of cases with only 1 follow-up: {len(set(cases) - set(follow_ups_df["Case ID"].unique()))}:')
        print(set(cases) - set(follow_ups_df["Case ID"].unique()))


    def test_number_of_follow_ups(self):
        df = pd.read_csv("follow_ups_data.csv")
        follow_ups = df["Follow-Up"].unique()
        self.assertEqual(len(follow_ups), 8591)

    def test_manually(self):
        clinical_data_path = "clinical_sorted.tsv"
        follow_ups_data_path = "follow_ups_data.csv"
        dataset = ClinicalDataset(clinical_data_path, follow_ups_data_path)
        train_dataloader, test_dataloader = dataset.get_train_test_dataloaders()
        print(f'Number of cases: {len(dataset)}')
        print(f'Number of markers: {len(dataset.markers)}')
        print(f'Number of follow-ups: {len(dataset.follow_ups)}')
        print(f'Number of patients in train dataset: {len(train_dataloader)}')
        print(f'Number of patients in test dataset: {len(test_dataloader)}')
        train_dataset, test_dataset = dataset.get_train_test_datasets()
        print(f'Number of patients in train dataset: {len(train_dataset)}')
        print(f'Number of patients in test dataset: {len(test_dataset)}')
        fourth_patient = dataset[3]
        print(f'Number of follow-ups for fourth patient: {len(fourth_patient)}')
        fourth_patient_first_follow_up = fourth_patient[0]
        print(f'Follow-up ID: {fourth_patient_first_follow_up.follow_up_id}')
        print(f'Days to follow-up: {fourth_patient_first_follow_up.days_to_follow_up}')
        print(f'Laboratory test: {fourth_patient_first_follow_up.follow_up_data}')
        print(f'Markers: {fourth_patient_first_follow_up.markers}')
        print(f'Patient ID: {fourth_patient.case_id}')
        print(f'Age at index: {fourth_patient.clinical_data.age_at_index}')
        print(f'Cause of death: {fourth_patient.clinical_data.cause_of_death}')
        print(f'Days to birth: {fourth_patient.clinical_data.days_to_birth}')
        print(f'Days to death: {fourth_patient.clinical_data.days_to_death}')
        print(f'Ethnicity: {fourth_patient.clinical_data.ethnicity}')
        print(f'Gender: {fourth_patient.clinical_data.gender}')
        print(f'Race: {fourth_patient.clinical_data.race}')
        print(f'Vital status: {fourth_patient.clinical_data.vital_status}')
        print(f'Age at diagnosis: {fourth_patient.clinical_data.age_at_diagnosis}')
        print(f'Days to last follow-up: {fourth_patient.clinical_data.days_to_last_follow_up}')
        print(
            f'Days to last known disease status: {fourth_patient.clinical_data.days_to_last_known_disease_status}')
        print(f'ISS stage: {fourth_patient.clinical_data.iss_stage}')
        print(f'Days to treatment end: {fourth_patient.clinical_data.days_to_treatment_end}')
        print(f'Days to treatment start: {fourth_patient.clinical_data.days_to_treatment_start}')
        print(f'Regimen or line of therapy: {fourth_patient.clinical_data.regimen_or_line_of_therapy}')
        print(f'Therapeutic agents: {fourth_patient.clinical_data.therapeutic_agents}')
        print(f'Treatment type: {fourth_patient.clinical_data.treatment_type}')
        print(f'Number of treatment lines: {len(fourth_patient.clinical_data.treatment_lines_data)}')
        lines = fourth_patient.clinical_data.treatment_lines_data
        print(f'Treatment lines: {fourth_patient.clinical_data.treatment_lines_data}')
        lines_of_therapy = fourth_patient.get_line_of_therapy_names()
        print(f'Lines of therapy: {lines_of_therapy}')
        print(f'Number of lines of therapy: {len(lines_of_therapy)}')
        first_line_of_therapy_follow_ups = fourth_patient.get_follow_ups_in_line_of_therapy(lines_of_therapy[0])
        print(
            f'Follow-up IDs in first line of therapy: {[follow_up.follow_up_id for follow_up in first_line_of_therapy_follow_ups]}')
        print(
            f'Days to follow-up in first line of therapy: {[follow_up.days_to_follow_up for follow_up in first_line_of_therapy_follow_ups]}')
        print(f'Number of follow-ups in first line of therapy: {len(first_line_of_therapy_follow_ups)}')
        print(
            f'Follow-ups before first line of therapy: {[follow_up.follow_up_id for follow_up in fourth_patient.get_follow_ups_before_line_of_therapy(lines_of_therapy[0])]}')
        print(
            f'Follow-ups after first line of therapy: {[follow_up.follow_up_id for follow_up in fourth_patient.get_follow_ups_after_line_of_therapy(lines_of_therapy[0])]}')
        print(
            f'Follow-ups in first line of therapy and before: {[follow_up.follow_up_id for follow_up in fourth_patient.get_follow_ups_in_line_of_therapy_and_before(lines_of_therapy[0])]}')
        print(
            f'Follow-ups in first line of therapy and after: {[follow_up.follow_up_id for follow_up in fourth_patient.get_follow_ups_in_line_of_therapy_and_after(lines_of_therapy[0])]}')
        print(
            f'Follow-up IDs in second line of therapy: {[follow_up.follow_up_id for follow_up in fourth_patient.get_follow_ups_in_line_of_therapy(lines_of_therapy[1])]}')
        print(
            f'Days to follow-up in second line of therapy: {[follow_up.days_to_follow_up for follow_up in fourth_patient.get_follow_ups_in_line_of_therapy(lines_of_therapy[1])]}')
        print(
            f'Number of follow-ups in second line of therapy: {len(fourth_patient.get_follow_ups_in_line_of_therapy(lines_of_therapy[1]))}')
        print(
            f'Follow-ups before second line of therapy: {[follow_up.follow_up_id for follow_up in fourth_patient.get_follow_ups_before_line_of_therapy(lines_of_therapy[1])]}')
        print(
            f'Follow-ups after second line of therapy: {[follow_up.follow_up_id for follow_up in fourth_patient.get_follow_ups_after_line_of_therapy(lines_of_therapy[1])]}')
        print(
            f'Follow-ups in second line of therapy and before: {[follow_up.follow_up_id for follow_up in fourth_patient.get_follow_ups_in_line_of_therapy_and_before(lines_of_therapy[1])]}')
        print(
            f'Follow-ups in second line of therapy and after: {[follow_up.follow_up_id for follow_up in fourth_patient.get_follow_ups_in_line_of_therapy_and_after(lines_of_therapy[1])]}')
        print(
            f'Therapeutic agents in first line of therapy: {fourth_patient.get_therapeutic_agents_in_line_of_therapy(lines_of_therapy[0])}')
        print(
            f'Therapeutic agents in second line of therapy: {fourth_patient.get_therapeutic_agents_in_line_of_therapy(lines_of_therapy[1])}')
        print(
            f'Days from diagnosis to first line of therapy: {fourth_patient.get_days_from_diagnosis_to_line_of_therapy(lines_of_therapy[0])}')
        print(
            f'Days from diagnosis to second line of therapy: {fourth_patient.get_days_from_diagnosis_to_line_of_therapy(lines_of_therapy[1])}')
        print(
            f'Days between first and second line of therapy: {fourth_patient.get_days_between_lines_of_therapy(lines_of_therapy[0], lines_of_therapy[1])}')


if __name__ == '__main__':
    unittest.main()
