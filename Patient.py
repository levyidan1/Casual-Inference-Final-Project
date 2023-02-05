from typing import List
import pandas as pd

from FollowUp import FollowUp
from ClinicalData import ClinicalData


class Patient:
    def __init__(self, clinical_data: pd.DataFrame, follow_ups_data: pd.DataFrame, case_id: str):
        self.case_id = case_id
        self.clinical_data = ClinicalData(clinical_data)
        # There is a "Follow-Up" column in the follow_ups_data dataframe. Each follow-up spreads over multiple rows.
        self.follow_ups = []
        self.follow_ups.extend(
            FollowUp(follow_up_data, follow_up_id)
            for follow_up_id, follow_up_data in follow_ups_data.groupby(
                "Follow-Up"
            )
        )
        self.follow_ups.sort(key=lambda follow_up: follow_up.days_to_follow_up)
        self.height = int(follow_ups_data["Patient Height"].unique())
        self.weight = int(follow_ups_data["Patient Weight"].unique())

    def __len__(self) -> int:
        return len(self.follow_ups)

    def __getitem__(self, idx: int) -> FollowUp:
        return self.follow_ups[idx]

    def get_line_of_therapy_names(self) -> List[str]:
        return self.clinical_data.treatment_lines_data["regimen_or_line_of_therapy"].unique()

    def get_follow_ups_in_line_of_therapy(self, line_of_therapy: str) -> List[FollowUp]:
        """ Returns follow-ups that are in the given line of therapy.
        Each line of therapy is a list of treatments that are given in a specific order.
        A line of therapy can be found in the self.treatment_lines_data.
        line_of_therapy can be "First line of therapy", "Second line of therapy", etc.
        """
        first_day_of_line_of_therapy = self.clinical_data.treatment_lines_data[
            self.clinical_data.treatment_lines_data[
                "regimen_or_line_of_therapy"
            ]
            == line_of_therapy
            ]["days_to_treatment_start"].min()
        last_day_of_line_of_therapy = self.clinical_data.treatment_lines_data[
            self.clinical_data.treatment_lines_data[
                "regimen_or_line_of_therapy"
            ]
            == line_of_therapy
            ]["days_to_treatment_end"].max()

        return [
            follow_up
            for follow_up in self.follow_ups
            if first_day_of_line_of_therapy <= follow_up.days_to_follow_up
               <= last_day_of_line_of_therapy
        ]

    def get_follow_ups_before_line_of_therapy(self, line_of_therapy: str) -> List[FollowUp]:
        """ Returns follow-ups that are before the given line of therapy.
        Each line of therapy is a list of treatments that are given in a specific order.
        A line of therapy can be found in the self.treatment_lines_data.
        line_of_therapy can be "First line of therapy", "Second line of therapy", etc.
        """
        first_day_of_line_of_therapy = self.clinical_data.treatment_lines_data[
            self.clinical_data.treatment_lines_data[
                "regimen_or_line_of_therapy"
            ]
            == line_of_therapy
            ]["days_to_treatment_start"].min()

        return [
            follow_up
            for follow_up in self.follow_ups
            if follow_up.days_to_follow_up < first_day_of_line_of_therapy
        ]

    def get_follow_ups_after_line_of_therapy(self, line_of_therapy: str) -> List[FollowUp]:
        """ Returns follow-ups that are after the given line of therapy.
        Each line of therapy is a list of treatments that are given in a specific order.
        A line of therapy can be found in the self.treatment_lines_data.
        line_of_therapy can be "First line of therapy", "Second line of therapy", etc.
        """
        last_day_of_line_of_therapy = self.clinical_data.treatment_lines_data[
            self.clinical_data.treatment_lines_data[
                "regimen_or_line_of_therapy"
            ]
            == line_of_therapy
            ]["days_to_treatment_end"].max()

        return [
            follow_up
            for follow_up in self.follow_ups
            if follow_up.days_to_follow_up > last_day_of_line_of_therapy
        ]

    def get_follow_ups_in_line_of_therapy_and_before(self, line_of_therapy: str) -> List[FollowUp]:
        """ Returns follow-ups that are in the given line of therapy and before it.
        Each line of therapy is a list of treatments that are given in a specific order.
        A line of therapy can be found in the self.treatment_lines_data.
        line_of_therapy can be "First line of therapy", "Second line of therapy", etc.
        """
        last_day_of_line_of_therapy = self.clinical_data.treatment_lines_data[
            self.clinical_data.treatment_lines_data[
                "regimen_or_line_of_therapy"
            ]
            == line_of_therapy
            ]["days_to_treatment_end"].max()

        return [
            follow_up
            for follow_up in self.follow_ups
            if follow_up.days_to_follow_up < last_day_of_line_of_therapy
        ]

    def get_follow_ups_in_line_of_therapy_and_after(self, line_of_therapy: str) -> List[FollowUp]:
        """ Returns follow-ups that are in the given line of therapy and after it.
        Each line of therapy is a list of treatments that are given in a specific order.
        A line of therapy can be found in the self.treatment_lines_data.
        line_of_therapy can be "First line of therapy", "Second line of therapy", etc.
        """
        first_day_of_line_of_therapy = self.clinical_data.treatment_lines_data[
            self.clinical_data.treatment_lines_data[
                "regimen_or_line_of_therapy"
            ]
            == line_of_therapy
            ]["days_to_treatment_start"].min()

        return [
            follow_up
            for follow_up in self.follow_ups
            if follow_up.days_to_follow_up >= first_day_of_line_of_therapy
        ]

    def get_therapeutic_agents_in_line_of_therapy(self, line_of_therapy: str) -> List[str]:
        """ Returns a list of the therapeutic agents used in the line of surgery.
        """
        return self.clinical_data.treatment_lines_data[
            self.clinical_data.treatment_lines_data[
                "regimen_or_line_of_therapy"
            ]
            == line_of_therapy
            ]["therapeutic_agents"].unique()

    def get_days_from_diagnosis_to_line_of_therapy(self, line_of_therapy: str) -> int:
        """ Returns the number of days from diagnosis to the start of the line of therapy.
        """
        return self.clinical_data.treatment_lines_data[
            self.clinical_data.treatment_lines_data["regimen_or_line_of_therapy"]
            == line_of_therapy
            ]["days_to_treatment_start"].min()

    def get_days_between_lines_of_therapy(self, line_of_therapy_1: str, line_of_therapy_2: str) -> int:
        """ Returns the number of days between the end of the first line of therapy and the start of the second line of therapy.
        """
        first_day_of_line_of_therapy_2 = self.get_days_from_diagnosis_to_line_of_therapy(
            line_of_therapy_2
        )
        first_day_of_line_of_therapy_1 = self.get_days_from_diagnosis_to_line_of_therapy(
            line_of_therapy_1
        )

        return first_day_of_line_of_therapy_2 - first_day_of_line_of_therapy_1
