from typing import Tuple

import pandas as pd


class TestData:
    def __init__(self, marker: str, value: float):
        self.name = marker
        self.value = value

    def __str__(self):
        return f"Test name: {self.name}, Value: {self.value}"

    def __repr__(self):
        return self.__str__()


class FollowUp:
    def __init__(self, follow_up_data: pd.DataFrame, follow_up_id: str):
        self.follow_up_data = follow_up_data
        self.follow_up_id = follow_up_id
        self.days_to_follow_up = int(follow_up_data["Days to Follow-Up"][0])
        self.markers = follow_up_data["Laboratory Test"].unique()

    def __len__(self) -> int:
        return len(self.markers)

    def __getitem__(self, idx) -> TestData:
        marker = self.markers[idx]
        marker_value = self.follow_up_data[
            self.follow_up_data["Laboratory Test"] == marker
            ]["Test Value"].values[0]
        return TestData(marker, marker_value)
