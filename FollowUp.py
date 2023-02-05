from typing import Tuple

import pandas as pd


class FollowUp:
    def __init__(self, follow_up_data: pd.DataFrame, follow_up_id: str):
        self.follow_up_data = follow_up_data
        self.follow_up_id = follow_up_id
        self.days_to_follow_up = int(follow_up_data["Days to Follow-Up"][0])
        self.markers = follow_up_data["Laboratory Test"].unique()

    def __len__(self) -> int:
        return len(self.markers)

    def __getitem__(self, idx) -> Tuple[pd.DataFrame, str]:
        marker = self.markers[idx]
        return self.follow_up_data.loc[marker], marker
