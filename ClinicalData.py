import contextlib
import pandas as pd
import math


class ClinicalData:
    def __init__(self, clinical_data: pd.DataFrame):
        self.clinical_data = clinical_data
        self.case_submitter_id = clinical_data["case_submitter_id"]
        try:
            self.age_at_index = int(clinical_data["age_at_index"][0]) if clinical_data["age_at_index"][
                                                                             0] != "\'--" else None
        except IndexError:
            self.age_at_index = int(clinical_data["age_at_index"]) if clinical_data["age_at_index"] != "\'--" else None
        self.cause_of_death = clinical_data["cause_of_death"][0] if clinical_data["cause_of_death"][
                                                                        0] != "\'--" else None
        try:
            self.days_to_birth = int(clinical_data["days_to_birth"][0]) if clinical_data["days_to_birth"][
                                                                               0] != "\'--" else None
        except ValueError:
                self.days_to_birth = int(clinical_data["days_to_birth"]) if clinical_data["days_to_birth"] != "\'--" else None

        try:
            self.days_to_death = int(clinical_data["days_to_death"][0]) if clinical_data["days_to_death"][
                                                                                0] != "\'--" else None
        except ValueError:
            self.days_to_death = int(clinical_data["days_to_death"]) if clinical_data["days_to_death"] != "\'--" else None
        self.ethnicity = clinical_data["ethnicity"][0] if clinical_data["ethnicity"][0] != "\'--" else None
        self.gender = clinical_data["gender"][0] if clinical_data["gender"][0] != "\'--" else None
        self.race = clinical_data["race"][0] if clinical_data["race"][0] != "\'--" else None
        self.vital_status = clinical_data["vital_status"][0] if clinical_data["vital_status"][0] != "\'--" else None
        self.age_at_diagnosis = int(clinical_data["age_at_diagnosis"][0]) if clinical_data["age_at_diagnosis"][
                                                                                 0] != "\'--" else None  # in days
        try:
            self.days_to_last_follow_up = int(clinical_data["days_to_last_follow_up"][0]) if \
                    clinical_data["days_to_last_follow_up"][0] != "\'--" else None  # in days
        except IndexError:
            self.days_to_last_follow_up = int(clinical_data["days_to_last_follow_up"]) if \
                    clinical_data["days_to_last_follow_up"] != "\'--" else None
        try:
            self.days_to_last_known_disease_status = int(clinical_data[
                                                             "days_to_last_known_disease_status"][0]) if clinical_data[
                                                                                                             "days_to_last_known_disease_status"][
                                                                                                             0] != "\'--" else None  # in days
        except IndexError:
            self.days_to_last_known_disease_status = int(clinical_data[
                                                             "days_to_last_known_disease_status"]) if clinical_data[
                                                                                                         "days_to_last_known_disease_status"] != "\'--" else None
        self.iss_stage = (
            list(clinical_data["iss_stage"][0].split(","))
            if clinical_data["iss_stage"][0] != "\'--"
            else None
        )
        self.days_to_treatment_end = []
        for day in clinical_data["days_to_treatment_end"]:
            if day != "\'--":
                try:
                    self.days_to_treatment_end.append(int(day))
                except ValueError:
                    self.days_to_treatment_end.append(math.inf)

            else:
                self.days_to_treatment_end.append(math.inf)
        self.days_to_treatment_start = []
        try:
            for day in clinical_data["days_to_treatment_start"]:
                if day not in ["inf", "\'--", " inf"]:
                    try:
                        self.days_to_treatment_start.append(int(day))
                    except OverflowError:
                        self.days_to_treatment_start.append(0)
                else:
                    self.days_to_treatment_start.append(0)
        except TypeError:
            if clinical_data["days_to_treatment_start"] not in ["inf", "\'--", " inf"]:
                try:
                    self.days_to_treatment_start.append(int(clinical_data["days_to_treatment_start"]))
                except OverflowError:
                    self.days_to_treatment_start.append(0)
            else:
                self.days_to_treatment_start.append(0)
        self.regimen_or_line_of_therapy = []
        if isinstance(clinical_data["regimen_or_line_of_therapy"], str):
            if clinical_data["regimen_or_line_of_therapy"] != "\'--":
                self.regimen_or_line_of_therapy.append(clinical_data["regimen_or_line_of_therapy"])
            else:
                self.regimen_or_line_of_therapy.append(None)
        else:
            for regimen in clinical_data["regimen_or_line_of_therapy"]:
                if regimen != "\'--":
                    self.regimen_or_line_of_therapy.append(regimen)
                else:
                    self.regimen_or_line_of_therapy.append(None)
        self.therapeutic_agents = []
        if isinstance(clinical_data["therapeutic_agents"], str):
            if clinical_data["therapeutic_agents"][0] != "\'--":
                self.therapeutic_agents.append(clinical_data["therapeutic_agents"])
            else:
                self.therapeutic_agents.append(None)
        else:
            for agent in clinical_data["therapeutic_agents"]:
                if agent != "\'--":
                    self.therapeutic_agents.append(agent)
                else:
                    self.therapeutic_agents.append(None)
        self.treatment_type = []
        if isinstance(clinical_data["treatment_type"], str):
            if clinical_data["treatment_type"] != "\'--":
                self.treatment_type.append(clinical_data["treatment_type"])
            else:
                self.treatment_type.append(None)
        else:
            for treatment in clinical_data["treatment_type"]:
                if treatment != "\'--":
                    self.treatment_type.append(treatment)
                else:
                    self.treatment_type.append(None)
        try:
            self.treatment_lines_data = pd.DataFrame(
                {
                    "regimen_or_line_of_therapy": self.regimen_or_line_of_therapy,
                    "days_to_treatment_start": self.days_to_treatment_start,
                    "days_to_treatment_end": self.days_to_treatment_end,
                    "therapeutic_agents": self.therapeutic_agents,
                    "treatment_type": self.treatment_type,
                }
            )
        except ValueError:
            self.treatment_lines_data = pd.DataFrame(
                {
                    "regimen_or_line_of_therapy": [self.regimen_or_line_of_therapy],
                    "days_to_treatment_start": [self.days_to_treatment_start],
                    "days_to_treatment_end": [self.days_to_treatment_end],
                    "therapeutic_agents": [self.therapeutic_agents],
                    "treatment_type": [self.treatment_type],
                }
            )
        with contextlib.suppress(TypeError):
            self.treatment_lines_data = self.treatment_lines_data.sort_values(
                ["days_to_treatment_start", "days_to_treatment_end"]
            )
