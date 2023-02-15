import contextlib
import pandas as pd
import math


class ClinicalData:
    def __init__(self, clinical_data: pd.DataFrame):
        self.clinical_data = clinical_data
        self.case_submitter_id = clinical_data["case_submitter_id"]
        is_series = isinstance(clinical_data["age_at_index"], pd.Series)
        self.days_to_treatment_end = []
        self.days_to_treatment_start = []
        self.regimen_or_line_of_therapy = []
        self.therapeutic_agents = []
        self.treatment_type = []
        if is_series:
            self.age_at_index =  int(clinical_data["age_at_index"][0]) if clinical_data["age_at_index"][
                                                                             0] != "\'--" else None
            self.cause_of_death = clinical_data["cause_of_death"][0] if clinical_data["cause_of_death"][
                                                                            0] != "\'--" else None
            self.days_to_birth = int(clinical_data["days_to_birth"][0]) if clinical_data["days_to_birth"][
                                                                               0] != "\'--" else None
            self.days_to_death = int(clinical_data["days_to_death"][0]) if clinical_data["days_to_death"][0] != "\'--" else None
            self.ethnicity = clinical_data["ethnicity"][0] if clinical_data["ethnicity"][0] != "\'--" else None
            self.gender = clinical_data["gender"][0] if clinical_data["gender"][0] != "\'--" else None
            self.race = clinical_data["race"][0] if clinical_data["race"][0] != "\'--" else None
            self.vital_status = clinical_data["vital_status"][0] if clinical_data["vital_status"][0] != "\'--" else None
            self.age_at_diagnosis = int(clinical_data["age_at_diagnosis"][0]) if clinical_data["age_at_diagnosis"][
                                                                                     0] != "\'--" else None  # in days
            self.days_to_last_follow_up = int(clinical_data["days_to_last_follow_up"][0]) if \
                        clinical_data["days_to_last_follow_up"][0] != "\'--" else None  # in days
            self.days_to_last_known_disease_status = int(clinical_data[
                                                             "days_to_last_known_disease_status"][0]) if clinical_data[
                                                                                                             "days_to_last_known_disease_status"][
                                                                                                             0] != "\'--" else None  # in days
            self.iss_stage = (
                list(clinical_data["iss_stage"][0].split(","))
                if clinical_data["iss_stage"][0] != "\'--"
                else None
            )
            for day in clinical_data["days_to_treatment_end"]:
                if day != "\'--":
                        self.days_to_treatment_end.append(int(day))
                else:
                    self.days_to_treatment_end.append(math.inf)
            for day in clinical_data["days_to_treatment_start"]:
                if day != "\'--" and day != float("inf"):
                    self.days_to_treatment_start.append(int(day))
                else:
                    self.days_to_treatment_start.append(0)
            for regimen in clinical_data["regimen_or_line_of_therapy"]:
                if regimen != "\'--":
                    self.regimen_or_line_of_therapy.append(regimen)
                else:
                    self.regimen_or_line_of_therapy.append(None)
            for agent in clinical_data["therapeutic_agents"]:
                if agent != "\'--":
                    self.therapeutic_agents.append(agent)
                else:
                    self.therapeutic_agents.append(None)
            for treatment in clinical_data["treatment_type"]:
                if treatment != "\'--":
                    self.treatment_type.append(treatment)
                else:
                    self.treatment_type.append(None)
            self.treatment_lines_data = pd.DataFrame(
                {
                    "regimen_or_line_of_therapy": self.regimen_or_line_of_therapy,
                    "days_to_treatment_start": self.days_to_treatment_start,
                    "days_to_treatment_end": self.days_to_treatment_end,
                    "therapeutic_agents": self.therapeutic_agents,
                    "treatment_type": self.treatment_type,
                }
            )
            self.treatment_lines_data = self.treatment_lines_data.sort_values(
                ["days_to_treatment_start", "days_to_treatment_end"]
            )

        else:
            self.age_at_index = int(clinical_data["age_at_index"]) if clinical_data["age_at_index"] != "\'--" else None
            self.cause_of_death = clinical_data["cause_of_death"] if clinical_data["cause_of_death"] !=  "\'--" else None
            self.days_to_birth = int(clinical_data["days_to_birth"]) if clinical_data[
                                                                            "days_to_birth"] != "\'--" else None
            self.days_to_death = int(clinical_data["days_to_death"]) if clinical_data["days_to_death"] != "\'--" else None
            self.ethnicity = clinical_data["ethnicity"] if clinical_data["ethnicity"] != "\'--" else None
            self.gender = clinical_data["gender"] if clinical_data["gender"] != "\'--" else None
            self.race = clinical_data["race"] if clinical_data["race"] != "\'--" else None
            self.vital_status = clinical_data["vital_status"] if clinical_data["vital_status"] != "\'--" else None
            self.age_at_diagnosis = int(clinical_data["age_at_diagnosis"]) if clinical_data["age_at_diagnosis"] \
                                                                                          != "\'--" else None  # in days
            self.days_to_last_follow_up = int(clinical_data["days_to_last_follow_up"]) if \
                        clinical_data["days_to_last_follow_up"] != "\'--" else None

            self.days_to_last_known_disease_status = int(clinical_data[
                                                             "days_to_last_known_disease_status"]) if clinical_data[
                                                                                                          "days_to_last_known_disease_status"] != "\'--" else None
            self.iss_stage = (
                list(clinical_data["iss_stage"].split(","))
                if clinical_data["iss_stage"] != "\'--"
                else None
            )
            if clinical_data["days_to_treatment_end"] != "\'--":
                self.days_to_treatment_end.append(int(clinical_data["days_to_treatment_end"]))
            else:
                self.days_to_treatment_end.append(math.inf)
            if clinical_data["days_to_treatment_start"] not in ["\'--", "inf"]:
                self.days_to_treatment_start.append(int(clinical_data["days_to_treatment_start"]))
            else:
                self.days_to_treatment_start.append(0)
            if clinical_data["regimen_or_line_of_therapy"] != "\'--":
                self.regimen_or_line_of_therapy.append(clinical_data["regimen_or_line_of_therapy"])
            else:
                self.regimen_or_line_of_therapy.append(None)
            if clinical_data["therapeutic_agents"][0] != "\'--":
                self.therapeutic_agents.append(clinical_data["therapeutic_agents"])
            else:
                self.therapeutic_agents.append(None)
            if clinical_data["treatment_type"] != "\'--":
                self.treatment_type.append(clinical_data["treatment_type"])
            else:
                self.treatment_type.append(None)
            self.treatment_lines_data = pd.DataFrame(
                {
                    "regimen_or_line_of_therapy": self.regimen_or_line_of_therapy,
                    "days_to_treatment_start": self.days_to_treatment_start,
                    "days_to_treatment_end": self.days_to_treatment_end,
                    "therapeutic_agents": self.therapeutic_agents,
                    "treatment_type": self.treatment_type,
                }
            )
