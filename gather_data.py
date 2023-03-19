import pandas as pd
from Patient import Patient
import numpy as np
import os

############################################################
clinical_data_path = "clinical_sorted.csv"
follow_ups_data_path = "follow_ups_data.csv"

exp_name = "exp1"

features = ['Albumin', 'Calcium', 'Total Protein', 'Creatinine', 'Immunoglobulin A', 'Absolute Neutrophil', 'Serum Free Immunoglobulin Light Chain, Kappa', 'Serum Free Immunoglobulin Light Chain, Lambda', 'M Protein', 'Immunoglobulin G', 'Leukocytes', 'Immunoglobulin M', 'Hemoglobin', 'Blood Urea Nitrogen', 'Glucose', 'Platelets']
#features = ['Lactate Dehydrogenase', 'Beta 2 Microglobulin', 'Albumin', 'Calcium', 'Total Protein', 'Creatinine', 'Immunoglobulin A', 'Absolute Neutrophil', 'Serum Free Immunoglobulin Light Chain, Kappa', 'Serum Free Immunoglobulin Light Chain, Lambda', 'M Protein', 'Immunoglobulin G', 'Leukocytes', 'Immunoglobulin M', 'Hemoglobin', 'Blood Urea Nitrogen', 'Glucose', 'Platelets']
treatments = ['Bortezomib', 'Ixazomib', 'Panobinostat', 'Carmustine', 'Carfilzomib', 'Lenalidomide', 'Dexamethasone', 'Melphalan', 'Cyclophosphamide', 'Bendamustine', 'Prednisone', 'Thalidomide', 'Pomalidomide', 'Elotuzumab', 'Other', 'Daratumumab', 'Doxorubicin']
############################################################

clinical_df = pd.read_csv(clinical_data_path)
follow_ups_df = pd.read_csv(follow_ups_data_path)

cases = clinical_df["case_id"].unique()
follow_ups = follow_ups_df["Follow-Up"].unique()
markers = follow_ups_df["Laboratory Test"].unique()

cases = [case for case in cases if case in follow_ups_df["Case ID"].unique()]
clinical_df.set_index("case_id", inplace=True)
follow_ups_df.set_index("Case ID", inplace=True)

X, T, Y = [], [], []

for ind in range(len(cases)):
    case_id = cases[ind]
    clinical_data = clinical_df.loc[case_id]
    follow_ups_data = follow_ups_df.loc[case_id]

    patient = Patient(clinical_data, follow_ups_data, case_id)
    therapy_names = patient.get_line_of_therapy_names()
    if len(therapy_names)<=1:
        continue
    for therapy_name, next_therapy_name in zip(therapy_names[:-1], therapy_names[1:]):
        therapy = patient.get_follow_ups_before_line_of_therapy(therapy_name)
        if len(therapy)==0:
            therapy = patient.get_follow_ups_in_line_of_therapy(therapy_name)
            if len(therapy)==0:
                continue
            latest_follow_up = therapy[0]
        else:
            latest_follow_up = therapy[-1]
        t = patient.get_therapeutic_agents_in_line_of_therapy(therapy_name)
        y = patient.get_days_between_lines_of_therapy(therapy_name, next_therapy_name)
        
        if patient.height>0 and patient.weight>0:
            markers = set(latest_follow_up.markers)
            if len(markers.intersection(features)) == len(features):
                sample = [patient.height, patient.weight]
                for feature in features:
                    val = latest_follow_up.marker2val(feature)
                    assert isinstance(val, float)
                    sample.append(val)
                X.append(sample)
                treatment = np.zeros((len(treatments),))
                for t_ in t:
                    if t_ in treatments:
                        treatment[treatments.index(t_)] = 1
                T.append(treatment)
                Y.append(y)

X = np.array(X)
T = np.array(T)
Y = np.array(Y)

os.makedirs("Data", exist_ok=True)
np.savez(f"Data/{exp_name}", X=X, T=T, Y=Y)