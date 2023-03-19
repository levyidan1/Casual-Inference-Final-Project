import pandas as pd
from Patient import Patient
import numpy as np
import os

############################################################
clinical_data_path = "clinical_sorted.csv"
follow_ups_data_path = "follow_ups_data.csv"

exp_name = "exp1_Y2"

features = ['Albumin', 'Calcium', 'Total Protein', 'Creatinine', 'Immunoglobulin A', 'Absolute Neutrophil', 'Serum Free Immunoglobulin Light Chain, Kappa', 'Serum Free Immunoglobulin Light Chain, Lambda', 'M Protein', 'Immunoglobulin G', 'Leukocytes', 'Immunoglobulin M', 'Hemoglobin', 'Blood Urea Nitrogen', 'Glucose', 'Platelets']
#features = ['Lactate Dehydrogenase', 'Beta 2 Microglobulin', 'Albumin', 'Calcium', 'Total Protein', 'Creatinine', 'Immunoglobulin A', 'Absolute Neutrophil', 'Serum Free Immunoglobulin Light Chain, Kappa', 'Serum Free Immunoglobulin Light Chain, Lambda', 'M Protein', 'Immunoglobulin G', 'Leukocytes', 'Immunoglobulin M', 'Hemoglobin', 'Blood Urea Nitrogen', 'Glucose', 'Platelets']
treatments = ['Bortezomib', 'Ixazomib', 'Panobinostat', 'Carmustine', 'Carfilzomib', 'Lenalidomide', 'Dexamethasone', 'Melphalan', 'Cyclophosphamide', 'Bendamustine', 'Prednisone', 'Thalidomide', 'Pomalidomide', 'Elotuzumab', 'Other', 'Daratumumab', 'Doxorubicin']

feature2range = {
    'Albumin': [33, 57], 
    'Calcium': [2.25, 2.62], 
    'Total Protein': [6, 8], 
    'Creatinine': [53, 114.9], 
    'Immunoglobulin A': [0.61, 3.56], 
    'Absolute Neutrophil': [1.7, 7], 
    'Serum Free Immunoglobulin Light Chain, Kappa': [3.3, 19.4], #
    'Serum Free Immunoglobulin Light Chain, Lambda': [5.71, 26.3], #
    'M Protein': [0, 3], #
    'Immunoglobulin G': [7.67, 15.90], 
    'Leukocytes': [3.5, 10.5], #
    'Immunoglobulin M': [0.37, 2.86], 
    'Hemoglobin': [7.45, 10.86], 
    'Blood Urea Nitrogen': [2.5, 7.14], 
    'Glucose': [3.885, 5.55], # 70-100 or 100-125?
    'Platelets': [150, 450], 

    'Lactate Dehydrogenase': [1.75, 5.55], 
    'Beta 2 Microglobulin': [0.70, 1.80]
}
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

        next_therapy = patient.get_follow_ups_before_line_of_therapy(next_therapy_name)
        if len(next_therapy)==0:
            next_therapy = patient.get_follow_ups_in_line_of_therapy(next_therapy_name)
            if len(next_therapy)==0:
                continue
            next_latest_follow_up = next_therapy[0]
        else:
            next_latest_follow_up = next_therapy[-1]
        ok, total = 0, 0
        for feature in features:
            val = next_latest_follow_up.marker2val(feature)
            if val is None:
                continue
            else:
                assert isinstance(val, float)
                feature_range = feature2range[feature]
                if feature_range[0]<=val<=feature_range[0]:
                    ok += 1
                total += 1
        assert total>0
        y = ok/total
        
        if patient.height>0 and patient.weight>0:
            markers = set(latest_follow_up.markers)
            
            sample = [patient.height, patient.weight]
            for feature in features:
                val = latest_follow_up.marker2val(feature)
                if val is None:
                    val = 3
                else:
                    assert isinstance(val, float)
                    feature_range = feature2range[feature]
                    if val<feature_range[0]:
                        val = 1
                    elif val>feature_range[1]:
                        val = 2
                    else:
                        val = 0
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