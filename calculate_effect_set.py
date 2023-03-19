import numpy as np
import pandas as pd
import os
from models import S_Learner, T_Learner, IPW, run_bandit_algorithm

############################################################
exp_name = "exp1_Y2"

treatments = ['Bortezomib', 'Ixazomib', 'Panobinostat', 'Carmustine', 'Carfilzomib', 'Lenalidomide', 'Dexamethasone', 'Melphalan', 'Cyclophosphamide', 'Bendamustine', 'Prednisone', 'Thalidomide', 'Pomalidomide', 'Elotuzumab', 'Other', 'Daratumumab', 'Doxorubicin']

alpha = 0.1
############################################################

npzfile = np.load(f"Data/{exp_name}.npz")
X = npzfile["X"]
T = npzfile["T"].astype(int)
Y = npzfile["Y"]

bad_lines = (np.isnan(X).nonzero()[0])
X = np.delete(X, bad_lines, 0)
T = np.delete(T, bad_lines, 0)
Y = np.delete(Y, bad_lines, 0)

print("number of samples:", X.shape[0])

treatments_sets = np.unique(T, axis=0)
n_treatments = treatments_sets.shape[0]
sets_names = []
for treatments_set in treatments_sets:
    name = []
    for ind in treatments_set.nonzero()[0]:
        name.append(treatments[ind])
    sets_names.append(name)

new_T = []
for t in T:
    matches = treatments_sets==t[None,...]
    T_ind = matches.all(1).nonzero()[0]
    assert T_ind.shape[0]==1
    new_T.append(T_ind)
T = np.array(new_T).squeeze(1)

rewards_bandits, actions_bandits = run_bandit_algorithm(X, T, Y, n_treatments, alpha)
rewards_S_learner, actions_S_learner = S_Learner(X, T, Y, n_treatments)
rewards_T_learner, actions_T_learner = T_Learner(X, T, Y, n_treatments)
rewards_IPW = IPW(X, T, Y, n_treatments, 10000)

print((actions_bandits==T[:,None]).any(axis=1).mean(), (actions_S_learner==T[:,None]).any(axis=1).mean(), (actions_T_learner==T[:,None]).any(axis=1).mean())

df = pd.DataFrame(columns=['Set', 'Count', 'Frequency', 'Bandits', 'S_Learner', 'T_Learner', 'IPW'])
for i in range(n_treatments):
    Ti = (T==i)
    df.loc[i] = [sets_names[i], 
        Ti.sum(), 
        Ti.mean(), 
        rewards_bandits[i], 
        rewards_S_learner[i], 
        rewards_T_learner[i], 
        rewards_IPW[i]]
os.makedirs("Results", exist_ok=True)
df.to_excel(f"Results/{exp_name}.xlsx")  

'''print("#"*50)
print(f"alpha={alpha}:\n")
for i in range(n_treatments):
    print("Treatment {} | bandits: {:.2f} S_Learner: {:.2f} T_Learner: {:.2f} IPW: {:.2f}".format(
        sets_names[i], 
        rewards_bandits[i], 
        rewards_S_learner[i], 
        rewards_T_learner[i], 
        rewards_IPW[i]))
print("")'''