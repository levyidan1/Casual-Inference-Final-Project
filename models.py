import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression

N = 10

class LinUCB:
    def __init__(self, n_features, n_treatments, alpha):
        self.n_features = n_features
        self.n_treatments = n_treatments
        self.alpha = alpha
        self.A = np.array([np.identity(self.n_features) for i in range(self.n_treatments)])
        self.b = np.array([np.zeros((self.n_features)) for i in range(self.n_treatments)])
        
    def choose_action(self, features):
        p = self.rewards(features)
        return np.argmax(p)
    
    def update(self, features, reward, treatment):
        self.A[treatment] += np.outer(features, features.T)
        self.b[treatment] += reward * features
    
    def rewards(self, features):
        theta = [np.linalg.inv(self.A[i]).dot(self.b[i]) for i in range(self.n_treatments)]
        p = np.array([theta[i].dot(features) + self.alpha * np.sqrt(features.dot(np.linalg.inv(self.A[i])).dot(features)) for i in range(self.n_treatments)])
        return p
        
def run_bandit_algorithm(data, treatments, rewards, n_treatments, alpha=0.1):
    n_samples, n_features = data.shape
    bandit = LinUCB(n_features, n_treatments, alpha)
    
    for i in range(n_samples):
        features = data[i, :n_features]
        bandit.update(features, rewards[i], treatments[i])
    
    all_rewards = []
    for i in range(n_samples):
        features = data[i, :n_features]
        all_rewards.append(bandit.rewards(features))
    all_rewards = np.array(all_rewards)
    
    return all_rewards.mean(0), all_rewards.argpartition(-N, axis=1)[:,-N:]

def S_Learner(X, T, Y, n_treatments):

    model = LinearRegression().fit(np.concatenate((X, T[:,None]), axis=1), Y)
    
    actions = []
    rewards = []
    for i in range(n_treatments):

        prediction = model.predict(np.concatenate((X, np.full((X.shape[0],1), fill_value=i)), axis=1))
        rewards.append(prediction.mean())
        actions.append(prediction)
    
    actions = np.stack(actions, axis=1)
    
    return rewards, actions.argpartition(-N, axis=1)[:,-N:]

def T_Learner(X, T, Y, n_treatments):

    actions = []
    rewards = []
    for i in range(n_treatments):
    
        X_Ti = X[T==i]
        Y_Ti = Y[T==i]
        model = LinearRegression().fit(X_Ti, Y_Ti)

        prediction = model.predict(X)
        rewards.append(prediction.mean())
        actions.append(prediction)
    
    actions = np.stack(actions, axis=1)
    
    return rewards, actions.argpartition(-N, axis=1)[:,-N:]

def IPW(X, T, Y, n_treatments, max_iter=500):
    model = LogisticRegression(max_iter=max_iter).fit(X, T)
    
    probs = model.predict_proba(X) # [n,treatments]

    weights = 1/probs[np.arange(X.shape[0]),T]
    
    weighted_Y = Y*weights
    
    rewards = [(weighted_Y[T==i].sum())/(weights[T==i].sum()) for i in range(n_treatments)]

    return rewards

if __name__=="__main__":
    n_examples = 100
    n_treatments = 15
    X = np.random.randn(n_examples,5)
    T = np.random.randint(0, n_treatments, (n_examples,))
    Y = np.random.randn(n_examples,)

    rewards1 = S_Learner(X, T, Y, n_treatments)
    rewards2 = T_Learner(X, T, Y, n_treatments)
    rewards3 = IPW(X, T, Y, n_treatments)
    rewards4 = run_bandit_algorithm(X, T, Y, n_treatments)
    print(len(rewards1))
    print(len(rewards2))
    print(len(rewards3))
    print(rewards4.shape)