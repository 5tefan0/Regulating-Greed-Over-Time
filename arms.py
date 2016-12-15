import numpy as np

def create_arms(number_of_arms,arms_behavior):
    arms = [0 for i in range(number_of_arms)]
    for i in range(number_of_arms):
        arms[i] = {}
        arms[i]["arm_id"] = str(i)
        arms[i]["total_pulls"] = 0
        arms[i]["rewards"] = []
        arms[i]["mean_reward"] = 0
        arms[i]["rounds_pulled"] = []
        arms[i]["distribution"] = arms_behavior
        if arms_behavior == "Bernoulli":
            arms[i]["p"] = np.random.uniform(0.0,1.0)
        else:
            arms[i]["p"] = "set parameters"
    return arms

def get_real_best_arm_index(arms, arms_behavior):
    if arms_behavior == "Bernoulli":
        A = [arms[i]["p"] for i in range(len(arms))]
    else:
        arms[i]["p"] = "set parameters"
    return np.argmax(A)

def get_best_arm_parameter(arms, arms_behavior):
    if arms_behavior == "Bernoulli":
        A = [arms[i]["p"] for i in range(len(arms))]
    else:
        arms[i]["p"] = "set parameters"
    return max(A)

def get_best_estimate_arm_index(arms):
    A = [arms[i]["mean_reward"] for i in range(len(arms))]
    return np.argmax(A)

def get_best_mean_estimate(arms):
    A = [arms[i]["mean_reward"] for i in range(len(arms))]
    return max(A)
