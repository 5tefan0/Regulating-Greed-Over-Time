import numpy as np

def create_arms(number_of_arms,arms_behavior, policies):
    arms = [0 for i in range(number_of_arms)]
    for i in range(number_of_arms):
        arms[i] = {}
        arms[i]["arm_id"] = 'arm_'+str(i)
        arms[i]["distribution"] = arms_behavior
        if arms_behavior == "Bernoulli":
            # If behavior is Bernoulli we randomly assign to each arm its probability of success
            arms[i]["p"] = np.random.uniform(0.0,1.0)
        elif arms_behavior == "Normal":
            # If behavior is Normal we randomly assign to each arm mean and std
        else:
            arms[i]["p"] = "set parameters"
        for policy in policies: # the following parameters depend on the policy:
            arms[i]["total_pulls_"+policy] = 0
            arms[i]["rewards_"+policy] = []
            arms[i]["mean_reward_"+policy] = 0
            arms[i]["rounds_pulled_"+policy] = []
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

def get_best_estimate_arm_index(arms,policy):
    A = [arms[i]["mean_reward_"+policy] for i in range(len(arms))]
    return np.argmax(A)

def get_best_mean_estimate(arms,policy):
    A = [arms[i]["mean_reward_"+policy] for i in range(len(arms))]
    return max(A)
