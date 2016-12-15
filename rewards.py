import numpy as np

def get_reward(arm_j):
    behavior = arm_j["distribution"]
    if behavior == "Bernoulli":
        reward = np.random.binomial(1,arm_j["p"])
    return reward

def initialize_mean_reward(arms):
    for j in range(len(arms)):
        reward = get_reward(arms[j])
        update(arm[j] , reward, j)
    return arms

def update_arm(arm_j,reward,time):
    arm_j["total_pulls"] += 1
    arm_j["rounds_pulled"].append(time)
    arm_j["rewards"].append(reward)
    arm_j["mean_reward"] = np.mean(arm_j["rewards"])
    return
