import numpy as np


def get_reward(arm_j):
    behavior = arm_j["distribution"]
    if behavior == "Bernoulli":
        reward = np.random.binomial(1,arm_j["p"])
    return reward

def initialize_mean_reward(arms,G,rewards_history,tradeoff_history,policy):
    for j in range(len(arms)):
        reward = get_reward(arms[j])
        update_arm(arms[j] , reward, j, policy)
        rewards_history[policy][j] = reward * G[j]
        tradeoff_history[policy][j] = False
    return arms

def update_arm(arm_j,reward,time,policy):
    arm_j["total_pulls_"+policy] += 1
    arm_j["rounds_pulled_"+policy].append(time)
    arm_j["rewards_"+policy].append(reward)
    arm_j["mean_reward_"+policy] = np.mean(arm_j["rewards_"+policy])
    return

def create_reward_history(number_of_turns, policies):
    """ Creates a dictionary that stores the rewards history for each
    policy applied """
    rewards_history = dict()
    for policy in policies:
        rewards_history[policy]=[False]*number_of_turns
    return rewards_history

def create_tradeoff_history(number_of_turns, policies):
    """ Creates a dictionary that stores the trade-off history
    (exploration VS exploitation) for each policy applied """
    tradeoff_history = dict()
    for policy in policies:
        tradeoff_history[policy]=[False]*number_of_turns
    return tradeoff_history
