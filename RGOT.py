# RGOT library for Python

from greed import set_G_as, compute_z
from arms import create_arms, get_real_best_arm_index, get_best_arm_parameter
from rewards import get_reward, initialize_mean_reward, update_arm

def RGOT(G,number_of_turns = 1000, number_of_arms = 10, arms_behavior = "Bernoulli", number_of_games = 1):
    policies = ["Epsilon-z-greedy"]
    arms = create_arms(number_of_arms,arms_behavior)
    arms = initialize_mean_reward(arms) ###


    return arms
    sequence_arms_pulled = []


    # decide strategy -> arm to pull


    # update


    # compute bound


    # compute betas to compare with 1/m
