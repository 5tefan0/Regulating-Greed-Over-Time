""" Regulating Greed Over Time """

#-------------------------------------------------------------------------------
# RGOT library for Python
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# Import useful modules
#-------------------------------------------------------------------------------
# bandit functions:
from greed import set_G_as, compute_z
from arms import create_arms, get_real_best_arm_index, get_best_arm_parameter, get_best_estimate_arm_index
from rewards import get_reward, initialize_mean_reward, update_arm, create_reward_history, create_tradeoff_history
from policy import choose_arm
# module used to get the highest values from a list:
import heapq
# pandas to handle data frames:
import pandas as pd
# bokeh for visualize results:
from bokeh.charts import Histogram, Bar, defaults, vplot, hplot, show, output_file
from bokeh.embed import file_html
from bokeh.resources import CDN
from bokeh.palettes import Spectral7, Blues
from bokeh.models import HoverTool, ColumnDataSource




num_of_turns=1500
num_of_turns

#-------------------------------------------------------------------------------
# The following functions plays one istance of the multi-armed bandit algorithm
#-------------------------------------------------------------------------------
def RGOT(G,number_of_turns = 1500, number_of_arms = 10, \
    arms_behavior = "Bernoulli", policies = ["Epsilon_greedy", "UCB", \
                         "Epsilon_z_greedy",  "UCB_z" , \
                         "Epsilon_soft_greedy", "UCB_soft", \
                         "variable_pool"]):          # set the policies you want to play
    arms = create_arms(number_of_arms,arms_behavior,policies)
    rewards_history = create_reward_history(number_of_turns, policies)
    tradeoff_history = create_tradeoff_history(number_of_turns, policies)
    #pool_size = [0]*number_of_turns
    for policy in policies:
        arms = initialize_mean_reward(arms,G,rewards_history,tradeoff_history,policy)          ## add option to not initialize
    for t in range(number_of_arms,number_of_turns):
        for policy in policies:
            best_arm_so_far = get_best_estimate_arm_index(arms, policy)
            z = compute_z(G) # may depend on policy
            arm_to_play, tradeoff = choose_arm_and_tradeoff(t, policy, arms, best_arm_so_far, G, z)
            x_t = get_reward(arms[arm_to_play])                                 # reward for the arm played
            rewards_history[policy][t] = x_t * G[t]                             # actual reward you get modified by the greed function
            tradeoff_history[policy][t] = tradeoff * x_t * G[t]
            update_arm(arms[arm_to_play],x_t,t,policy)                          # update the arm performance under this policy
    return pd.DataFrame(arms), rewards_history, tradeoff_history
#-------------------------------------------------------------------------------
RGOT(G,number_of_turns = 200, number_of_arms = 10, \
    arms_behavior = "Bernoulli", policies = ["variable_pool", \
        "Epsilon_z_greedy", "Epsilon_greedy", "Epsilon_soft_greedy", \
         "UCB_z" , "UCB_soft", "UCB"])

number_of_games = 15
G = set_G_as(num_of_turns,"Wave")


#-------------------------------------------------------------------------------
# Play a number of games and save the cumulative reward to compare policies
#-------------------------------------------------------------------------------
def compare_algorithms(number_of_games,  G, number_of_turns = 1000, number_of_arms = 10,
     arms_behavior = "Bernoulli", policies = ["Epsilon_greedy", "UCB", \
                          "Epsilon_z_greedy",  "UCB_z" , \
                          "Epsilon_soft_greedy", "UCB_soft", \
                          "variable_pool"]):
    cumulative_rewards = {}
    trade_off = {}
    for policy in policies:
        cumulative_rewards[policy]= [False] * number_of_games # storage for the cululative rewards at the end of each game
        trade_off[policy]= [False] * number_of_games
    for game in xrange(number_of_games):
        print("\nPlaying game #"+str(game+1))
        df,rewards_history, tradeoff_history = RGOT(G,number_of_turns = number_of_turns, number_of_arms=number_of_arms, arms_behavior = arms_behavior, policies=policies)
        for policy in policies:
            cumulative_rewards[policy][game]=sum(rewards_history[policy])
            trade_off[policy][game]=sum(tradeoff_history[policy])
    return cumulative_rewards, trade_off






# to try:
g = set_G_as(num_of_turns,"Wave")

#-------------------------------------------------------------------------------


cumulative_rewards, trade_off = compare_algorithms(number_of_games,  G, number_of_turns = 1500, number_of_arms = 50,
     policies = [ "variable_pool", \
    "Epsilon_z_greedy", "Epsilon_greedy", "Epsilon_soft_greedy", \
    "UCB_z" , "UCB_soft", "UCB"])


#-------------------------------------------------------------------------------
# to show Histogram:
ndf = policies_performance(cumulative_rewards2)
output_file("means.html")
hist = Bar(ndf, values='mean', title="Average cumulative reward for each policy",legend=None,color=Spectral7)
show(hist)
# OLD
def policies_performance(cumulative_rewards): # needs numpy and pandas
    # returns a dataframe where the indexes are the policies and the columns are
    # mean and standard deviation of the cululative rewards
    x=[]
    for policy in cumulative_rewards.keys():
        x.append([np.mean(cumulative_rewards[policy]), \
        np.std(cumulative_rewards[policy])   ])
    return pd.DataFrame(np.array(x),columns=['mean', 'std'], index = cumulative_rewards.keys() )
#-------------------------------------------------------------------------------


# good examples:
#cumulative_rewards, trade_off = compare_algorithms(number_of_games,  G, number_of_turns = 1500, number_of_arms = 50,
#     policies = [ "variable_pool", \
#    "Epsilon_z_greedy", "Epsilon_greedy", "Epsilon_soft_greedy", \
#    "UCB_z" , "UCB_soft", "UCB"])
# cumulative_rewards2 = compare_algorithms(30, g, number_of_turns = 400, number_of_arms=70, policies = ["UCB_z" , "UCB_soft", "UCB","variable_pool" ])
# cumulative_rewards2 = compare_algorithms(30, g, number_of_turns = 400, number_of_arms=70, policies = ["UCB_z" , "UCB_soft", "UCB","variable_pool","Epsilon_z_greedy", "Epsilon_greedy", "Epsilon_soft_greedy" ])



#-------------------------------------------------------------------------------
# Plotting
#-------------------------------------------------------------------------------

hist = Histogram(df['total_pulls_Epsilon_greedy'], title="total_pulls_Epsilon_greedy" )
show(hist)

df2=pd.DataFrame(CR)
p = Bar(df2, 'arm_id', values='total_pulls_Epsilon_greedy', title="Tot")
show(p)

p = Bar(df, 'arm_id', values='total_pulls_Epsilon_greedy', title="Tot")
show(p)

b = Bar(df, 'arm_id', values='mean_reward_Epsilon_greedy', title="Tot", legend=None)
show(b)

c = Bar(df, 'arm_id', values='rewards_Epsilon_greedy', agg='sum' , title="Tot")
show(c)


len([i for i in range(num_of_turns)])
###
from bokeh.plotting import figure, output_file, show
pp = figure(title="simple line example", x_axis_label='x', y_axis_label='y')
pp.legend.location='bottom_left'
pp.line([i for i in range(num_of_turns)], r, legend="Temp.", line_width=2)
show(pp)
html = file_html(pp, CDN, "my plot")
html


    # compute bound

import matplotlib.cm as cm
colormap =cm.get_cmap("jet")
different_colors=7
color_mapping=colormap(np.linspace(0,1,different_colors),1,True)
bokeh_palette=["#%02x%02x%02x" % (r, g, b) for r, g, b in color_mapping[:,0:3]]

    # compute betas to compare with 1/m
