
import numpy as np

def set_G_as(number_of_turns, keyword): ###
    G = [0 for i in range(number_of_turns)]
    if keyword == "Wave":
        for t in range(number_of_turns):
            G[t]= 21 + 20*np.sin(0.25*t)
    elif keyword == "Christmas":
        for t in range(number_of_turns):
            G[t]= 1 ###
    elif keyword == "Step":
        for t in range(number_of_turns):
            G[t]= 1 ###
    else:
        print("Keyword not found. You can choose a greed function of type 'Wave', 'Christmas', or 'Step'. Alternatively you can choose your greed function G(t).")
        return
    return G

def compute_z(G,percentile=75):
    return np.percentile(G,percentile)
