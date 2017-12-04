#-------------------------------------------------------------------------------
# Import useful modules
#-------------------------------------------------------------------------------
import numpy as np
#-------------------------------------------------------------------------------




#-------------------------------------------------------------------------------
# Examples of greed functions
#-------------------------------------------------------------------------------
def set_G_as(number_of_turns, keyword):
    """ You can choose a greed function of type 'Wave', 'Christmas', or 'Step' """
    G = [0 for i in range(number_of_turns)]
    if keyword == "Wave":
        for t in xrange(number_of_turns):
            G[t]= 21 + 20*np.sin(0.25*t)
    elif keyword == "Christmas":
        christmas_peak = 1000
        for t in xrange(number_of_turns):
            G[t]= 21 + 20*np.sin(0.25*t)
            if t >= number_of_turns*0.5 and t < number_of_turns*0.6:
                G[t] = christmas_peak
    elif keyword == "Step":
        low_value = 20
        high_value = 40
        for t in xrange(number_of_turns):
            if t >= number_of_turns*0.25 and t < number_of_turns*0.5:
                G[t] = high_value
            elif t >= number_of_turns*0.75:
                G[t] = high_value
            else:
                G[t]= low_value
    else:
        print("Keyword not found. You can choose a greed function of type \
         'Wave', 'Christmas', or 'Step'. Alternatively you can choose your \
         greed function G(t).")
        return
    return G

#Example:
greed_function = set_G_as(1000, 'Step')
greed_function

def compute_z(G,percentile=65):
    return np.percentile(G,percentile)
