import numpy as np
import heapq


#-------------------------------------------------------------------------------
# Core engine of the algorithms. Outputs the index of the arm to play.
#-------------------------------------------------------------------------------
def choose_arm(t, policy, arms, best_arm_so_far, G, z):
    m = len(arms)
    if policy == "Epsilon_greedy":
        # set algorithm parameters
        c = 11.0
        d = 0.1
        k = c/d
        probability_of_random_exploration = min(1.0, k*m/t)
        exploration = np.random.binomial(1, probability_of_random_exploration, size=None)
        if exploration == 0:
            return best_arm_so_far
        else:
            return np.random.randint(0,len(arms))
    elif policy == "Epsilon_z_greedy":
        # set algorithm parameters
        c = 11.0
        d = 0.1
        k = c/d
        if G[t] > z:
            return best_arm_so_far
        else:
            probability_of_random_exploration = min(1.0, k*m/t)
            exploration = np.random.binomial(1, probability_of_random_exploration, size=None)
            if exploration == 0:
                return best_arm_so_far
            else:
                return np.random.randint(0,len(arms))
    elif policy == "Epsilon_soft_greedy":
        # set algorithm parameters
        c = 11.0
        d = 0.1
        k = c/d
        psi_t = np.log( 1.0 + 1.0 / G[t] ) / np.log( 1.0 + 1.0 / np.min(G) )   ## min over all
        probability_of_random_exploration = min(psi_t, k*m/t)
        exploration = np.random.binomial(1, probability_of_random_exploration, size=None)
        if exploration == 0:
            return best_arm_so_far
        else:
            return np.random.randint(0,len(arms))
    elif policy == "UCB":
        arm_to_play = -1
        best_UCB_found = -1
        for j in xrange(len(arms)):
            T_j = arms[j]["total_pulls_"+policy]
            UCB_j = arms[j]["mean_reward_"+policy] + np.sqrt( 2.0 * np.log( t ) / T_j )
            if UCB_j > best_UCB_found:
                arm_to_play = j
                best_UCB_found = UCB_j
        return arm_to_play
    elif policy == "UCB_z":
        if G[t] > z:
            return best_arm_so_far
        else:
            arm_to_play = -1
            best_UCB_found = -1
            for j in xrange(len(arms)):
                T_j = arms[j]["total_pulls_"+policy]
                UCB_j = arms[j]["mean_reward_"+policy] + np.sqrt( 2.0 * np.log( t ) / T_j )
                if UCB_j > best_UCB_found:
                    arm_to_play = j
                    best_UCB_found = UCB_j
            return arm_to_play
    elif policy == "UCB_soft":
        xi_t = (1.0 + t / G[t])
        arm_to_play = -1
        best_UCB_found = -1
        for j in xrange(len(arms)):
            T_j = arms[j]["total_pulls_"+policy]
            UCB_j = arms[j]["mean_reward_"+policy] + np.sqrt( 2.0 * np.log( xi_t ) / T_j )
            if UCB_j > best_UCB_found:
                arm_to_play = j
                best_UCB_found = UCB_j
        return arm_to_play
    elif policy == "variable_pool":
        c = 1.1
        m_t = min(m ,max(1, int(np.floor(c*m/t*G[t])))) # pool size decreasing
        A = [arms[i]["mean_reward_"+policy] for i in range(len(arms))]
        # the following gets the m_t highest INDECES of list A, and chooses the one in the random position. Also returns the pool size
        return heapq.nlargest(m_t, xrange(len(A)), key=A.__getitem__)[np.random.randint(0,m_t)]
    else:
        print("\n Policy not found \n")
        return

def choose_arm_and_tradeoff(t, policy, arms, best_arm_so_far, G, z):
    """ Returns True if the policy exploited the best arm found so far. """
    m = len(arms)
    if policy == "Epsilon_greedy":
        # set algorithm parameters
        c = 11.0
        d = 0.1
        k = c/d
        probability_of_random_exploration = min(1.0, k*m/t)
        exploration = np.random.binomial(1, probability_of_random_exploration, size=None)
        if exploration == 0:
            return best_arm_so_far, True
        else:
            return np.random.randint(0,len(arms)) , False
    elif policy == "Epsilon_z_greedy":
        # set algorithm parameters
        c = 11.0
        d = 0.1
        k = c/d
        if G[t] > z:
            return best_arm_so_far, True
        else:
            probability_of_random_exploration = min(1.0, k*m/t)
            exploration = np.random.binomial(1, probability_of_random_exploration, size=None)
            if exploration == 0:
                return best_arm_so_far, True
            else:
                return np.random.randint(0,len(arms)), False
    elif policy == "Epsilon_soft_greedy":
        # set algorithm parameters
        c = 11.0
        d = 0.1
        k = c/d
        psi_t = np.log( 1.0 + 1.0 / G[t] ) / np.log( 1.0 + 1.0 / np.min(G) )   ## min over all
        probability_of_random_exploration = min(psi_t, k*m/t)
        exploration = np.random.binomial(1, probability_of_random_exploration, size=None)
        if exploration == 0:
            return best_arm_so_far , True
        else:
            return np.random.randint(0,len(arms)) , False
    elif policy == "UCB":
        arm_to_play = -1
        second_best_arm = -1
        best_UCB_found = -1
        for j in xrange(len(arms)):
            T_j = arms[j]["total_pulls_"+policy]
            UCB_j = arms[j]["mean_reward_"+policy] + np.sqrt( 2.0 * np.log( t ) / T_j )
            if UCB_j > best_UCB_found:
                second_best_arm = arm_to_play # the arm you thought you were playing now is second best
                arm_to_play = j # update best arm
                best_UCB_found = UCB_j
        if arms[arm_to_play]["mean_reward_"+policy] > arms[second_best_arm]["mean_reward_"+policy]: #\
        #+ np.sqrt( 2.0 * np.log( t ) / arms[second_best_arm]["total_pulls_"+policy] ):
            Exploitation = True
        else:
            Exploitation = False
        return arm_to_play, Exploitation
    elif policy == "UCB_z":
        if G[t] > z:
            Exploitation = True
            return best_arm_so_far, Exploitation
        else:
            arm_to_play = -1
            second_best_arm = -1
            best_UCB_found = -1
            for j in xrange(len(arms)):
                T_j = arms[j]["total_pulls_"+policy]
                UCB_j = arms[j]["mean_reward_"+policy] + np.sqrt( 2.0 * np.log( t ) / T_j )
                if UCB_j > best_UCB_found:
                    second_best_arm = arm_to_play # the arm you thought you were playing now is second best
                    arm_to_play = j # update best arm
                    best_UCB_found = UCB_j
            if arms[arm_to_play]["mean_reward_"+policy] > arms[second_best_arm]["mean_reward_"+policy]: #\
            #+ np.sqrt( 2.0 * np.log( t ) / arms[second_best_arm]["total_pulls_"+policy] ):
                Exploitation = True
            else:
                Exploitation = False
            return arm_to_play, Exploitation
    elif policy == "UCB_soft":
        xi_t = (1.0 + t / G[t])
        arm_to_play = -1
        second_best_arm = -1
        best_UCB_found = -1
        for j in xrange(len(arms)):
            T_j = arms[j]["total_pulls_"+policy]
            UCB_j = arms[j]["mean_reward_"+policy] + np.sqrt( 2.0 * np.log( xi_t ) / T_j )
            if UCB_j > best_UCB_found:
                second_best_arm = arm_to_play # the arm you thought you were playing now is second best
                arm_to_play = j # update best arm
                best_UCB_found = UCB_j
        if arms[arm_to_play]["mean_reward_"+policy] > arms[second_best_arm]["mean_reward_"+policy]: #\
        #+ np.sqrt( 2.0 * np.log( xi_t ) /  arms[second_best_arm]["total_pulls_"+policy] ):
            Exploitation = True
        else:
            Exploitation = False
        return arm_to_play, Exploitation
    elif policy == "variable_pool":
        c = 1.1
        m_t = min(m ,max(1, int(np.floor(c*m/t*G[t])))) # pool size decreasing
        if m_t == 1:
            Exploitation = True
        else:
            Exploitation = False
        A = [arms[i]["mean_reward_"+policy] for i in range(len(arms))]
        # the following gets the m_t highest INDECES of list A, and chooses the one in the random position
        return heapq.nlargest(m_t, xrange(len(A)), key=A.__getitem__)[np.random.randint(0,m_t)] , Exploitation
    else:
        print("\n Policy not found \n")
        return
