import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ucb = pd.read_csv("C:/Users/Albus Dumbledore/OneDrive/Desktop/DEsktop/machine learning/P14-Part6-Reinforcement-Learning/P14-Part6-Reinforcement-Learning/Section 31 - Upper Confidence Bound (UCB)/Python/Ads_CTR_Optimisation.csv")

#creating ucb algo 
N = 10000  #number of rounds
d = 10     #number of ads

no_of_selections  = [0] * d  #both initialized vector by zero
sum_of_reward     = [0] * d


import math

"""
for n in range(0,N):
    average_reward  = 0
    max_upper_bound = 0
    ad = 0
    for i in range(0,d):
        average_reward = no_of_selections[i] / sum_of_reward[i]
        delta_i = math.sqrt(3/2 * math.log(n+1) / no_of_selections[i])  #we changed log(n) beacuse in python indexing is started with 0 but formula not
        upper_bound = average_reward + delta_i
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i   #this will store the i'th ad that has max_upper bound """
"""
i commented this thing because this is the formula for nth round
but we have to deal with starting 10 rounds beacuse we have to gain
some information..we can't predict with no information"""

"""
we will start with 1st round chossing 1st ad and finish with
10th round choosing 10th ad """

ads_selected = []
total_reward = 0
for n in range(0,N):
    max_upper_bound = 0
    ad = 0
    for i in range(0,d):
        if(no_of_selections[i]>0):
            average_reward = sum_of_reward[i] / no_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n+1) / no_of_selections[i])  #we changed log(n) beacuse in python indexing is started with 0 but formula not
            upper_bound = average_reward + delta_i
        else:
            upper_bound =1e400 #chose large upper bound
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i 
    ads_selected.append(ad)
    no_of_selections[ad] = no_of_selections[ad] + 1
    reward = ucb.values[n,ad] #this will tell the that at column = ad whether it is 0 or 1
    sum_of_reward[ad] = sum_of_reward[ad] + reward #this will calculate sum of reward
    
    total_reward = total_reward + reward #the final sum
            
    
#visualization

plt.hist(ads_selected)
plt.title("ADS_selection by ucb")
plt.xlabel("ads")
plt.ylabel("no of the time ads was selected")
plt.show()