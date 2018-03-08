import numpy as np
from config import cf
from util import *
import random
import copy


'''
# state
x = np.array([theta, w])
theta = [-pi, pi]

# motion model
fx = np.array([x[2], cf.a * np.sin(x[1])- cf.b * x[2] + u])

# stage cost
gx = 1 - np.exp(cf.k * np.cos(x[1]) - cf.k) + cf.r / 2. * u**2

# sigma
sigma = np.eye([2,2]) * cf.sigma
'''


# policy iteration
print(['Degree', cf.theta_step])
print(['Angular Velocity', cf.w_step])
print(['Control', cf.u_step])

# initialize policy and value
cc = [tuple(row) for row in cf.x]
kk = np.random.choice(cf.u, len(cc)).tolist()
policy = dict(zip(tuple(cc), kk))
values = [0] * np.size(cf.x, 0)
V = dict(zip(cc, values))

diff = 1000000
k = 0
while diff >= cf.diff_ep:
    old_V = V.copy()
    for x in cf.x:
        x_d = x / 100.
        # policy evaluation
        u = policy[tuple(x)]
        cost = computecost(x_d, u)
        x_new, prob = motion(x_d, u)
        x_new = np.floor(x_new*100).astype(int).tolist()
        Vk = np.array([old_V[tuple(i)] for i in x_new])
        
        V[tuple(x)] = sum(Vk * prob) * cf.gamma + cost

        # policy improvement
        bl = []
        for u in cf.u:  
            cost = computecost(x_d, u) 
            x_new, prob = motion(x_d, u)
            x_new = np.floor(x_new*100).astype(int).tolist()
            Vk = np.array([old_V[tuple(i)] for i in x_new])
            bl.append(sum(Vk * prob) * cf.gamma + cost)
            
        # update policy
        policy[tuple(x)] = cf.u[np.argmin(bl)]

    k += 1
    diff = sum(abs(np.array(np.sort(V.values())) - np.array(np.sort(old_V.values()))))
    print([['step', k], ['diff', diff]])

# get traj
# initial state 
x0 = np.array(V.keys()[2]) / 100.
theta_seq, u_seq = findseq(x0, policy)

visualization(np.array(theta_seq), np.array(u_seq), 'PI', cf.saveanimate)

a = 5



        

