import numpy as np
from config import cf
from util import *


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


# value iteration
print(['Degree', cf.theta_step])
print(['Angular Velocity', cf.w_step])
print(['Control', cf.u_step])


values = np.random.rand(np.size(cf.x, 0)).tolist() * 100
cc = tuple([tuple(row) for row in cf.x])
V = dict(zip(cc, values))
# optimal contol
policy = {}
policy_m = np.zeros((len(cf.theta), len(cf.w)))

diff = 1000000
k = 0
while diff >= cf.diff_ep:
    old_V = V.copy()
    for x in cf.x:
        x_d = x / 100.
        bl = []
        for u in cf.u:
            cost = computecost(x_d, u)
            x_new, prob = motion(x_d, u)
            # if (x_new*100).astype(int)[0][0] == -253:
            #     print('aaaaaaaaaaaaaaaaaa')
            # convert x to keys
            x_new = np.floor(x_new*100).astype(int).tolist()
            Vk = np.array([old_V[tuple(i)] for i in x_new])
            bl.append(sum(Vk * prob) * cf.gamma + cost)

        # update value
        # x_keys = np.floor(x*100).astype(int).tolist()
        V[tuple(x)] = np.min(bl)
        # update policy
        optima_u = cf.u[np.argmin(bl)]
        policy[tuple(x)] = optima_u
        # print(['cost', cost])
        # print(['bl', bl])
        
        policy_m[np.where(cf.theta_l == x[0]), np.where(cf.w_l == x[1])] = optima_u
        
    
    k += 1
    diff = sum(abs(np.array(np.sort(V.values())) - np.array(np.sort(old_V.values()))))
    print([['step', k], ['diff', diff]])
    
# get traj
# initial state 
x0 = np.array(V.keys()[2]) / 100.
theta_seq, u_seq = findseq(x0, policy)
visualization(np.array(theta_seq), np.array(u_seq), 'VI', cf.saveanimate)

fig,ax = plt.subplots()
ax.imshow(policy_m)
ax.colorbar()
ax.set_xticks(cf.w)
ax.set_xticklabels('Angular Velocity')
ax.set_yticks(cf.theta)
ax.set_yticklabels('theta')
plt.show()


a = 5



        

