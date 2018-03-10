import numpy as np
from config import cf
from util import *
import random
import copy
import pickle

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
V = np.random.rand(len(cf.theta), len(cf.w)) * 10
# optimal contol
policy = np.random.choice(cf.u, len(cf.theta) * len(cf.w)).reshape(len(cf.theta), len(cf.w))
q = np.zeros((len(cf.theta), len(cf.w)))

diff_p = 10000
k = 0
while diff_p >= cf.diff_p:
    # policy evaluation
    diff_p = 0
    diff = 100000
    while diff >= cf.diff_ep:
        diff = 0
        for row, theta in enumerate(cf.theta):
            for col, w  in enumerate(cf.w):
                x = np.array([theta, w])
                u = policy[row, col]
                cost = computecost(x, u)
                x_new, prob = motion(x, u)
                Vk = []
                for xx in x_new:
                    r, c = findind(xx)
                    Vk.append(V[r, c])
                Vk = np.array(Vk)
                
                new_V = sum(Vk * prob) * cf.gamma + cost
                diff += abs(V[row, col] - new_V)
                V[row, col] = new_V
        print(['policy Evaluation', ['diff', diff]])

    # policy improvement
    for row, theta in enumerate(cf.theta):
        for col, w  in enumerate(cf.w):
            x = np.array([theta, w])
            bl = []
            for u in cf.u:
                cost = computecost(x, u) 
                x_new, prob = motion(x, u)
                Vk = []
                for xx in x_new:
                    r, c = findind(xx)
                    Vk.append(V[r, c])
                Vk = np.array(Vk)
                bl.append(sum(Vk * prob) * cf.gamma + cost)
            
            # update policy
            optima_u = cf.u[np.argmin(bl)]
            diff_p += abs(policy[row, col] - optima_u)
            policy[row, col] = optima_u
            q[row, col] = 1 - np.exp(cf.k * np.cos(x[0]) - cf.k)
            
    k += 1
    print([['step', k], ['Policy I', diff_p]])

# Saving the objects:
# with open('plot/pi_policy_%s.pkl' %(cf.theta_step), 'w') as f:  # Python 3: open(..., 'wb')
#     pickle.dump(policy, f)
# get traj
# initial state 
x0 = cf.x0
theta_seq, u_seq = findseq(x0, policy)
visualization(np.array(theta_seq), np.array(u_seq), 'PI', cf.saveanimate)

fig, ax = plt.subplots()
ax.imshow(policy)
# ax.set_xticks(cf.theta)
ax.set_xticklabels(cf.theta)
# ax.set_yticks(cf.w)
ax.set_yticklabels(cf.w)
plt.title('Policy')
plt.xlabel('theta')
plt.ylabel('w')
plt.savefig('plot/PI_policy_step_%s.jpg' %(cf.theta_step))
# plt.show()

fig, ax = plt.subplots()
ax.imshow(V)
# ax.set_xticks(cf.theta)
ax.set_xticklabels(cf.theta)
# ax.set_yticks(cf.w)
ax.set_yticklabels(cf.w)
plt.title('Value')
plt.xlabel('theta')
plt.ylabel('w')
plt.savefig('plot/PI_value_step_%s.jpg' %(cf.theta_step))

fig, ax = plt.subplots()
ax.imshow(q)
# ax.set_xticks(cf.theta)
ax.set_xticklabels(cf.theta)
# ax.set_yticks(cf.w)
ax.set_yticklabels(cf.w)
plt.title('Q(x)')
plt.xlabel('theta')
plt.ylabel('w')
plt.savefig('plot/PI_Q_step_%s.jpg' %(cf.theta_step))


a = 5



        

