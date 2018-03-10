import numpy as np
from config import cf
from util import *
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


# value iteration
print(['Degree', cf.theta_step])
print(['Angular Velocity', cf.w_step])
print(['Control', cf.u_step])


# row is theta
# col is w
V = np.random.rand(len(cf.theta), len(cf.w)) * 10
# optimal contol
policy = np.zeros((len(cf.theta), len(cf.w)))
q = np.zeros((len(cf.theta), len(cf.w)))

diff = 1000000
k = 0
while diff >= cf.diff_ep:
    diff = 0
    for row, theta in enumerate(cf.theta):
        for col, w  in enumerate(cf.w):
            x = np.array([theta, w])
            # optimal V value
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

            # update value
            # x_keys = np.floor(x*100).astype(int).tolist()
            diff += abs(V[row, col] - np.min(bl))
            V[row, col] = np.min(bl)
            # update policy
            optima_u = cf.u[np.argmin(bl)]
            policy[row, col] = optima_u
            q[row, col] = 1 - np.exp(cf.k * np.cos(x[0]) - cf.k)
            # print(['cost', cost])
            # print(['bl', bl])
                        
    
    k += 1
    # diff = sum(abs(np.array(V.values()) - np.array(old_V.values())))
    print([['step', k], ['diff', diff]])
    
# Saving the objects:
# with open('plot/vi_policy_%s.pkl' %(cf.theta_step), 'w') as f:  # Python 3: open(..., 'wb')
#     pickle.dump(policy, f)

# interpolation
# f = interpolate.interp2d(np.array(policy.keys())[:, 0], np.array(policy.keys())[:, 1], np.array(policy.values()), kind='cubic')
# tck = interpolate.bisplrep(np.array(policy.keys())[:, 0], np.array(policy.keys())[:, 1], np.array(policy.values()), s=0)
x0 = np.array([1.4, 0])
theta_seq, u_seq = findseq(x0, policy)
visualization(np.array(theta_seq), np.array(u_seq), 'VI', cf.saveanimate)

fig, ax = plt.subplots()
ax.imshow(policy)
# ax.set_xticks(cf.theta)
ax.set_xticklabels(cf.theta)
# ax.set_yticks(cf.w)
ax.set_yticklabels(cf.w)
plt.title('Policy')
plt.xlabel('theta')
plt.ylabel('w')
plt.savefig('plot/VI_policy_step_%s_noise_%s.jpg' %(cf.theta_step, cf.sigma[0, 0]))
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
plt.savefig('plot/VI_value_step_%s_noise_%s.jpg' %(cf.theta_step, cf.sigma[0, 0]))

fig, ax = plt.subplots()
ax.imshow(q)
# ax.set_xticks(cf.theta)
ax.set_xticklabels(cf.theta)
# ax.set_yticks(cf.w)
ax.set_yticklabels(cf.w)
plt.title('Q(x)')
plt.xlabel('theta')
plt.ylabel('w')
plt.savefig('plot/VI_Q_step_%s_noise_%s.jpg' %(cf.theta_step, cf.sigma[0, 0]))

# Getting back the objects:
# with open('Optimal_Control_Pendulum\\plot\\vi_policy_2.pkl') as f:  # Python 3: open(..., 'rb')
#     policy = pickle.load(f)

a = 5



        

