import numpy as np
from config import cf
import math
from scipy.stats import multivariate_normal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# compute cost
def computecost(x, u):
    gx = 1 - np.exp(cf.k * np.cos(x[0]) - cf.k) + cf.r * u**2 / 2.
    return gx

# compute gaussian distribution
def gassian(mean, cov, x):
    y = multivariate_normal.pdf(x, mean = mean, cov = cov)
    return y

def wrap(x):
    if x < -3.14:
        t = abs((x + 3.14) % (3.14 * 2.))
        x = x + 3.14 * 2. * t
    if x > 3.14:
        t = abs((x - 3.14) % (3.14 * 2.))
        x = x - 3.14 * 2. * t
    return x

def new_point(x, max, step, cov, olist):
    upper_lim = np.min([x+cov, max])
    lower_lim = np.max([x-cov, -max])
    num_up = int(math.floor((upper_lim - x) / step))
    num_low = int(math.floor((-lower_lim + x) / step))
    idx = np.where(olist == x)
    sample = olist[idx[0][0]-num_low : idx[0][0]+num_up+1]
    # low = np.flip(np.arange(x, lower_lim - 0.0001, -step)[1:], 0)
    return sample


# output x' and corresponding cost
# compute corresponding probability
def motion(x, u):
    fx = np.array([x[1], cf.a * np.sin(x[0])- cf.b * x[1] + u])
    x_bar = x + fx * cf.dt
    x_bar = [cf.theta[np.argmin(abs(cf.theta - wrap(x_bar[0])))], cf.w[np.argmin(abs(cf.w - x_bar[1]))]]
    cov = (cf.sigma.dot(cf.sigma.T)) * cf.dt
    # pick x' within one std
    new_theta = new_point(x_bar[0], 3.14, cf.theta_step, cov[0,0], cf.theta)
    new_w = new_point(x_bar[1], cf.w_max, cf.w_step, cov[1,1], cf.w)
    # print(np.shape(x_bar))
    # pick x' near mean
    new_x = []  
    prob = []
    for w in new_w:
        for th in new_theta:
            x = [th, w]
            new_x.append(x)
            prob.append(gassian(x_bar, cov, x))
    
    # normalize probability
    prob = np.array(prob)
    prob = prob / sum(prob)
    return np.array(new_x), prob

def findseq(x0, policy):
    theta_seq = []
    theta_seq.append(x0[0])
    u_seq = []
    t = 0
    while t <= cf.T:
        x_list = np.floor(x0*100).astype(int).tolist()
        u = policy[tuple(x_list)]
        u_seq.append(u)
        xs, prob = motion(x0, u)
        x_new = xs[np.argmin(prob)]
        theta_seq.append(x_new[0])
        x0 = x_new
        t += cf.dt
    x_list = np.floor(x0*100).astype(int).tolist()
    u = policy[tuple(x_list)]
    u_seq.append(u)
    return theta_seq, u_seq

################################################################################################
'''
Inverted Pendulum Animation
'''
def visualization(theta, u, title, saveanimate = False):
    t = np.arange(0.0, cf.T, cf.dt)
    x1 = np.sin(theta)
    y1 = np.cos(theta)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
    ax.grid()
    ax.axis('equal')
    plt.axis([-2, 2, -2, 2])

    line, = ax.plot([], [], 'o-', lw=2)
    time_template = 'time = %.1fs \nangle = %.2frad\ncontrol = %.2f'
    time_text = ax.text(0.05, 0.8, '', transform=ax.transAxes)

    def init():
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text

    def animate(i):
        thisx = [0, x1[i]]
        thisy = [0, y1[i]]
        line.set_data(thisx, thisy)
        time_text.set_text(time_template % (t[i], theta[i], u[i]))
        return line, time_text

    if saveanimate:
        # save video
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        ani = animation.FuncAnimation(fig, animate, np.arange(1, t.shape[0]),
                            interval=25, blit=True, init_func=init)
        ani.save(("video/" + title + "_%f.mp4" %(cf.w_step)), writer=writer)

    else:
        ani = animation.FuncAnimation(fig, animate, np.arange(1, t.shape[0]),
                              interval=25, blit=True, init_func=init)
        plt.show()


    


