import numpy as np
import math

class cf:
    a=1
    b=0.2
    sigma = np.eye(2,2) * 0.2
    k=1
    r=0.00005
    
    # discount factor
    gamma=0.9

    # 
    '''
    discretization:
    change theta step size will automatically change w step size, and control step size
    '''
    dt = 0.1
    theta_step = 0.1
    w_max = 2
    w_step = theta_step / dt * 0.2
    # w_step = 0.5
    
    u_max = 3
    u_step = theta_step / dt * 0.2

    # plan horizon
    T = 20

    # discrete u
    u = np.arange(-u_max, u_max+u_step, u_step)
    theta = np.arange(-3.14, 3.14+0.0001, theta_step)
    theta_l = (np.floor(theta*100)).astype(int)
    
    w = np.arange(-w_max, w_max, w_step)
    w_l = (np.floor(w*100)).astype(int)

    x = []
    for i in w:
        for th in theta:
            x.append([int(math.floor(th*100)), int(math.floor(i*100))])
    x = np.array(x)

    # value iteration
    diff_ep = 5

    # save animation
    saveanimate = True