import numpy as np
import math

class cf:

    # motion model and cost parameter
    a = 1
    b = 0
    sigma = np.eye(2,2) * 0.001
    k = 0.8
    r = 0.001
    
    # discount factor
    gamma=0.8

    # 
    '''
    discretization:
    change theta step size will automatically change w step size, and control step size
    '''
    dt = 0.05
    theta_step = 0.05
    w_max = 5
    # w_step = theta_step / dt * 0.2
    w_step = 0.1
    
    u_max = 5
    u_step = 0.1

    # plan horizon
    T = 20

    # discrete u
    u = np.arange(-u_max, u_max+u_step, u_step)
    theta = np.arange(-3.14, 3.14+0.0001, theta_step)
    # theta = np.array([-2, 2])
    
    w = np.arange(-w_max, w_max+w_step, w_step)
    # x = []
    # for i in w:
    #     for th in theta:
    #         x.append([int(math.floor(th*100)), int(math.floor(i*100))])
    # x = np.array(x)

    # value iteration
    diff_ep = 0.1

    # policy iteration
    diff_p = 0.1

    # save animation
    saveanimate = True

    # starting point
    x0 = np.array([1.4, 0])
