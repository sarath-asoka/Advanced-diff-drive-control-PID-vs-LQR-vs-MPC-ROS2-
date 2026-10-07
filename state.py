import numpy as np

def f_unicycle(x, u, m=2.0):
    p, vel = x          # unpack the state
    F = u[0]            # unpack the input
    p_dot   = vel
    vel_dot = F / m
    return np.array([p_dot, vel_dot])

def f(x, u):
    px, py, theta = x     # unpack state
    v, omega = u          # unpack input
    # TODO: three rates, using only theta, v, omega
    x_dot = v * np.cos(theta)
    y_dot = v * np.sin(theta)
    return np.array([x_dot, y_dot, omega])

k = f([0, 0, np.pi/2], [1.0, 0.0])