import numpy as np
import matplotlib.pyplot as plt

def f(x, u):
    """Continuous unicycle model. x=[x,y,theta], u=[v,omega]. Returns x_dot."""
    # TODO: return np.array([...])

def step_euler(x, u, dt):
    # TODO: one Euler step

def step_rk4(x, u, dt):
    # TODO: k1..k4, then the weighted average (formula sheet, section 2)

def simulate(step_fn, x0, u, dt, T):
    """Run step_fn for T seconds. Return array of states, shape (n_steps+1, 3)."""
    n = int(round(T / dt))
    xs = [x0]
    # TODO: loop n times, append the next state
    return np.array(xs)

def true_circle(t, v, omega):
    R = v / omega
    # TODO: return x_true, y_true  (formulas above)

if __name__ == "__main__":
    x0 = np.array([0.0, 0.0, 0.0])
    u  = np.array([1.0, 0.5])
    T  = 20.0

    for dt in [0.1, 0.5]:
        for name, step in [("euler", step_euler), ("rk4", step_rk4)]:
            xs = simulate(step, x0, u, dt, T)
            # TODO: final error = distance between xs[-1, :2] and true_circle(T, ...)
            # TODO: print name, dt, error
            # TODO: plot xs[:,0], xs[:,1] with a label

    # TODO: also plot the true circle (dense t from 0 to T), equal axis, legend
    plt.show()
