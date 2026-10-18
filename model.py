import numpy as np
import matplotlib.pyplot as plt

def f(x, u):
    """Continuous unicycle model. x=[x,y,theta], u=[v,omega]. Returns x_dot."""
    px, py, theta = x     # unpack state
    v, omega = u          # unpack input
    x_dot = v * np.cos(theta)
    y_dot = v * np.sin(theta)
    return np.array([x_dot, y_dot, omega])

def step_euler(x, u, dt):
    x_new = x + dt * f(x,u)
    return x_new

def step_rk4(x, u, dt):
    k1 = f(x,u)
    k2 = f(x+dt*0.5*k1, u)
    k3 = f(x+dt*0.5*k2, u)
    k4 = f(x+dt*k3, u)

    x_new = x + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    return x_new

def simulate(step_fn, x0, u, dt, T):
    """Run step_fn for T seconds. Return array of states, shape (n_steps+1, 3)."""
    n = int(round(T / dt))
    xs = [x0]
    for _ in range(n):
        x_next = step_fn(xs[-1],u,dt)
        xs.append(x_next)

    return np.array(xs)

def true_circle(t, v, omega):
    R = v / omega
    x_true = R * np.sin(omega*t)
    y_true = R * (1 - np.cos(omega*t))
    return np.array([x_true, y_true])

if __name__ == "__main__":
    x0 = np.array([0.0, 0.0, 0.0])
    u  = np.array([1.0, 0.5])
    T  = 20.0

    for dt in [0.1, 0.5]:
        for name, step in [("euler", step_euler), ("rk4", step_rk4)]:
            xs = simulate(step, x0, u, dt, T)
            error = np.linalg.norm(xs[-1, :2] - true_circle(T, u[0], u[1]))
            print(f"{name:6s} dt={dt}: error = {error:.2e} m")
            plt.plot(xs[:,0], xs[:,1], label=f"{name} dt={dt}")

    t = np.linspace(0, T, 500)
    c = true_circle(t, 1.0, 0.5)
    plt.plot(c[0], c[1], 'k--', label='true')
    
    plt.legend()
    plt.show()
