import numpy as np
import matplotlib.pyplot as plt

def make_figure8(a, T_lap, dt):
    """Returns t, x, y, theta, v, omega  (each a 1-D array) for one lap."""
    t = np.arange(0, T_lap, dt)
    s_dot = 2*np.pi / T_lap
    s = s_dot * t
    x = a * np.sin(s)
    y = a * np.sin(s) * np.cos(s)

    x_dot = a * np.cos(s) * s_dot
    y_dot = a * np.cos(2*s) * s_dot
    x_ddot = -a * np.sin(s) * np.square(s_dot)
    y_ddot = -2 * a * np.sin(2*s) * np.square(s_dot)

    theta = np.arctan2(y_dot,x_dot)
    v = np.sqrt(x_dot**2 + y_dot**2)
    omega = (x_dot*y_ddot - y_dot*x_ddot)/(x_dot**2 + y_dot**2)
    return t, x, y, theta, v, omega

if __name__ == "__main__":
    a, T_lap, dt = 2.0, 1.0, 0.05
    t, x, y, theta, v, omega = make_figure8(a, T_lap, dt)

    # 1. Path + heading arrows (every 20th point)
    plt.figure()
    plt.plot(x, y)
    k = slice(None, None, 20)
    plt.quiver(x[k], y[k], np.cos(theta[k]), np.sin(theta[k]))
    plt.axis('equal'); plt.title('figure-8')

    # 2. v and omega vs time
    plt.figure()
    plt.plot(t, v, label='v [m/s]')
    plt.plot(t, omega, label='ω [rad/s]')
    plt.xlabel('time [s]'); plt.legend(); plt.grid(True)

    # 3. Feasibility (diamond): v_wheel_max = 1.0, L = 0.5
    margin = np.max(v + 0.25 * np.abs(omega))
    print(f"T_lap={T_lap}: margin = {margin:.3f}  ->  feasible: {margin <= 1.0}")

    plt.show()
    plt.show()