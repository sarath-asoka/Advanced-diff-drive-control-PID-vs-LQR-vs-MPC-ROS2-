import numpy as np
import matplotlib.pyplot as plt
from model import step_rk4
from reference import make_figure8
from tracking_error import tracking_error

V_WHEEL_MAX, L = 1.0, 0.5

from scipy.linalg import solve_continuous_are

Q = np.diag([1.0, 10.0, 1.0])   # error weights: [e_along, e_cross, e_θ]  (we care most about e_cross)
R = np.diag([1.0, 1.0])         # effort weights: [δv, δω]
B = np.array([[1, 0],
              [0, 0],
              [0, 1]], dtype=float)

def saturate(v, w):
    """Scale (v, w) into the diamond |v| + (L/2)|w| <= V_WHEEL_MAX, keeping w/v."""
    m = abs(v) + (L/2) * abs(w)
    if m <= V_WHEEL_MAX:
        return v,w
    else:
        c = V_WHEEL_MAX/m
        return v*c, w*c


def lqr_controller(e, v_ref, w_ref):
    A = np.array([[0, w_ref, 0], [-w_ref, 0, v_ref], [0, 0, 0]])
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P # K = R⁻¹ Bᵀ P   
    du = -K@e
    return v_ref + du[0], w_ref + du[1]
    
def p_controller(e, v_ref, w_ref, k_along, k_cross, k_th):
    """Feedforward + P feedback (formula sheet section 6). Returns v, w."""
    e_along, e_cross, e_th = e
    v = v_ref - k_along * e_along
    w = w_ref - k_cross * e_cross - k_th * e_th
    return v,w

def run(x0, gains, controller="p", a=2.0, T_lap=20.0, dt=0.05):
    t, xr, yr, thr, vr, wr = make_figure8(a, T_lap, dt)
    x = np.array(x0, dtype=float)
    xs, es = [], []
    for i in range(len(t)):
        x_ref = [xr[i], yr[i], thr[i]]          # the ghost at time t[i]
        e = tracking_error(x, x_ref)
        if controller == "p":
            v, w = p_controller(e, vr[i], wr[i], *gains)
        else:
            v, w = lqr_controller(e, vr[i], wr[i])      

        v, w = saturate(v, w)
        x = step_rk4(x, [v, w], dt)

        xs.append(x); es.append(e)
    return t, np.array(xs), np.array(es), (xr, yr)

if __name__ == "__main__":
    x0 = [-0.3, 0.3, 0]                       # offset start
    for name, controller in [("P",   "p"),
                             ("LQR", "lqr")]:
        t, xs, es, (xr, yr) = run(x0, (1.0, 2.0, 2.0), controller=controller)
        rmse    = np.sqrt(np.mean(es[:, 1]**2))
        rmse_ss = np.sqrt(np.mean(es[t > 5, 1]**2))
        print(f"{name:4s}  full-lap RMSE = {rmse:.4f} m   steady-state RMSE = {rmse_ss*1000:.2f} mm")
        plt.plot(xs[:, 0], xs[:, 1], label=name)
    plt.plot(xr, yr, 'k--', label='reference')
    plt.axis('equal')
    plt.legend()
    plt.show()