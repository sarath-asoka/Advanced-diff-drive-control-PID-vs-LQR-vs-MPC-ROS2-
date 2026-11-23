import numpy as np
import matplotlib.pyplot as plt
from model import step_rk4
from reference import make_figure8
from tracking_error import tracking_error
import time
from mpc import build_mpc, N

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
    n = len(t)
    x = np.array(x0, dtype=float)
    xs, es = [], []

    if controller == "mpc":
        opti, U, X0, XR, UR, Xpred, J = build_mpc()      # build ONCE
        U_guess = np.zeros((2, N))
        solve_times = []

    for i in range(n):
        x_ref = [xr[i], yr[i], thr[i]]
        e = tracking_error(x, x_ref)

        if controller == "p":
            v, w = p_controller(e, vr[i], wr[i], *gains)
        elif controller == "lqr":
            v, w = lqr_controller(e, vr[i], wr[i])
        elif controller == "mpc":
            idx = [(i + k) % n for k in range(N + 1)]    # next N+1 ghost points (wraps around the lap)
            # TODO A: opti.set_value(X0, ...)
            # TODO B: opti.set_value(XR, ...)
            # TODO C: opti.set_value(UR, ...)
            opti.set_value(X0, x)                                          # A: where the robot IS now
            opti.set_value(XR, np.vstack([xr[idx], yr[idx], thr[idx]]))    # B: ghost's NEXT 2 s from step i
            opti.set_value(UR, np.vstack([vr[idx[:-1]], wr[idx[:-1]]]))    # C: ghost's commands, N of them
               
            opti.set_initial(U, U_guess)                 # warm start
            t0 = time.time()
            sol = opti.solve()
            solve_times.append(time.time() - t0)
            u_plan = sol.value(U)
            v, w = u_plan[:, 0]                          # use ONLY the first command
            U_guess = np.hstack([u_plan[:, 1:], u_plan[:, -1:]])   # shift for next step

        v, w = saturate(v, w)
        x = step_rk4(x, [v, w], dt)
        xs.append(x); es.append(e)

    if controller == "mpc":
        print(f"   MPC solve time: mean {1000*np.mean(solve_times):.1f} ms, "
              f"max {1000*np.max(solve_times):.1f} ms")
    return t, np.array(xs), np.array(es), (xr, yr)

if __name__ == "__main__":
    x0 = [-0.3, 0.3, 0]                       # offset start
    for name, controller in [("P",   "p"),
                             ("LQR", "lqr"),
                             ("MPC", "mpc")]:
        t, xs, es, (xr, yr) = run(x0, (1.0, 2.0, 2.0), controller=controller)
        rmse    = np.sqrt(np.mean(es[:, 1]**2))
        rmse_ss = np.sqrt(np.mean(es[t > 5, 1]**2))
        print(f"{name:4s}  full-lap RMSE = {rmse:.4f} m   steady-state RMSE = {rmse_ss*1000:.2f} mm")
        plt.plot(xs[:, 0], xs[:, 1], label=name)

    plt.plot(xr, yr, 'k--', label='reference')
    plt.axis('equal')
    plt.legend()
    plt.show()