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

def run(x0, gains, controller="p", a=2.0, T_lap=20.0, dt=0.05, N=40, obs=None):   # add obs=None
    t, xr, yr, thr, vr, wr = make_figure8(a, T_lap, dt)
    n = len(t)
    x = np.array(x0, dtype=float)
    xs, es = [], []

    if controller == "mpc":
        opti, U, X0, XR, UR, PT, Xpred, J = build_mpc(N, obs)
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

            opti.set_value(X0, x)                                          # A: where the robot IS now
            opti.set_value(XR, np.vstack([xr[idx], yr[idx], thr[idx]]))    # B: ghost's NEXT 2 s from step i
            opti.set_value(UR, np.vstack([vr[idx[:-1]], wr[idx[:-1]]]))    # C: ghost's commands, N of them

            opti.set_initial(U, U_guess)                 # warm start
            t0 = time.time()
            
            j = idx[-1]
            A_end = np.array([[0, wr[j], 0], [-wr[j], 0, vr[j]], [0, 0, 0]])
            P_end = solve_continuous_are(A_end, B, Q, R)
            opti.set_value(PT, P_end)

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

# if __name__ == "__main__":
    x0 = [-0.3, 0.3, 0]                       # offset start
    # for name, controller in [("P",   "p"),
    #                          ("LQR", "lqr"),
    #                          ("MPC", "mpc")]:
    #     t, xs, es, (xr, yr) = run(x0, (1.0, 2.0, 2.0), controller=controller)
    #     rmse    = np.sqrt(np.mean(es[:, 1]**2))
    #     rmse_ss = np.sqrt(np.mean(es[t > 5, 1]**2))
    #     print(f"{name:4s}  full-lap RMSE = {rmse:.4f} m   steady-state RMSE = {rmse_ss*1000:.2f} mm")
    #     plt.plot(xs[:, 0], xs[:, 1], label=name)

    # --- Experiment 2: horizon sweep ---
    # for N in [40, 20, 10, 5]:
    #     print(f"N = {N}  (horizon {N*0.05:.2f} s)")
    #     t, xs, es, (xr, yr) = run(x0, None, controller="mpc", N=N)
    #     rmse    = np.sqrt(np.mean(es[:, 1]**2))
    #     rmse_ss = np.sqrt(np.mean(es[t > 5, 1]**2))
    #     print(f"   full-lap RMSE = {rmse:.4f} m   steady-state RMSE = {rmse_ss*1000:.2f} mm")
    #     plt.plot(xs[:, 0], xs[:, 1], label=f"MPC N={N}")

    # plt.plot(xr, yr, 'k--', label='reference')
    # plt.axis('equal')
    # plt.legend()
    # plt.show()

# if __name__ == "__main__":
#     x0 = [-1.0, 1.0, np.pi]                   # HARD start: 1.4 m away, facing backwards
#     runs = [("P", "p", 40), ("LQR", "lqr", 40), ("MPC N=10", "mpc", 10), ("MPC N=40", "mpc", 40)]

#     fig, ax = plt.subplots(1, 3, figsize=(16, 5))
#     for name, controller, N in runs:
#         print(name)
#         t, xs, es, (xr, yr) = run(x0, (1.0, 2.0, 2.0), controller=controller, N=N)
#         e_cr = np.abs(es[:, 1])
#         rmse = np.sqrt(np.mean(e_cr**2))
#         big = np.where(e_cr >= 0.05)[0]                      # steps still more than 5 cm off
#         t_settle = t[big[-1] + 1] if len(big) and big[-1] + 1 < len(t) else 0.0
#         print(f"   full-lap RMSE = {rmse:.4f} m   max error = {e_cr.max():.2f} m   settles at t = {t_settle:.1f} s")

#         # speed along the robot's own heading (negative = driving backwards)
#         all_x = np.vstack([x0, xs])
#         v_est = (np.diff(all_x[:, 0])*np.cos(all_x[:-1, 2]) + np.diff(all_x[:, 1])*np.sin(all_x[:-1, 2])) / 0.05

#         k = t < 10                                           # first 10 s only
#         ax[0].plot(xs[k, 0], xs[k, 1], label=name)
#         ax[1].plot(t[k], e_cr[k], label=name)
#         ax[2].plot(t[k], v_est[k], label=name)

#     ax[0].plot(xr, yr, 'k--', label='reference'); ax[0].axis('equal'); ax[0].set_title('path (first 10 s)')
#     ax[1].set_title('|e_cross| [m]'); ax[1].set_xlabel('t [s]')
#     ax[2].set_title('speed v [m/s]  (negative = reversing)'); ax[2].set_xlabel('t [s]')
#     for a in ax: a.legend(); a.grid(True)
#     plt.tight_layout(); plt.show()

if __name__ == "__main__":
    x0 = [0.0, 0.0, np.pi/4]                  # start exactly on the ghost
    s = 2*np.pi*9/20                          # path point at t = 9 s
    obs = (2*np.sin(s), 2*np.sin(s)*np.cos(s))
    R_OBS, R_ROBOT, MARGIN = 0.2, 0.15, 0.05  # d_min = 0.4

    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    for N in [10, 20]:
        print(f"MPC N={N}")
        t, xs, es, (xr, yr) = run(x0, None, controller="mpc", N=N, obs=obs)
        dist = np.hypot(xs[:, 0] - obs[0], xs[:, 1] - obs[1])
        print(f"   closest approach = {dist.min():.3f} m (must be >= 0.400)")
        print(f"   max |e_cross| = {np.abs(es[:, 1]).max():.2f} m   most behind ghost: e_along = {es[:, 0].min():.2f} m")

        all_x = np.vstack([x0, xs])
        v_est = (np.diff(all_x[:, 0])*np.cos(all_x[:-1, 2]) + np.diff(all_x[:, 1])*np.sin(all_x[:-1, 2])) / 0.05
        ax[0].plot(xs[:, 0], xs[:, 1], label=f"N={N}")
        ax[1].plot(t, es[:, 1], label=f"e_cross N={N}"); ax[1].plot(t, es[:, 0], '--', label=f"e_along N={N}")
        ax[2].plot(t, v_est, label=f"N={N}")

    ax[0].plot(xr, yr, 'k--', label='reference')
    ax[0].add_patch(plt.Circle(obs, R_OBS, color='r'))
    ax[0].add_patch(plt.Circle(obs, 0.4, color='r', fill=False, ls=':'))
    ax[0].axis('equal'); ax[0].set_title('path')
    ax[1].set_title('errors [m]'); ax[2].set_title('speed v [m/s]')
    for a in ax[1:]: a.set_xlabel('t [s]')
    for a in ax: a.legend(); a.grid(True)
    plt.tight_layout(); plt.show()