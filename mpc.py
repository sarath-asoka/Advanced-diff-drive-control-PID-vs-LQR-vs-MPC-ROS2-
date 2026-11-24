import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from reference import make_figure8

N, DT, L = 40, 0.05, 0.5                 # 40 steps x 0.05 s = 2 s horizon
Q_AL, Q_CR, Q_TH = 1.0, 10.0, 1.0        # error weights (same as LQR)
R_V, R_W = 1.0, 1.0                      # effort weights

def f_ca(x, u):                          # your Day 1 model, with ca.cos / ca.sin
    return ca.vertcat(u[0]*ca.cos(x[2]), u[0]*ca.sin(x[2]), u[1])

def rk4_ca(x, u, dt):                    # your Day 1 RK4, unchanged
    k1 = f_ca(x, u)
    k2 = f_ca(x + 0.5*dt*k1, u)
    k3 = f_ca(x + 0.5*dt*k2, u)
    k4 = f_ca(x + dt*k3, u)
    return x + dt/6*(k1 + 2*k2 + 2*k3 + k4)

def build_mpc(N, obs=None, d_min=0.4):
    opti = ca.Opti()
    U  = opti.variable(2, N)             # UNKNOWNS: the N commands (v, w)
    X0 = opti.parameter(3)               # GIVEN each time: robot's current state
    XR = opti.parameter(3, N+1)          # GIVEN: ghost states for the horizon
    UR = opti.parameter(2, N)            # GIVEN: ghost commands (feedforward)
    PT = opti.parameter(3, 3)            # NEW: terminal weight P, given each step

    x = X0
    J = 0
    xs = [X0]
    for k in range(N):
        x = rk4_ca(x, U[:, k], DT)       # predict next state with the model
        xs.append(x)
        r = XR[:, k+1]                   # ghost at the same future time

        dx = x[0] - r[0]
        dy = x[1] - r[1]

        e_along = ca.cos(r[2])*dx + ca.sin(r[2])*dy
        e_cross = -ca.sin(r[2])*dx + ca.cos(r[2])*dy
        e_th2 = 2*(1 - ca.cos(x[2] - r[2]))          # heading penalty, no wrap needed

        dv = U[0, k] - UR[0, k]     # planned v  minus ghost v,  at step k
        dw = U[1, k] - UR[1, k]     # planned ω  minus ghost ω,  at step k

        J += Q_AL* e_along**2 + Q_CR * e_cross**2 + Q_TH * e_th2 + R_V * dv**2 + R_W * dw**2

        v, w = U[0, k], U[1, k]
        opti.subject_to(v + L/2*w <= 1)
        opti.subject_to(v - L/2*w <= 1)
        opti.subject_to(-v + L/2*w <= 1)
        opti.subject_to(-v - L/2*w <= 1)

        if obs is not None:
            # TODO: one line. The predicted position (x[0], x[1]) must stay at least
            #       d_min from the obstacle center obs = (x_o, y_o). Use the squared form.
            opti.subject_to( ((x[0]-obs[0])**2 + (x[1]-obs[1])**2) >= d_min**2 )

    # ── after the loop (no indentation inside the for) ──
    e_N = ca.vertcat(e_along, e_cross, ca.sin(x[2] - r[2]))   # error at the LAST predicted step
    J += (1/DT) * e_N.T @ PT @ e_N                            # added ONCE

    opti.minimize(J)
    opti.solver('ipopt', {'print_time': False, 'expand': True}, {'print_level': 0, 'sb': 'yes'})
    return opti, U, X0, XR, UR,PT, ca.horzcat(*xs), J

if __name__ == "__main__":
    opti, U, X0, XR, UR,PT, Xpred, J = build_mpc(N)
    t, xr, yr, thr, vr, wr = make_figure8(2.0, 20.0, DT)

    opti.set_value(X0, [-0.3, 0.3, 0])                                   # offset start
    opti.set_value(XR, np.vstack([xr[:N+1], yr[:N+1], thr[:N+1]]))       # ghost, next 2 s
    opti.set_value(UR, np.vstack([vr[:N], wr[:N]]))
    opti.set_value(PT, np.zeros((3, 3)))      # no terminal cost → must give the Day 4 result
    sol = opti.solve()

    u, xp = sol.value(U), sol.value(Xpred)
    print("cost       =", round(sol.value(J), 2))
    print("first cmd  =", np.round(u[:, 0], 3))
    print("max diamond=", round(np.max(np.abs(u[0]) + 0.25*np.abs(u[1])), 3))

    plt.plot(xr[:N+1], yr[:N+1], 'k--o', ms=3, label='ghost (next 2 s)')
    plt.plot(xp[0], xp[1], 'b-o', ms=3, label='MPC plan')
    plt.axis('equal'); plt.legend(); plt.show()