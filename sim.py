import numpy as np
import matplotlib.pyplot as plt
from model import step_rk4
from reference import make_figure8
from tracking_error import tracking_error

V_WHEEL_MAX, L = 1.0, 0.5

def saturate(v, w):
    """Scale (v, w) into the diamond |v| + (L/2)|w| <= V_WHEEL_MAX, keeping w/v."""
    m = abs(v) + (L/2) * abs(w)
    if m <= V_WHEEL_MAX:
        return v,w
    else:
        c = V_WHEEL_MAX/m
        return v/c, w/c

    
def p_controller(e, v_ref, w_ref, k_along, k_cross, k_th):
    """Feedforward + P feedback (formula sheet section 6). Returns v, w."""
    e_along, e_cross, e_th = e
    v = v_ref - k_along * e_along
    w = w_ref - k_cross * e_cross - k_th * e_th
    return v,w

def run(x0, gains, a=2.0, T_lap=20.0, dt=0.05):
    t, xr, yr, thr, vr, wr = make_figure8(a, T_lap, dt)
    x = np.array(x0, dtype=float)
    xs, es = [], []
    for i in range(len(t)):
        x_ref = [xr[i], yr[i], thr[i]]          # the ghost at time t[i]
        e = tracking_error(x, x_ref)
        v, w = p_controller(e, vr[i], wr[i], *gains)
        v, w = saturate(v, w)
        x = step_rk4(x, [v, w], dt)

        xs.append(x); es.append(e)
    return t, np.array(xs), np.array(es), (xr, yr)

if __name__ == "__main__":
    for name, x0, gains in [
        ("ff only, perfect start", [0, 0, np.pi/4],  (0, 0, 0)),
        ("ff only, offset start",  [-0.3, 0.3, 0],   (0, 0, 0)),
        ("ff + P,  offset start",  [-0.3, 0.3, 0],   (1.0, 2.0, 2.0)),
    ]:
        t, xs, es, (xr, yr) = run(x0, gains)
        rmse = np.sqrt(np.mean(es[:, 1]**2))
        print(f"{name:24s}  cross-track RMSE = {rmse:.4f} m")
        plt.plot(xs[:, 0], xs[:, 1], label=name)
    plt.plot(xr, yr, 'k--', label='reference')
    plt.axis('equal'); plt.legend(); plt.show()