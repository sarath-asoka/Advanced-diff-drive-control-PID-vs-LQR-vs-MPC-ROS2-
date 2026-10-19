import numpy as np

def wrap(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def tracking_error(x, x_ref):
    """x = robot [x, y, θ],  x_ref = reference [x_ref, y_ref, θ_ref].
       Returns [e_along, e_cross, e_θ]."""
    # TODO: dx, dy
    # TODO: e_along, e_cross   (section 5)
    # TODO: e_θ with wrap
    x = np.array(x)
    x_ref = np.array(x_ref)

    dx, dy, dth = x -x_ref

    e_along = np.cos(x_ref[2])*dx + np.sin(x_ref[2])*dy
    e_cross = -np.sin(x_ref[2])*dx + np.cos(x_ref[2])*dy
    e_th = wrap(dth)

    return np.array([e_along, e_cross, e_th])

if __name__ == "__main__":
    tests = {
        "A": ([0, 0.3, 0],               [0, 0, 0],                  [0, 0.3, 0]),
        "B": ([2.3, 0, -np.pi/2],        [2, 0, -np.pi/2],           [0, 0.3, 0]),
        "C": ([0.5, 0, 0],               [0, 0, 0],                  [0.5, 0, 0]),
        "D": ([0, 0, np.deg2rad(179)],   [0, 0, np.deg2rad(-179)],   [0, 0, np.deg2rad(-2)]),
    }
    for name, (x, x_ref, expected) in tests.items():
        e = tracking_error(x, x_ref)
        ok = np.allclose(e, expected)
        print(f"{name}: {np.round(e, 4)}  expected {np.round(expected, 4)}  ->  {'PASS' if ok else 'FAIL'}")