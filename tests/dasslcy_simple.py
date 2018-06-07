import dasslcy
import numpy as np

res_global = np.zeros(5)  # pre-allocating globally

def model3(t, y, yp, par):  # ----------- The parameter may be a whole class
    res = np.empty(5)
    res[0] = yp[0] - y[2]
    res[1] = yp[1] - y[3]
    res[2] = yp[2] + y[4]*y[0]
    res[3] = yp[3] + y[4]*y[1] + par.g
    if (par.dae == 3):
        res[4] = y[0]*y[0] + y[1]*y[1] - par.L*par.L
        ires = 0
    elif (par.dae == 2):
        res[4] = y[0]*y[2] + y[1]*y[3]
        ires = 0
    elif (par.dae == 1):
        res[4] = y[2]**2 + y[3]**2 - par.g*y[1] - par.L**2*y[4]
        ires = 0
    elif (par.dae == 0):
        res[4] = yp[4] + 3*y[3]*par.g/par.L**2
        ires = 0
    else:
        print("Invalid index.")
        ires = -1
    return res, ires

def model3_mod(t, y, yp, par):
    #res = np.empty(5)
    # res = res_global
    global res_global
    res_global[0] = yp[0] - y[2]
    res_global[1] = yp[1] - y[3]
    res_global[2] = yp[2] + y[4]*y[0]
    res_global[3] = yp[3] + y[4]*y[1] + par.g
    if (par.dae == 3):
        res_global[4] = y[0]*y[0] + y[1]*y[1] - par.L*par.L
        ires = 0
    elif (par.dae == 2):
        res_global[4] = y[0]*y[2] + y[1]*y[3]
        ires = 0
    elif (par.dae == 1):
        res_global[4] = y[2]**2 + y[3]**2 - par.g*y[1] - par.L**2*y[4]
        ires = 0
    elif (par.dae == 0):
        res_global[4] = yp[4] + 3*y[3]*par.g/par.L**2
        ires = 0
    else:
        print("Invalid index.")
        ires = -1
    return res_global, ires

class pend_par:  # ----------------#|
    g = 9.81  # | Defining the parameter class for
    L = 1.0  # | the pendulum model
    dae = 3  # |
    # res = np.empty(5) #pre-allocating

def set_scenario():
    t0 = np.linspace(0.0, 50.0, 10000.0)
    y0 = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    yp0 = None
    par = pend_par()
    atol = 1e-10
    rtol = 1e-8
    # ---- The dependent variable index vector (needed for high index DAE)
    index = np.array([1, 1, 2, 2, 3])
    return (t0, y0, yp0, par, rtol, atol, index)


def run_scenario3():
    args = set_scenario()
    t, y, yp = dasslcy.solve(model3, *args)
    y_ref = [0.93226827, -0.36176772, -0.96384619, -2.48381292, 10.64973574]
    assert(np.all(np.isclose(y[-1], y_ref)))
    pass

def run_scenario3_mod():
    args = set_scenario()
    t, y, yp = dasslcy.solve(model3_mod, *args)
    y_ref = [0.93226827, -0.36176772, -0.96384619, -2.48381292, 10.64973574]
    assert(np.all(np.isclose(y[-1], y_ref)))

def main():
    run_scenario3_mod()

if __name__ == '__main__':
    main()

