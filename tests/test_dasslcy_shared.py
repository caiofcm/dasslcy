import pytest
import dasslcy
import numpy as np

############################################
###########################################
#
#
# 	 FIXTURES 	 
#
#
###########################################
###########################################

@pytest.fixture()
def scenario0():
    def model0(t, y, yp, res):  # --------------- Minimum of 3 input arguments
        # ------------- Always allocate res as a numpy array, even if it has len = 1.
        # res = np.empty(1)
        res[0] = yp[0] + 2*y[0]  # ------- Declare the residual
        ires = 0  # ---------------------- Set ires = 0 if everything is ok
        # -------------- Beware: ires must always be returned as second output.
        return res, ires
    t0 = np.array([0.0, 1.0])
    y0 = np.array([1.0])
    return (model0, t0, y0)

@pytest.fixture()
def scenario1():
    def model1(t, y, yp, res):  # --------------- Just another example
        # res = np.empty(2)
        res[0] = yp[0]-20*np.cos(20*t)
        res[1] = yp[1]+20*np.sin(20*t)
        return res, 0  # ----------------- ires can be a literal
    t0 = np.linspace(0.0, 1.0, 100)
    y0 = np.array([0.0, 1.0])
    # ------------------ Derivatives at initial condition (optional)
    yp0 = np.array([1.0, 0.0])
    return (model1, t0, y0)


@pytest.fixture()
def scenario2():
    def model2(t, y, yp, par, res):  # ------------- Maximum of 4 input arguments
        # res = np.empty(3)

        k1 = par[0]  # ||||||||||||||||||||||||||||||||||||||||||||||||
        k2 = par[1]  # |                                              |
        Ca = y[0]
        dCa = yp[0]  # --------#| aliasing is optional, but always encoraged   |
        Cb = y[1]
        dCb = yp[1]  # |                                              |
        Cc = y[2]
        dCc = yp[2]  # ||||||||||||||||||||||||||||||||||||||||||||||||

        res[0] = -k1*Ca - dCa
        res[1] = k1*Ca - k2*Cb - dCb
        res[2] = k2*Cb - dCc
        ires = 0
        return res, ires
    t0 = np.array([500.0])
    y0 = np.array([1.0, 0.0, 0.0])
    yp0 = None
    par = np.array([0.01, 0.02])  # ------- The optional parameter vector
    atol = 1e-8  # ----------------------- The absolute tolerance
    rtol = 1e-6  # ----------------------- The relative tolerance
    return (model2, t0, y0, yp0, par, rtol, atol)

@pytest.fixture()
def scenario3():
    def model3(t, y, yp, par, res):  # ----------- The parameter may be a whole class
        # res = np.empty(5)
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
    class pend_par:  # ----------------#|
        g = 9.81  # | Defining the parameter class for
        L = 1.0  # | the pendulum model
        dae = 3  # |
    t0 = np.linspace(0.0, 50.0, 10000.0)
    y0 = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    yp0 = None
    par = pend_par()
    atol = 1e-10
    rtol = 1e-8
    # ---- The dependent variable index vector (needed for high index DAE)
    index = np.array([1, 1, 2, 2, 3])
    return (model3, t0, y0, yp0, par, rtol, atol, index)


############################################
###########################################
#
#
# 	 TESTS 	 
#
#
###########################################
###########################################


def test_scenario0_shared_res(scenario0):
    t, y, yp = dasslcy.solve(*scenario0, share_res=1)
    assert(np.isclose(y[-1], 0.1353356))


def test_scenario1_shared_res(scenario1):
    t, y, yp = dasslcy.solve(*scenario1, share_res=1)
    assert(np.all(np.isclose(y[-1], [0.91294581, 0.40808469])))


def test_scenario2_shared_res(scenario2):
    t, y, yp = dasslcy.solve(*scenario2, share_res=1)
    assert(np.all(np.isclose(y[-1], [0.00673799, 0.00669256, 0.98656945])))


def test_scenario3_shared_res(scenario3):
    t, y, yp = dasslcy.solve(*scenario3, share_res=1)
    y_ref = [0.93226827, -0.36176772, -0.96384619, -2.48381292, 10.64973574]
    assert(np.all(np.isclose(y[-1], y_ref, rtol=1e-3)))


