import numpy as np
import dasslc

def model0(t, y, yp):  # --------------- Minimum of 3 input arguments
    # ------------- Always allocate res as a numpy array, even if it has len = 1.
    res = np.empty(1)
    res[0] = yp[0] + 2 * y[0]  # ------- Declare the residual
    ires = 0  # ---------------------- Set ires = 0 if everything is ok
    # -------------- Beware: ires must always be returned as second output.
    return res, ires


def dummy_res(tf):
	return tf*2.0

y0 = np.array([1.0])  # ------------------ Initial condition
# ------------------ Derivatives at initial condition (optional)
yp0 = np.array([1.0])
tsp = np.array([0.0, 1.0])

dasslc.hello(8.0)

print('call res')
a = dasslc.call_res(dummy_res, 4.0)
print(a)

print('call with numpy array')
dasslc.solve(model0, tsp, y0)
