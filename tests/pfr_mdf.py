import timeit
import numpy as np
try:
    import matplotlib.pyplot as plt
    has_plt = True
except ImportError:
    has_plt = False
import dasslcy
from numba import jit


class pfr():
    D = 1.0
    vz = 1.0
    k = 1.0
    Cf = 1.0
    #N = 20 #intervals
    z0 = 0.0
    zf = 1.0

    def __init__(self, N = 20):
        self.N = N
        self.h = self.get_h()
        self.initialize_intermediaries()

    def get_h(self):
        return (self.zf - self.z0) / self.N

    def initialize_intermediaries(self):
        self.r = np.empty(self.N)

def pfr_as_list(par):
    pfr_list_version = [par.D, par.vz, par.k, par.Cf, par.N, par.h]
    return pfr_list_version

def model_pfr(t, y, yp, par):
    res = np.empty(par.N)
    r = par.r
    N = par.N
    dCi = yp
    Ci = y
    aux1 = par.D / (par.vz * par.h)
    C0 = 1.0/(1.0 + aux1) * (aux1*Ci[0] + par.Cf)
    CNp1 = Ci[N - 1]
    r[:] = par.k*Ci
    aux2 = par.D / par.h**2
    aux3 = par.vz / (2 * par.h)
    res[0] = aux2 * (Ci[1] - 2.0 * Ci[0] + C0) - aux3 * (Ci[1] - C0) + r[0] - dCi[0]
    for i in np.arange(1, N - 1):
        tt1 = aux2 * (Ci[i + 1] - 2.0 * Ci[i] + Ci[i - 1])
        tt2 = -aux3 * (Ci[i+1] - Ci[i-1]) + r[i]
        res[i] = tt1 + tt2 - dCi[i]
    res[N-1] = aux2 * (CNp1 - 2.0 * Ci[N-1] + Ci[N-2]) - aux3 * (CNp1 - Ci[N - 2]) + r[N - 1] - dCi[N-1]
    ires = 0
    return res, ires

def model_pfr_shared(t, y, yp, par, res):
    r = par.r
    N = par.N
    dCi = yp
    Ci = y
    aux1 = par.D / (par.vz * par.h)
    C0 = 1.0 / (1.0 + aux1) * (aux1 * Ci[0] + par.Cf)
    CNp1 = Ci[N - 1]
    r[:] = par.k * Ci
    aux2 = par.D / par.h**2
    aux3 = par.vz / (2 * par.h)
    res[0] = aux2 * (Ci[1] - 2.0 * Ci[0] + C0) - \
        aux3 * (Ci[1] - C0) + r[0] - dCi[0]
    for i in np.arange(1, N - 1):
        tt1 = aux2 * (Ci[i + 1] - 2.0 * Ci[i] + Ci[i - 1])
        tt2 = -aux3 * (Ci[i + 1] - Ci[i - 1]) + r[i]
        res[i] = tt1 + tt2 - dCi[i]
    res[N - 1] = aux2 * (CNp1 - 2.0 * Ci[N - 1] + Ci[N - 2]) - \
        aux3 * (CNp1 - Ci[N - 2]) + r[N - 1] - dCi[N - 1]
    ires = 0
    return res, ires

# def model_pfr_numba(t, y, yp):
#     D = 1.0
#     vz = 1.0
#     k = 1.0
#     Cf = 1.0
#     N = 20
#     h = 1.0
#     # r = par[6]
#     r = np.empty(N)
#     res = np.empty(N)

#     dCi = yp
#     Ci = y
#     aux1 = D / (vz * h)
#     C0 = 1.0 / (1.0 + aux1) * (aux1 * Ci[0] + Cf)
#     CNp1 = Ci[N - 1]
#     r[:] = k * Ci
#     aux2 = D / h**2
#     aux3 = vz / (2 * h)
#     res[0] = aux2 * (Ci[1] - 2.0 * Ci[0] + C0) - \
#         aux3 * (Ci[1] - C0) + r[0] - dCi[0]
#     for i in np.arange(1, N - 1):
#         tt1 = aux2 * (Ci[i + 1] - 2.0 * Ci[i] + Ci[i - 1])
#         tt2 = -aux3 * (Ci[i + 1] - Ci[i - 1]) + r[i]
#         res[i] = tt1 + tt2 - dCi[i]
#     res[N - 1] = aux2 * (CNp1 - 2.0 * Ci[N - 1] + Ci[N - 2]) - \
#         aux3 * (CNp1 - Ci[N - 2]) + r[N - 1] - dCi[N - 1]
#     ires = 0
#     return res, ires
#     return 0, 0

# def model_numba_jit():
#     model_pfr_jit = jit(model_pfr_numba, nopython=True)
#     return model_pfr_jit

######################################### Solve model2 ###################
    #print('------- Solve model_pfr  ---------- ')

#--------------------------------------------
# 	 SETUPS
#--------------------------------------------

def setup_base(N):
    par = pfr(N)
    t0 = np.array([5.0])
    y0 = np.zeros(par.N)
    yp0 = None
    atol = 1e-8
    rtol = 1e-6
    return [t0, y0, yp0, par, rtol, atol]

def setup_solver_regular(N):
    return tuple([model_pfr] + setup_base(N))

def setup_solver_shared(N):
    return tuple([model_pfr_shared] + setup_base(N))

# def setup_solver_numba(N):
#     spt = setup_base(N)
#     spt[3] = None
#     jitted_model = model_numba_jit()
#     #Test
#     jitted_model(0.0, np.zeros(20), np.array(20))
#     return tuple([jitted_model] + spt)

def run_solver(N):
    args = setup_solver_regular(N)
    dasslcy.solve(*args)
    return

def run_solver_numba(N):
    args = setup_solver_numba(N)
    dasslcy.solve(*args)

#--------------------------------------------
# 	 TIMING FUNCS
#--------------------------------------------

def timing_regular(N, rpt_args):
    setup_str = '''
from __main__ import setup_solver_regular, dasslcy
args = setup_solver_regular({})
    '''.format(N)
    # tt = timeit.timeit('dasslcy.solve(*args)', setup=setup_str, number=1)
    tt = run_timeit('dasslcy.solve(*args)', setup=setup_str,
                    rpt_args=rpt_args)
    print('Regular solver: {} s'.format(tt))

def timing_shared(N, rpt_args):
    setup_str = '''
from __main__ import setup_solver_shared, dasslcy;
args = setup_solver_shared({})
    '''.format(N)
    tt = run_timeit('dasslcy.solve(*args, share_res=1)',
                    setup=setup_str, rpt_args=rpt_args)
    print('Shared mem solver: {} s'.format(tt))

# def timing_numba(N, rpt_args):
#     setup_str = '''
# from __main__ import setup_solver_numba, dasslcy;
# args = setup_solver_numba({})
#     '''.format(N)
#     tt = run_timeit('dasslcy.solve(*args)',
#                     setup=setup_str, rpt_args=rpt_args)
#     print('DASSLCY + NUMBA: {} s'.format(tt))

def run_timeit(stmt, setup, rpt_args):
    tmr = timeit.Timer(stmt, setup).repeat(*rpt_args)
    min_val = min(tmr)
    #print(tmr)
    return min_val

def timing():
    N = 20
    args_tt = (N, (1, 1))
    timing_regular(*args_tt)
    timing_shared(*args_tt)
    # timing_numba(*args_tt)






if __name__ == '__main__':
    timing()
    #run_solver_numba(20)

#--------------------------------------------
# 	 helper funcs
#--------------------------------------------

def visualize(t, y, yp, par):
    if has_plt:
        plt.figure(3)
        idxsplt = np.linspace(0, par.N - 1, 4, dtype=int)
        labels = ['idx:{}'.format(ii) for ii in idxsplt]
        curves = plt.plot(t, y[:, idxsplt], '-o')
        plt.ylabel('y')
        plt.xlabel('time')
        plt.title('PFR Solution')
        plt.legend(curves, labels)
    print('States at final time: {}'.format(y[-1, :]))

    if has_plt:
        plt.show()
