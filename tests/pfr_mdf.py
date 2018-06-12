import timeit
import numpy as np
try:
    import matplotlib.pyplot as plt
    has_plt = True
except ImportError:
    has_plt = False
import dasslcy
from numba import jit
import pytest
import pyximport; pyximport.install(
    setup_args={'include_dirs': np.get_include()})
import pfr_cython_model

def base_model_calculations(res, y, yp, N, D, vz, k, Cf, h):
    dCi = yp
    Ci = y
    aux1 = D / (vz * h)
    C0 = 1.0 / (1.0 + aux1) * (aux1 * Ci[0] + Cf)
    CNp1 = Ci[N - 1]
    aux2 = D / h**2
    aux3 = vz / (2 * h)
    res[0] = aux2 * (Ci[1] - 2.0 * Ci[0] + C0) - \
        aux3 * (Ci[1] - C0) + k * Ci[0] - dCi[0]
    for i in np.arange(1, N - 1):
        tt1 = aux2 * (Ci[i + 1] - 2.0 * Ci[i] + Ci[i - 1])
        tt2 = -aux3 * (Ci[i + 1] - Ci[i - 1]) + k * Ci[i]
        res[i] = tt1 + tt2 - dCi[i]
    res[N - 1] = aux2 * (CNp1 - 2.0 * Ci[N - 1] + Ci[N - 2]) - \
        aux3 * (CNp1 - Ci[N - 2]) + k * Ci[N - 1] - dCi[N - 1]
    pass

def numpy_broadcast_model_calc(res, y, yp, N, D, vz, k, Cf, h):
    dCi = yp
    Ci = y
    aux1 = D / (vz * h)
    C0 = 1.0 / (1.0 + aux1) * (aux1 * Ci[0] + Cf)
    CNp1 = Ci[N - 1]
    aux2 = D / h**2
    aux3 = vz / (2 * h)
    res[0] = aux2 * (Ci[1] - 2.0 * Ci[0] + C0) - \
        aux3 * (Ci[1] - C0) + k * Ci[0] - dCi[0]
    tt1 = aux2 * (Ci[2:] - 2.0 * Ci[1:-1] + Ci[0:-2])
    tt2 = -aux3 * (Ci[2:] - Ci[0:-2]) + k * Ci[1:-1]
    res[1:-1] = tt1 + tt2 - dCi[1:-1]

    res[N - 1] = aux2 * (CNp1 - 2.0 * Ci[N - 1] + Ci[N - 2]) - \
        aux3 * (CNp1 - Ci[N - 2]) + k * Ci[N - 1] - dCi[N - 1]
    pass

def get_h(zf, z0, intervals):
        return (zf - z0) / intervals

class pfr():


    def __init__(self, N = 20):
        self.D = 1.0
        self.vz = 1.0
        self.k = 1.0
        self.Cf = 1.0
        self.z0 = 0.0
        self.zf = 1.0
        self.N = N
        self.h = self.get_h()
        #self.initialize_intermediaries()

    def get_h(self):
        return (self.zf - self.z0) / self.N

    # def initialize_intermediaries(self):
    #     self.r = np.empty(self.N)

def model_pfr(t, y, yp, par):
    res = np.empty(par.N)
    N = par.N
    base_model_calculations(
            res, y, yp, N, par.D, par.vz, par.k, par.Cf, par.h)
    return res, 0

def model_pfr_shared(t, y, yp, par, res):
    N = par.N
    base_model_calculations(
            res, y, yp, N, par.D, par.vz, par.k, par.Cf, par.h)
    return res, 0


def model_pfr_shared_direct(t, y, yp, par, res):
    N = par.N
    D, vz, k, Cf, h = par.D, par.vz, par.k, par.Cf, par.h
    dCi = yp
    Ci = y
    aux1 = D / (vz * h)
    C0 = 1.0 / (1.0 + aux1) * (aux1 * Ci[0] + Cf)
    CNp1 = Ci[N - 1]
    aux2 = D / h**2
    aux3 = vz / (2 * h)
    res[0] = aux2 * (Ci[1] - 2.0 * Ci[0] + C0) - \
        aux3 * (Ci[1] - C0) + k * Ci[0] - dCi[0]
    for i in np.arange(1, N - 1):
        tt1 = aux2 * (Ci[i + 1] - 2.0 * Ci[i] + Ci[i - 1])
        tt2 = -aux3 * (Ci[i + 1] - Ci[i - 1]) + k * Ci[i]
        res[i] = tt1 + tt2 - dCi[i]
    res[N - 1] = aux2 * (CNp1 - 2.0 * Ci[N - 1] + Ci[N - 2]) - \
        aux3 * (CNp1 - Ci[N - 2]) + k * Ci[N - 1] - dCi[N - 1]
    return res, 0


def model_pfr_shared_numpy_broadcast(t, y, yp, par, res):
    N = par.N
    numpy_broadcast_model_calc(
        res, y, yp, N, par.D, par.vz, par.k, par.Cf, par.h)
    return res, 0

def numba_model_pfr(N):
    PFR = pfr(N)
    D = PFR.D
    vz = PFR.vz
    k = PFR.k
    Cf = PFR.Cf
    h = PFR.h

    jitted_base_model = jit(base_model_calculations, nopython=True)

    def model_nmb_pfr(t, y, yp, res):
        #res = np.empty(N)
        jitted_base_model(
            res, y, yp, N, D, vz, k, Cf, h)
        ires = 0
        return res, ires
    return model_nmb_pfr

def model_numba_jit(N = 20):
    func = numba_model_pfr(N)
    model_pfr_jit = jit(func, nopython=True)
    return model_pfr_jit


def model_pfr_shared_cython(t, y, yp, par, res):
    N = par.N
    pfr_cython_model.base_model_calculations(
        res, y, yp, N, par.D, par.vz, par.k, par.Cf, par.h)
    return res, 0

# def model_pfr_typed_cython(t, y, yp, res):


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

def setup_solver_shared_np_broadcast(N):
    return tuple([model_pfr_shared_numpy_broadcast] + setup_base(N))

def setup_solver_shared_cython(N):
    return tuple([model_pfr_shared_cython] + setup_base(N))

def setup_solver_numba(N):
    spt = setup_base(N)
    spt[3] = None
    jitted_model = model_numba_jit(N)
    #Test
    #jitted_model(0.0, np.zeros(20), np.array(20))
    return tuple([jitted_model] + spt)

def setup_solver_typed_cython(N):
    pfr_cython_model.initialize_cy_pfr_model(N)
    spt = setup_base(N)
    spt[3] = None
    #spt_rm = spt[0:3]
    fun = pfr_cython_model.cython_model_pfr_shared
    return tuple([fun] + spt)

#--------------------------------------------
# 	 RUNNERS
#--------------------------------------------

def run_solver(N):
    args = setup_solver_regular(N)
    t, y, yp = dasslcy.solve(*args)
    return

def run_solver_numba(N):
    args = setup_solver_numba(N)
    dasslcy.solve(*args, share_res=1)

#--------------------------------------------
# 	 TIMING FUNCS
#--------------------------------------------

def run_timeit(stmt, setup, rpt_args, logic = None):
    tmr = timeit.Timer(stmt, setup).repeat(*rpt_args)
    print(stmt, tmr)
    if logic is None:
        val = min(tmr)
    else:
        val = logic(tmr)
    #print(tmr)
    return val

from functools import partial
run_timeit_dischard_first = partial(run_timeit,
        logic=lambda v: min(v[1:]) if len(v) > 1 else v[0])

def timing_regular(N, rpt_args):
    setup_str = '''
from __main__ import setup_solver_regular, dasslcy
args = setup_solver_regular({})
    '''.format(N)
    # tt = timeit.timeit('dasslcy.solve(*args)', setup=setup_str, number=1)
    tt = run_timeit_dischard_first('dasslcy.solve(*args)', setup=setup_str,
                    rpt_args=rpt_args)
    print('Regular solver: {} s'.format(tt))

def timing_shared(N, rpt_args):
    setup_str = '''
from __main__ import setup_solver_shared, dasslcy;
args = setup_solver_shared({})
    '''.format(N)
    tt = run_timeit_dischard_first('dasslcy.solve(*args, share_res=1)',
                    setup=setup_str, rpt_args=rpt_args)
    print('Shared mem solver: {} s'.format(tt))

def timing_shared_np_broadcast(N, rpt_args):
    setup_str = '''
from __main__ import setup_solver_shared_np_broadcast, dasslcy;
args = setup_solver_shared_np_broadcast({})
    '''.format(N)
    tt = run_timeit_dischard_first('dasslcy.solve(*args, share_res=1)',
                    setup=setup_str, rpt_args=rpt_args)
    print('Shared mem NP BROADCAST: {} s'.format(tt))

def timing_shared_cython(N, rpt_args):
    setup_str = '''
from __main__ import setup_solver_shared_cython, dasslcy;
args = setup_solver_shared_cython({})
    '''.format(N)
    tt = run_timeit_dischard_first('dasslcy.solve(*args, share_res=1)',
                    setup=setup_str, rpt_args=rpt_args)
    print('Shared mem Cython Naive: {} s'.format(tt))

def timing_typed_cython(N, rpt_args):
    setup_str = '''
from __main__ import setup_solver_typed_cython, dasslcy;
args = setup_solver_typed_cython({})
    '''.format(N)
    tt = run_timeit_dischard_first('dasslcy.solve(*args, share_res=1)',
                    setup=setup_str, rpt_args=rpt_args)
    print('TYPED CYTHON: {} s'.format(tt))

def timing_numba(N, rpt_args):
    setup_str = '''
from __main__ import setup_solver_numba, dasslcy;
args = setup_solver_numba({})
    '''.format(N)
    tt = run_timeit_dischard_first('dasslcy.solve(*args, share_res=1)',
                    setup=setup_str, rpt_args=rpt_args)
    print('DASSLCY + NUMBA: {} s'.format(tt))

def new_numba(N):
    import numba
    spec = [
        ('N', numba.int32),
        ('D', numba.float64),
        ('vz', numba.float64),
        ('k', numba.float64),
        ('Cf', numba.float64),
        ('z0', numba.float64),
        ('zf', numba.float64),
        ('h', numba.float64),
    ]
    Numba_PFR = numba.jitclass(spec)(pfr)
    jitted_pfr_model = numba.jit(model_pfr_shared_direct, nopython=True)
    numba_pfr = Numba_PFR(N)
    numba_args = setup_base(N)
    numba_args[3] = numba_pfr
    dasslcy.solve(jitted_pfr_model, *numba_args, share_res=1)
    return [jitted_pfr_model] + numba_args

def timing_new_numa(N, rpt_args):
    setup_str = '''
from __main__ import new_numba, dasslcy;
args = new_numba({})
    '''.format(N)
    tt = run_timeit_dischard_first('dasslcy.solve(*args, share_res=1)',
                                   setup=setup_str, rpt_args=rpt_args)
    print('NEW NUMBA: {} s'.format(tt))

def timing():
    N = 100
    args_tt = (N, (2, 1))
    timing_regular(*args_tt)
    timing_shared(*args_tt)
    timing_shared_np_broadcast(*args_tt)
    timing_shared_cython(*args_tt)
    timing_typed_cython(*args_tt)
    timing_numba(*args_tt)
    timing_new_numa(*args_tt)

#--------------------------------------------
# 	 SIMPLE TEST JUST TO MAKE SURE THEY ARE MATCHING
#--------------------------------------------
@pytest.fixture()
def Nfix(): return 20

def test_py(Nfix):
    args = setup_solver_regular(Nfix)
    t, y, yp = dasslcy.solve(*args)
    assert(np.isclose(sum(y[-1]), 81.9599366, 1e-3))


def test_py_shared(Nfix):
    args = setup_solver_shared(Nfix)
    t, y, yp = dasslcy.solve(*args, share_res=1)
    assert(np.isclose(sum(y[-1]), 81.9599366, 1e-3))


def test_py_shared_broadcast(Nfix):
    args = setup_solver_shared_np_broadcast(Nfix)
    t, y, yp = dasslcy.solve(*args, share_res=1)
    assert(np.isclose(sum(y[-1]), 81.9599366, 1e-3))


def test_py_cython(Nfix):
    args = setup_solver_shared_cython(Nfix)
    t, y, yp = dasslcy.solve(*args, share_res=1)
    assert(np.isclose(sum(y[-1]), 81.9599366, 1e-3))

def test_py_cython_typed(Nfix):
    args = setup_solver_typed_cython(Nfix)
    t, y, yp = dasslcy.solve(*args, share_res=1)
    assert(np.isclose(sum(y[-1]), 81.9599366, 1e-3))

def test_numba(Nfix):
    args = setup_solver_numba(Nfix)
    t, y, yp = dasslcy.solve(*args, share_res=1)
    assert(np.isclose(sum(y[-1]), 81.9599366, 1e-3))


#--------------------------------------------
# 	 MAIN
#--------------------------------------------

if __name__ == '__main__':
    timing()
    #run_solver(20)
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
