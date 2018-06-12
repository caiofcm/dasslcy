import numpy as np
cimport numpy as np

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
    for i in range(1, N - 1):
        tt1 = aux2 * (Ci[i + 1] - 2.0 * Ci[i] + Ci[i - 1])
        tt2 = -aux3 * (Ci[i + 1] - Ci[i - 1]) + k * Ci[i]
        res[i] = tt1 + tt2 - dCi[i]
    res[N - 1] = aux2 * (CNp1 - 2.0 * Ci[N - 1] + Ci[N - 2]) - \
        aux3 * (CNp1 - Ci[N - 2]) + k * Ci[N - 1] - dCi[N - 1]
    pass

cdef class Pfr_Cython:
    cdef:
        public double D, vz, k, Cf, z0, zf, h
        public int N

    def __init__(self, N = 20):
        self.N = N
        self.D = 1.0
        self.vz = 1.0
        self.k = 1.0
        self.Cf = 1.0
        self.z0 = 0.0
        self.zf = 1.0
        self.h = self.get_h()
        # self.initialize_intermediaries()

    cdef get_h(self):
        return (self.zf - self.z0) / self.N

#    cdef initialize_intermediaries(self):
#        self.r = np.empty(self.N)

    cdef cythonized_base_model_calculations(self, np.float64_t[:] res, np.float64_t[:] y, np.float64_t[:] yp,
        int N, double D, double vz, double k, double Cf, double h):
        cdef:
            int i
        cdef np.float64_t[:] dCi = yp
        cdef np.float64_t[:] Ci = y
        cdef double aux1 = D / (vz * h)
        cdef double C0 = 1.0 / (1.0 + aux1) * (aux1 * Ci[0] + Cf)
        cdef double CNp1 = Ci[N - 1]
        cdef double aux2 = D / h**2
        cdef double aux3 = vz / (2 * h)
        res[0] = aux2 * (Ci[1] - 2.0 * Ci[0] + C0) - \
            aux3 * (Ci[1] - C0) + k * Ci[0] - dCi[0]
        for i in range(1, N - 1):
            tt1 = aux2 * (Ci[i + 1] - 2.0 * Ci[i] + Ci[i - 1])
            tt2 = -aux3 * (Ci[i + 1] - Ci[i - 1]) + k * Ci[i]
            res[i] = tt1 + tt2 - dCi[i]
        res[N - 1] = aux2 * (CNp1 - 2.0 * Ci[N - 1] + Ci[N - 2]) - \
            aux3 * (CNp1 - Ci[N - 2]) + k * Ci[N - 1] - dCi[N - 1]
        pass

cdef Pfr_Cython cyPfr
def initialize_cy_pfr_model(N):
    global cyPfr
    cyPfr = Pfr_Cython(N)
    pass

def cython_model_pfr_shared(double t, np.float64_t[:] y, np.float64_t[:] yp, np.float64_t[:] res):
    cyPfr.cythonized_base_model_calculations(
            res, y, yp, cyPfr.N, cyPfr.D, cyPfr.vz, cyPfr.k, cyPfr.Cf, cyPfr.h)
    return res, 0
