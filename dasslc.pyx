# distutils: sources = dasslc/dasslc.c
# distutils: include_dirs = dasslc/

import numpy as np
cimport numpy as np
cimport dasslc_def

np.import_array()

cdef object pyres

cpdef hello(double tf):
    print('hello {}'.format(tf))

cdef int eval_residual_function(void* context, double tf):
    try:
        # recover Python function object from void* argument
        func = <object>context
        # call function, convert result into 0/1 for True/False
        return func(tf)
    except:
        # catch any Python errors and return error indicator
        return -1

cdef dasslc_def.BOOL residuals(dasslc_def.PTR_ROOT *root, 
                                dasslc_def.REAL t, dasslc_def.REAL *y, 
                                dasslc_def.REAL *yp, dasslc_def.REAL *res, 
                                dasslc_def.BOOL *jac):
    #cdef int[:] y_view = <double*>y
    cdef:
        np.npy_intp shape[1]
        int size
        cdef int carr[3]
        dasslc_def.REAL *y_
        np.ndarray ndarray
        np.float64_t[:] res_view
        np.ndarray y_np
        np.ndarray yp_np

    #print('size = {}'.format(size))
    #print(y[0])
    size = root.rank
    shape[0] = <np.npy_intp>size
#
    y_np = np.PyArray_SimpleNewFromData(1, shape, np.NPY_DOUBLE, y)
    yp_np = np.PyArray_SimpleNewFromData(1, shape, np.NPY_DOUBLE, yp)
#
    return_pyres = pyres(t, y_np, yp_np)
    cdef np.float64_t[:] py_calc_res = return_pyres[0]
    #cdef dasslc_def.REAL *res_py = &(return_pyres[0][0])
    #res = &py_calc_res[0]
    #print('y[0]={}'.format(y[0]))
    #print('y_np[0]={}'.format(y_np[0]))
    #print('res[0]={}'.format(res[0]))
    #cdef int ires = return_pyres[1]
    res[0] = yp[0] + 2.0*y[0]
    print('res={} and py_calc_res[0]={}'.format(res[0], py_calc_res[0]))

    #PAREI AQUI-> O ponteiro do py_calc_res se perde, pois res é criado no py e depois morre
    # Ver como é feito nas funcões do dasslc em puro c para evitar a multipla definicao do res
    ires = 0
    return ires

def call_res(residual_function, tf):
    return eval_residual_function(<void*> residual_function, tf)

cpdef solve(resfun, np.float64_t[:] tspan, np.float64_t[:] y0, 
                    np.float64_t[:] yp0 = None, rpar=None, rtol=1e-6, 
                    atol=1e-8):
    print(tspan[0])
    print(tspan[1])
    cdef dasslc_def.PTR_ROOT root
    global pyres
    pyres = resfun
    cdef int neq = y0.size
    print('neq={}'.format(neq))
    cdef dasslc_def.REAL t0 = tspan[0]
    cdef int index[1]
    index[0] = 0
    cdef dasslc_def.BOOL err = dasslc_def.daSetup("?",&root, residuals, neq, 
                                t0, &y0[0], NULL, NULL, NULL, NULL);
    if err > 0:
        print('Setup error here')
    return