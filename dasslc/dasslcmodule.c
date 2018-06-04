#include <Python.h>
#include <numpy/arrayobject.h>
#include "dasslc.h"

#define FREEALL() Py_XDECREF(t_array); Py_XDECREF(y_array); Py_XDECREF(yp_array);\
                  Py_XDECREF(idx_array);\
                  if (t) free(t);\
                  if (y) free(y);\
                  if (yp) free(yp);\
                  if (index) free(index);\
                  daFree(&root);


/* The function's prototypes */
static PyObject* dasslc_solve(PyObject *self, PyObject *args);


/* The method's table */
static PyMethodDef dasslcMethods[] = {
    {"solve", dasslc_solve, METH_VARARGS , "Solve the problem."},
    {NULL, NULL, 0, NULL}
};


/* The  global variables */
static DASSLC_RES residuals;         //The C residual function
static PyObject *pyres = NULL;       //The Python residual function


/* The module initialization function */
PyMODINIT_FUNC initdasslc(void){
    PyObject *m = Py_InitModule("dasslc", dasslcMethods);
    if (m == NULL)
        return;

    /* Load numpy functionality. */
    import_array();
}


/* The function declaration */
static PyObject* dasslc_solve(PyObject *self, PyObject *args){
    //python call: dasslc.solve(resfun, tspan, y0, yp0, rpar, rtol, atol, index)

    /* The memory allocation */
    int ntp, ntp2, neq, ndr = -1, idxnum = -1, *index = NULL;
    double t0, tf, dt, *t = NULL, *y = NULL, *yp = NULL;
    PyArrayObject *t_sol, *y_sol, *yp_sol, *t_array = NULL, *y_array = NULL, *yp_array = NULL, *idx_array = NULL;
    PyObject *result, *arglist;
    PTR_ROOT root;

    /* The python inputs and outputs */
    double atol = 1e-8, rtol = 1e-6;
    PyObject *t_obj, *y_obj, *resfun_obj, *yp_obj = NULL, *rpar_obj = NULL, *idx_obj = NULL;

    /* Parse inputs */
    if (!PyArg_ParseTuple(args, "OOO|OOddO", &resfun_obj, &t_obj, &y_obj, &yp_obj, &rpar_obj, &rtol, &atol, &idx_obj))
        return NULL;

    /* Check if residual function is callable (1)*/
    if (!PyCallable_Check(resfun_obj)){
        FREEALL();
        PyErr_SetString(PyExc_TypeError, "Cannot call provided residual function.");
        return NULL;
    }
    Py_XINCREF(resfun_obj);      //Add a reference to new callback 
    Py_XDECREF(pyres);           //Dispose of previous callback 
    pyres = resfun_obj;          //Remember new callback 
    
    /* Interpret the input objects as numpy arrays. */
    t_array = (PyArrayObject*)PyArray_FROM_OTF(t_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    y_array = (PyArrayObject*)PyArray_FROM_OTF(y_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    if (yp_obj && yp_obj != Py_None)
        yp_array = (PyArrayObject*)PyArray_FROM_OTF(yp_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    if (idx_obj && idx_obj != Py_None)
        idx_array = (PyArrayObject*)PyArray_FROM_OTF(idx_obj, NPY_DOUBLE, NPY_IN_ARRAY);

    /* If that didn't work, throw an exception. */
    if (t_array == NULL || y_array == NULL){
        FREEALL();
        PyErr_SetString(PyExc_TypeError, "t and y must be numpy 1D-arrays");
        return NULL;
    }

    /* Get dimensions */
    ntp = PyArray_SHAPE(t_array) ? (int)PyArray_DIM(t_array, 0) : 1;
    neq = PyArray_SHAPE(y_array) ? (int)PyArray_DIM(y_array, 0) : 1;

    /* Check if residual function is callable (2) */
    if (rpar_obj && rpar_obj != Py_None)
        arglist = Py_BuildValue("dOOO",3.14,y_obj,y_obj,rpar_obj);
    else
        arglist = Py_BuildValue("dOO",3.14,y_obj,y_obj);

    result = PyObject_CallObject(pyres, arglist);
    PyObject *dummyO;
    int dummyi;
    if (!result || !PyArg_ParseTuple(result, "Oi",&dummyO,&dummyi)){
        FREEALL();
        PyErr_SetString(PyExc_TypeError, "There's some problem in residual function. Check I/O!");
        return NULL;
    }
    PyArrayObject *dummyVec = (PyArrayObject*)PyArray_FROM_OTF(dummyO, NPY_DOUBLE, NPY_IN_ARRAY);
    int dummyVar = PyArray_SHAPE(dummyVec) ? (int)PyArray_DIM(dummyVec, 0) : 1;
    if (dummyVar != neq){
        FREEALL();
        Py_XDECREF(result);
        Py_XDECREF(arglist);
        PyErr_SetString(PyExc_TypeError, "Residual function must return a vector with the same length of y0!");
        return NULL;
    }
    Py_XDECREF(result);
    Py_XDECREF(arglist);

    /* Get pointers to the data as C-types. */
    t = (double*) malloc(ntp*sizeof(double));
    y = (double*) malloc(neq*sizeof(double));
    for (int i = 0; i < ntp; i++)
        t[i] = *(double*)PyArray_GETPTR1(t_array,i);
    for (int j = 0; j < neq; j++)
        y[j] = *(double*)PyArray_GETPTR1(y_array,j);

    if (yp_array){
        ndr = PyArray_SHAPE(yp_array) ? (int)PyArray_DIM(yp_array, 0) : 1;
        if (ndr >= 0 && ndr != neq){ //Throw an exception
            FREEALL();
            PyErr_SetString(PyExc_TypeError, "yp0 must have the same length of y0!");
            return NULL;
        }
        yp = (double*) malloc(neq*sizeof(double));
        for (int j = 0; j < neq; j++)
            yp[j] = *(double*)PyArray_GETPTR1(yp_array,j);
    }
    if (idx_array){
        idxnum = PyArray_SHAPE(idx_array) ? (int)PyArray_DIM(idx_array, 0) : 1;
        if (idxnum >= 0 && idxnum != neq){//Throw an exception
            FREEALL();
            PyErr_SetString(PyExc_TypeError, "Index vector must have the same length of y0!");
            return NULL;
        }
        index = (int*) malloc(neq*sizeof(int));
        for (int j = 0; j < neq; j++)
            index[j] = (int) *(double*)PyArray_GETPTR1(idx_array,j);
            //Workaround: Problem creating an int array directly in python
            //            So, create a double array then convert it to int here
    }

    /* Set the rpar if any */
    if (!rpar_obj || rpar_obj == Py_None)
        root.user = NULL;
    else
        root.user = (void*)rpar_obj;

    /* Create the solution vector */
    ntp2 = ntp > 2 ? ntp : 100;
    npy_intp dims1[1] = {ntp2};
    npy_intp dims2[2] = {ntp2,neq};
    t_sol = (PyArrayObject*)PyArray_SimpleNew(1,dims1,NPY_DOUBLE);
    y_sol = (PyArrayObject*)PyArray_SimpleNew(2,dims2,NPY_DOUBLE);
    yp_sol = (PyArrayObject*)PyArray_SimpleNew(2,dims2,NPY_DOUBLE);

    /* Call the daSetup function */
    t0 = ntp == 1 ? 0 : t[0];
    BOOL err = daSetup("?",&root,residuals,neq,t0,y,yp,(int*)index,NULL,NULL);
    if (err){
        FREEALL();
        char buff[128] = "Setup error: ";
        sprintf(buff,"%d",err);
        PyErr_SetString(PyExc_TypeError, buff);
        return NULL;
    }

    /* Configure root structure */
    root.iter.stol = 1;
    root.iter.atol[0] = atol;
    root.iter.rtol[0] = rtol;

    /* Find initial derivatives if not given */
    if (ntp == 1){
        dt = (double) t[0]/(ntp2-1);
        tf = t0 + dt;
    }else if (ntp == 2){
        dt = (double) (t[1]-t[0])/(ntp2-1);
        tf = t0 + dt;
    }else{
        tf = t[1];
    }

    if (yp == NULL){
        err = dasslc(INITIAL_COND, &root, residuals, &t0, tf, NULL, NULL);
        if (err < 0){
            FREEALL();
            char buff[128] = "Failed in finding consistent initial condition. Error: ";
            sprintf(buff,"%d",err);
            PyErr_SetString(PyExc_TypeError, buff);
            return NULL;
        }
    }

    /* Update soluton vector */
    *(double*)PyArray_GETPTR1(t_sol,0) = root.t;
    for (int j = 0; j < neq; j++){
        *(double*)PyArray_GETPTR2(y_sol,0,j) = root.y[j];
        *(double*)PyArray_GETPTR2(yp_sol,0,j) = root.yp[j];
    }

    /* Call the dasslc function for all tspan */
    for (int i = 1; i < ntp2; i++){
        tf = ntp > 2 ? t[i] : t0 + dt;
        err = dasslc(TRANSIENT, &root, residuals, &t0, tf, NULL, NULL);
        if (err < 0){
            FREEALL();
            char buff[128] = "Error during integration: ";
            sprintf(buff,"%d",err);
            PyErr_SetString(PyExc_TypeError, buff);
            return NULL;
        }
        *(double*)PyArray_GETPTR1(t_sol,i) = root.t;
        for (int j = 0; j < neq; j++){
            *(double*)PyArray_GETPTR2(y_sol,i,j) = root.y[j];
            *(double*)PyArray_GETPTR2(yp_sol,i,j) = root.yp[j];
        }
    }

    /* Clean Up */
    Py_XDECREF(t_array);
    Py_XDECREF(y_array);
    Py_XDECREF(yp_array);
    Py_XDECREF(idx_array);
    free(t);
    free(y);
    if (yp) free(yp);
    if (index) free(index);
    daFree(&root);

    /* Build the output tuple */
    return Py_BuildValue("OOO", t_sol, y_sol, yp_sol);
}

static BOOL residuals(PTR_ROOT *root, REAL t, REAL *y, REAL *yp, REAL *res, BOOL *jac){
    //Interface with the python residual function

    /* Memory allocation */
    PyObject *arglist, *result, *res_obj;
    PyArrayObject *res_array, *y_array, *yp_array;
    int ires = -1;

    /* Build the arglist (convert c-array to PyArray) */
    int neq = root -> rank;
    npy_intp dims[1] = {neq};
    y_array = (PyArrayObject*)PyArray_SimpleNew(1,dims,NPY_DOUBLE);
    yp_array = (PyArrayObject*)PyArray_SimpleNew(1,dims,NPY_DOUBLE);

    for (int i = 0; i < neq; i++){
        *(double*)PyArray_GETPTR1(y_array,i) = y[i];
        *(double*)PyArray_GETPTR1(yp_array,i) = yp[i];
    }

    /* Parse arglist checking if rpar exists */
    if (root->user)
        arglist = Py_BuildValue("dOOO",t,y_array,yp_array,(PyObject*)root->user);
    else
        arglist = Py_BuildValue("dOO",t,y_array,yp_array);

    /* Call the python function */
    result = PyObject_CallObject(pyres, arglist);
    Py_XDECREF(arglist);

    /* Parse the result tuple */
    PyArg_ParseTuple(result, "Oi", &res_obj, &ires);
    res_array = (PyArrayObject*)PyArray_FROM_OTF(res_obj, NPY_DOUBLE, NPY_IN_ARRAY);

    if (ires){
        Py_XDECREF(y_array);
        Py_XDECREF(yp_array);
        Py_XDECREF(result);
        return ires;
    }

    /* Convert result to c-array res */
    for (int i = 0; i < neq; i++)
        res[i] = *(double*)PyArray_GETPTR1(res_array,i);

    /* Clean up */
    Py_XDECREF(y_array);
    Py_XDECREF(yp_array);
    Py_XDECREF(result);

    return ires;
}
