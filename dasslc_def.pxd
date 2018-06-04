
cdef extern from "dasslc/dasslc.h":
    ctypedef int BOOL
    ctypedef signed char SET
    ctypedef double REAL
    
    enum:
        NONE = 0
        SETUP = 1
        STEADY_STATE = 2
        INITIAL_COND = 3
        TRANSIENT = 4

    # The structures
    ctypedef struct ITER_SET:
        BOOL stol
        REAL *rtol
        REAL *atol    
    ctypedef struct PTR_ROOT:
        int rank
        ITER_SET iter
        REAL *y
        REAL *yp
        REAL t
        void *user

    ctypedef struct DATABASE:
        pass

    # The function types
    ctypedef BOOL (*DASSLC_RES)(PTR_ROOT *, REAL, REAL *, REAL *, REAL *, BOOL *)
    ctypedef BOOL (*DASSLC_JAC)(PTR_ROOT *, REAL, REAL *, REAL *, REAL, void *, DASSLC_RES *)
    ctypedef BOOL (*DASSLC_PSOL)(PTR_ROOT *, REAL *, DASSLC_RES *)
    
    # The functions declarations
    BOOL dasslc (SET, PTR_ROOT *, DASSLC_RES, REAL *, REAL, DASSLC_JAC *, DASSLC_PSOL *);
    BOOL daSetup (char *, PTR_ROOT *, DASSLC_RES, int, REAL, REAL *, REAL *,
					int *, DATABASE *, DATABASE **)
    void daFree (PTR_ROOT *)
