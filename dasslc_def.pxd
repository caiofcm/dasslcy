
cdef extern from "dasslc/dasslc.h":
    ctypedef int BOOL
    ctypedef signed char SET
    ctypedef double REAL
    ctypedef struct PTR_ROOT:
        int rank
    ctypedef struct DATABASE:
        pass
    ctypedef BOOL (*DASSLC_RES)(PTR_ROOT *, REAL, REAL *, REAL *, REAL *, BOOL *)
    ctypedef BOOL (*DASSLC_JAC)(PTR_ROOT *, REAL, REAL *, REAL *, REAL, void *, DASSLC_RES *)
    ctypedef BOOL (*DASSLC_PSOL)(PTR_ROOT *, REAL *, DASSLC_RES *)
    
    BOOL dasslc (SET, PTR_ROOT *, DASSLC_RES *, REAL *, REAL, DASSLC_JAC *, DASSLC_PSOL *);
    BOOL daSetup (char *, PTR_ROOT *, DASSLC_RES, int, REAL, REAL *, REAL *,
					int *, DATABASE *, DATABASE **)
