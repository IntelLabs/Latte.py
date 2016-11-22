#include <immintrin.h>
#include <mkl.h>
#include <stdio.h>
#include <cmath>
#include <omp.h>
#include <unistd.h>
#if $INCLUDE_RUNTIME
#include "$LATTE_PACKAGE_PATH/runtime/runtime.h"
#endif
#define SIMDWIDTH $SIMDWIDTH
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define MAX(x, y) (((x) > (y)) ? (x) : (y))

#if $INCLUDE_OPENCL
#ifdef APPLE
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#endif

#if $INCLUDE_LIBXSMM
#include <libxsmm.h>
#include <libxsmm_dnn.h>
#endif

$TRANSPOSE

extern "C"
