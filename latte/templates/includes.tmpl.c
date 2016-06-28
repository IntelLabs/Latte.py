#include <immintrin.h>
#include <mkl.h>
#include <stdio.h>
#include <cmath>
#include <omp.h>
#include <unistd.h>
#include <tbb/tbb.h>
#include "$LATTE_PACKAGE_PATH/runtime/runtime.h"
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

$TRANSPOSE

extern "C"
