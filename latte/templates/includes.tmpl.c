#include <immintrin.h>
#include <mkl.h>
#include <stdio.h>
#include <cmath>
#include <omp.h>
#include <unistd.h>
#include <tbb/tbb.h>
#include "$LATTE_PACKAGE_PATH/runtime/runtime.h"
#define SIMDWIDTH 8
#define TILE_SIZE 8
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define MAX(x, y) (((x) > (y)) ? (x) : (y))

$TRANSPOSE

extern "C"
