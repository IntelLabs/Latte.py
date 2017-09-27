'''
Copyright (c) 2015, Intel Corporation

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''
import os
import logging
import ctypes
from ctree.templates.nodes import FileTemplate, StringTemplate
import ctree.c.nodes as C
import latte.util as util
import ctree

logger = logging.getLogger("latte")

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("ctree").setLevel(logging.WARN)
#os.environ["KMP_AFFINITY"] = "compact,granularity=fine,0,0"

vec_config = os.getenv("LATTE_VEC_CONFIG", "AVX-2")
try:
    SIMDWIDTH = {
        "AVX": 8,
        "AVX-2": 8,
        "AVX-512": 16
    }[vec_config]
except KeyError:
    raise Exception("ERROR: Invalid LATTE_VEC_CONFIG value = {}.  Supported values are {} ".format(vec_config, vec_configs.keys()))

parallel_strategies = [
    "SIMPLE_LOOP",       # nested parallel_for (like basic TBB)
    "FLOWGRAPH_LOOP",    # FlowGraph model for first level
                         # parallelism, parallel_for for nested
    "OPENMP",            # pragma omp parallel for (supports collapse(2))
    "OPENCL_SIMPLE_LOOP",# converts parallel loops to NDRange kernel
    "LIBXSMMOPENMP"     # LIBXSMM with OpenMP
]

parallel_strategy = os.getenv("LATTE_PARALLEL_STRATEGY", "SIMPLE_LOOP")

MODES = [
    "RELEASE",
    "DEV"
]
MODE = os.getenv("LATTE_MODE", "RELEASE")

TIMERS = [
    "ON",
    "OFF"
]
TIMER = os.getenv("LATTE_TIMER", "OFF")
 
AUTO_FUSION_OPTION = [
    "ON",
    "OFF"
]
AUTO_FUSION = os.getenv("LATTE_AUTO_FUSION", "OFF")

SIMPLE_FUSION_OPTION = [
    "ON",
    "OFF"
]
SIMPLE_FUSION = os.getenv("LATTE_SIMPLE_FUSION", "OFF") 



prefetch_options = [
    "ON",
    "OFF"
]
 
prefetch_options = [
    "ON",
    "OFF"
]
prefetch_option = os.getenv("LATTE_PREFETCH_MODE", "ON")

unroll_options = [
    "ON",
    "OFF"
]
 

unroll_option = os.getenv("LATTE_UNROLL", "ON")



if parallel_strategy not in parallel_strategies:
    logger.warn("Invalid parallel strategy [%s], defaulting to OPENMP", parallel_strategy)
    parallel_strategy = "OPENMP"

nthreads = os.getenv("LATTE_NUM_THREADS", None)
if parallel_strategy == "OPENCL_SIMPLE_LOOP":
    import pycl as cl
    cl_ctx = cl.clCreateContext()
    cl_queue = cl.clCreateCommandQueue(cl_ctx)
elif parallel_strategy in ["SIMPLE_LOOP"] or parallel_strategy in ["FLOWGRAPH_LOOP"]:
    package_path = os.path.dirname(os.path.abspath(__file__))
    _file = FileTemplate(os.path.dirname(os.path.abspath(__file__)) +
            "/runtime/runtime.cpp",
            {"LATTE_PACKAGE_PATH": StringTemplate(package_path)})

    c_file = C.CFile("runtime", [_file])
    module = util.mpi_compile(ctree.nodes.Project([c_file]))
    init_nthreads = module.get_callable("init_nthreads", ctypes.CFUNCTYPE(None, ctypes.c_int))
    init_default = module.get_callable("init_default", ctypes.CFUNCTYPE(None))
    if nthreads is not None:
        init_nthreads(int(nthreads))
    else:
        init_default()
    if parallel_strategy in ["FLOWGRAPH_LOOP"]:
      img_block_size = os.getenv("LATTE_PIPELINE_BLOCK_SIZE", 16)
      img_block_size = int(img_block_size)
elif parallel_strategy == "OPENMP":
    if nthreads is not None:
        os.environ["OMP_NUM_THREADS"] = nthreads

codegen_strategies = [
    "GEMM",       # MKL GEMM formulation
    "AUTOVEC",    # Automatic vectorization
    "LIBXSMM"    # use libxsmm
]

codegen_strategy = os.getenv("LATTE_CODEGEN_STRATEGY", "AUTOVEC")

if codegen_strategy not in codegen_strategies:
    logger.warn("Invalid codegen strategy [%s], defaulting to GEMM", codegen_strategy)
    codegen_strategy = "AUTOVEC"

logger.info("========== Configuration ==========")
logger.info("    march             = %s", vec_config)
logger.info("    parallel_strategy = %s", parallel_strategy)
logger.info("    nthreads          = %s", nthreads or "unspecified")
logger.info("    codegen_strategy = %s", codegen_strategy)
logger.info("    prefetch_option = %s", prefetch_option)
logger.info("    unroll_option = %s", unroll_option)
logger.info("    AUTO_FUSION = %s", AUTO_FUSION)
logger.info("    SIMPLE_FUSION = %s", SIMPLE_FUSION)
logger.info("    mode = %s", MODE)
if "ON" in TIMER:
  logger.info("    timer = %s", "ON")
if parallel_strategy in ["FLOWGRAPH_LOOP"]:
  logger.info("    pipeline_block = %s", img_block_size)
logger.info("===================================")
