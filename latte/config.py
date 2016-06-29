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
    "OPENCL_SIMPLE_LOOP" # converts parallel loops to NDRange kernel
]

parallel_strategy = os.getenv("LATTE_PARALLEL_STRATEGY", "SIMPLE_LOOP")

if parallel_strategy not in parallel_strategies:
    logger.warn("Invalid parallel strategy [%s], defaulting to SIMPLE_LOOP", parallel_strategy)
    parallel_strategy = "SIMPLE_LOOP"

nthreads = os.getenv("LATTE_NUM_THREADS", None)

if parallel_strategy == "OPENCL_SIMPLE_LOOP":
    import pycl as cl
    cl_ctx = cl.clCreateContext()
    cl_queue = cl.clCreateCommandQueue(cl_ctx)
elif parallel_strategy in ["SIMPLE_LOOP"]:
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
elif parallel_strategy == "OPENMP":
    if nthreads is not None:
        os.environ["OMP_NUM_THREADS"] = nthreads

logger.info("========== Configuration ==========")
logger.info("    march             = %s", vec_config)
logger.info("    parallel_strategy = %s", parallel_strategy)
logger.info("    nthreads          = %s", nthreads or "unspecified")
logger.info("===================================")
