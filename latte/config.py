import os
import logging

logger = logging.getLogger("latte")

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("ctree").setLevel(logging.WARN)
# os.environ["KMP_AFFINITY"] = "compact,granularity=fine,0,0"

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

if parallel_strategy == "OPENCL_SIMPLE_LOOP":
    import pycl as cl
    cl_ctx = cl.clCreateContext()
    cl_queue = cl.clCreateCommandQueue(cl_ctx)

logger.info("========== Configuration ==========")
logger.info("    march             = %s", vec_config)
logger.info("    parallel_strategy = %s", parallel_strategy)
logger.info("===================================")
