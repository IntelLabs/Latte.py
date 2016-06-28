import os
import logging

logger = logging.getLogger("latte")

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("ctree").setLevel(logging.WARN)
os.environ["KMP_AFFINITY"] = "compact,granularity=fine,0,0"

vec_config = os.getenv("LATTE_VEC_CONFIG", "AVX-2")
try:
    SIMDWIDTH = {
        "AVX": 8,
        "AVX-2": 8,
        "AVX-512": 16
    }[vec_config]
except KeyError:
    raise Exception("ERROR: Invalid LATTE_VEC_CONFIG value = {}.  Supported values are {} ".format(vec_config, vec_configs.keys()))

# Parallel Strategies Overview
# ============================
# SIMPLE_LOOP - nested parallel_for (like basic TBB)
# FLOWGRAPH_LOOP - FlowGraph model for first level parallelism,
#     parallel_for for depth 1 nested parallelism
# OPENMP - pragma omp parallel for (currently supports collapse for depth 2)
parallel_strategy = os.getenv("LATTE_PARALLEL_STRATEGY", "SIMPLE_LOOP")
if parallel_strategy not in ["SIMPLE_LOOP", "FLOWGRAPH_LOOP", "OPENMP"]:
    logger.warn("Invalid parallel strategy [%s], defaulting to SIMPLE_LOOP", parallel_strategy)
    parallel_strategy = "SIMPLE_LOOP"
logger.info("========== Configuration ==========")
logger.info("    march             = %s", vec_config)
logger.info("    parallel_strategy = %s", parallel_strategy)
logger.info("===================================")
