import os

os.environ["KMP_AFFINITY"] = "compact,granularity=fine,0,0"

vec_config = os.getenv("LATTE_VEC_CONFIG", "AVX-2")
print("Latte running with vector instruction set {}".format(vec_config))
try:
    SIMDWIDTH = {
        "AVX": 8,
        "AVX-2": 8,
        "AVX-512": 16
    }[vec_config]
except KeyError:
    raise Exception("ERROR: Invalid LATTE_VEC_CONFIG value = {}.  Supported values are {} ".format(vec_config, vec_configs.keys()))
