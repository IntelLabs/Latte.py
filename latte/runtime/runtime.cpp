#include "$LATTE_PACKAGE_PATH/runtime/runtime.h"

extern "C"
void init_default() {
  scheduler_init = new tbb::task_scheduler_init(tbb::task_scheduler_init::automatic);
  ap = new tbb::affinity_partitioner();
}

extern "C"
void init_nthreads(int nthread) {
  scheduler_init = new tbb::task_scheduler_init(nthread);
  ap = new tbb::affinity_partitioner();
}

extern "C"
void cleanup() {
    delete ap;
    delete scheduler_init;
}
