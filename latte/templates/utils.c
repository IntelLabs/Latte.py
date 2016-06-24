#include <unistd.h>

extern "C"
double get_cpu_freq() {
	unsigned long long t0, t1;
	t0 = __rdtsc();
	sleep(1);
	t1 = __rdtsc();
	return (double)(t1 - t0);
}
