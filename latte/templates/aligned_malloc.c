extern "C"
void * aligned_malloc(long size) {
    return (void *) _mm_malloc(size, 64);
}
