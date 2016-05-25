extern "C"
void * aligned_malloc(int size) {
    return (void *) _mm_malloc(size, 64);
}
