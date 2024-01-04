#include "xtn.h"
#include <sys/sysinfo.h>
#include <time.h>

int print_err(const char* str) {
	fprintf(stderr, "Error: %s\n", str);
	return ERROR;
}

float startTime = 0;

int clock_start() {
	startTime = (float)clock() / CLOCKS_PER_SEC;
}

void print_args(XTNArgs args) {
	printf("XTNArgs{\n");
	printf("\tdistance: %d\n", args.distance);
	printf("\tverbose: %d\n", args.verbose);
	printf("\tseq1Len: %'d\n", args.seq1Len);
	printf("\tseq1Path: \"%s\"\n", args.seq1Path);
	printf("\toutputPath: \"%s\"\n", args.outputPath);
	printf("}\n");
}

void _cudaFree(void* a) {
	cudaFree(a);
}
void _cudaFree(void* a, void* b) {
	cudaFree(a);
	cudaFree(b);
}
void _cudaFree(void* a, void* b, void* c) {
	cudaFree(a);
	cudaFree(b);
	cudaFree(c);
}
void _cudaFree(void* a, void* b, void* c, void* d) {
	cudaFree(a);
	cudaFree(b);
	cudaFree(c);
	cudaFree(d);
}
void _cudaFree(void* a, void* b, void* c, void* d, void* e) {
	cudaFree(a);
	cudaFree(b);
	cudaFree(c);
	cudaFree(d);
	cudaFree(e);
}
void _cudaFree(void* a, void* b, void* c, void* d, void* e, void* f) {
	cudaFree(a);
	cudaFree(b);
	cudaFree(c);
	cudaFree(d);
	cudaFree(e);
	cudaFree(f);
}
void _cudaFree(void* a, void* b, void* c, void* d, void* e, void* f, void* g) {
	cudaFree(a);
	cudaFree(b);
	cudaFree(c);
	cudaFree(d);
	cudaFree(e);
	cudaFree(g);
}

void _cudaFreeHost(void* a, void* b) {
	cudaFreeHost(a);
	cudaFreeHost(b);
}

void _cudaFreeHost(void* a, void* b, void* c) {
	cudaFreeHost(a);
	cudaFreeHost(b);
	cudaFreeHost(c);
}

void _cudaFreeHost(void* a, void* b, void* c, void* d) {
	cudaFreeHost(a);
	cudaFreeHost(b);
	cudaFreeHost(c);
	cudaFreeHost(d);
}

template <typename T>
void _cudaFreeHost2D(T** a, int n) {
	for (int i = 0; i < n; i++)
		cudaFreeHost(a[i]);
	cudaFreeHost(a);
}

void _free(void* a, void* b) {
	free(a);
	free(b);
}

void _free(void* a, void* b, void* c) {
	free(a);
	free(b);
	free(c);
}

int divide_ceil(int a, int b) {
	return (a + b - 1) / b;
}

template <typename T>
T* device_to_host(T* arr, int n) {
	T* temp;
	size_t tempBytes = sizeof(T) * n;
	cudaMallocHost(&temp, tempBytes);
	cudaMemcpy(temp, arr, tempBytes, cudaMemcpyDeviceToHost);
	return temp;
}

template <typename T>
T* host_to_device(T* arr, int n) {
	T* temp;
	size_t tempBytes = sizeof(T) * n;
	cudaMalloc(&temp, tempBytes);
	cudaMemcpy(temp, arr, tempBytes, cudaMemcpyHostToDevice);
	return temp;
}

#define gpuerr() { print_cuda_error( __FILE__, __LINE__); }

void print_cuda_error(const char *file, int line) {
	cudaError_t code = cudaGetLastError();
	if (code != cudaSuccess)
		printf("Cuda error at %s %s %d\n", cudaGetErrorName(code), file, line);
}

void print_int_arr(int* arr, int n) {
	printf("[ ");
	int* arr2 = device_to_host(arr, n);
	for (int i = 0; i < n; i++) {
		printf("%d", arr2[i]);
		if (i != n - 1)
			printf(", ");
	}
	printf(" ] n=%d\n", n);
	if (n > 0)
		cudaFreeHost(arr2);
}

void print_char_arr(char* arr, int n) {
	printf("[ ");
	char* arr2 = device_to_host(arr, n);
	for (int i = 0; i < n; i++) {
		printf("%d", arr2[i]);
		if (i != n - 1)
			printf(", ");
	}
	printf(" ] n=%d\n", n);
	if (n > 0)
		cudaFreeHost(arr2);
}

void print_int2_arr(Int2* arr, int n) {
	printf("[ ");
	Int2* arr2 = device_to_host(arr, n);
	for (int i = 0; i < n; i++) {
		printf("(%d %d)", arr2[i].x, arr2[i].y);
		if (i != n - 1)
			printf(", ");
	}
	printf(" ] n=%d\n", n);
	if (n > 0)
		cudaFreeHost(arr2);
}

void print_size_t_arr(size_t* arr, int n) {
	printf("[ ");
	size_t* arr2 = device_to_host(arr, n);
	for (int i = 0; i < n; i++) {
		printf("%lu", arr2[i]);
		if (i != n - 1)
			printf(", ");
	}
	printf(" ] n=%d\n", n);
	cudaFreeHost(arr2);
}

void print_gpu_memory() {
	size_t mf, ma;
	cudaMemGetInfo(&mf, &ma);
	printf("GPU Memory: %'lu / %'lu\n", mf, ma);
}

void print_main_memory() {
	struct sysinfo si;
	sysinfo (&si);
	printf("Main Memory: %'lu / %'lu\n", si.freeram, si.totalram);
}

void print_tp(int verbose, const char* step, size_t throughput) {
	if (verbose)
		printf("step %s completed with throughput: %'lu\n", step, throughput);
}

void print_bandwidth(int chunkLen, int bandwidth, const char* process) {
	float endTime = (float)clock() / CLOCKS_PER_SEC;
	printf("process %s started with bandwidth %'d / %'d %'.0f\n",
	       process, chunkLen, bandwidth, endTime - startTime);
}

void print_v(int verbose, const char* message) {
	if (verbose)
		printf("%s\n", message);
}

size_t get_gpu_memory() {
	size_t mf, ma;
	cudaMemGetInfo(&mf, &ma);
	return mf;
}

size_t get_main_memory() {
	struct sysinfo si;
	sysinfo (&si);
	return si.freeram;
}