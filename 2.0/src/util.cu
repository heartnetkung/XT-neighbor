#include "xtn.h"

int print_err(const char* str) {
#ifndef TEST_ENV
	fprintf(stderr, "Error: %s\n", str);
#endif
	return ERROR;
}

void print_args(XTNArgs args) {
	printf("XTNArgs{\n");
	printf("\tdistance: %d\n", args.distance);
	printf("\tverbose: %d\n", args.verbose);
	printf("\tseq1Len: %d\n", args.seq1Len);
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

void print_tp(int verbose, const char* step, int throughput) {
	if (verbose)
		printf("step %s completed with throughput: %'d\n", step, throughput);
}

void _cudaFreeHost(void* a, void* b) {
	cudaFreeHost(a);
	cudaFreeHost(b);
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
	int tempBytes = sizeof(T) * n;
	cudaMallocHost(&temp, tempBytes);
	cudaMemcpy(temp, arr, tempBytes, cudaMemcpyDeviceToHost);
	return temp;
}

template <typename T>
T* host_to_device(T* arr, int n) {
	T* temp;
	int tempBytes = sizeof(T) * n;
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
	cudaFreeHost(arr2);
}

void print_size_t_arr(size_t* arr, int n) {
	printf("2.2\n");
	printf("[ ");
	size_t* arr2 = device_to_host(arr, n);
	printf("2.3\n");
	for (int i = 0; i < n; i++) {
		printf("2.31\n");
		printf("%lu", arr2[i]);
		printf("2.32\n");
		if (i != n - 1)
			printf(", ");
		printf("2.33\n");
	}
	printf("2.4\n");
	printf(" ] n=%d\n", n);
	printf("2.5\n");
	cudaFreeHost(arr2);
	printf("2.6\n");
}