#include "xtn.h"
#include <sys/sysinfo.h>
#include <time.h>

#define gpuerr() { print_cuda_error( __FILE__, __LINE__); }

void print_cuda_error(const char *file, int line) {
	cudaError_t code = cudaGetLastError();
	if (code != cudaSuccess)
		printf("Cuda error at %s %s %d\n", cudaGetErrorName(code), file, line);
}

int print_err(const char* str) {
	fprintf(stderr, "Error: %s\n", str);
	return ERROR;
}

int print_err_line(const char* str, int line) {
	fprintf(stderr, "Error: %s\n", str);
	fprintf(stderr, "Error line: %d\n", line);
	return ERROR;
}

float startTime = 0;

float clock_start() {
	if (verboseGlobal) {
		startTime = (float)clock() / CLOCKS_PER_SEC;
		return startTime;
	}
	return 0;
}

float get_time() {
	if (verboseGlobal)
		return (float)clock() / CLOCKS_PER_SEC;
	return 0;
}

void print_args(XTNArgs args) {
	printf("XTNArgs{\n");
	printf("\tdistance: %d\n", args.distance);
	printf("\tverbose: %d\n", args.verbose);
	printf("\tseqLen: %'d\n", args.seqLen);
	printf("\tseqPath: \"%s\"\n", args.seqPath);
	printf("\toutputPath: \"%s\"\n", args.outputPath);
	printf("\tmeasure: \"%s\"\n", (args.measure == LEVENSHTEIN) ? "leven" : "hamming");
	printf("\tinfoPath: \"%s\"\n", args.infoPath);
	printf("\tinfoLen: %'d\n", args.infoLen);
	printf("}\n");
}

template <typename T1, typename T2>
void _cudaMalloc(T1* &a, T2* &b, size_t len) {
	cudaMalloc(&a, sizeof(T1)*len); gpuerr();
	cudaMalloc(&b, sizeof(T2)*len); gpuerr();
}
template <typename T1, typename T2, typename T3>
void _cudaMalloc(T1* &a, T2* &b, T3* &c, size_t len) {
	cudaMalloc(&a, sizeof(T1)*len); gpuerr();
	cudaMalloc(&b, sizeof(T2)*len); gpuerr();
	cudaMalloc(&c, sizeof(T3)*len); gpuerr();
}
void _cudaFree(void* a) {
	cudaFree(a); gpuerr();
}
void _cudaFree(void* a, void* b) {
	cudaFree(a); gpuerr();
	cudaFree(b); gpuerr();
}
void _cudaFree(void* a, void* b, void* c) {
	cudaFree(a); gpuerr();
	cudaFree(b); gpuerr();
	cudaFree(c); gpuerr();
}
void _cudaFree(void* a, void* b, void* c, void* d) {
	cudaFree(a); gpuerr();
	cudaFree(b); gpuerr();
	cudaFree(c); gpuerr();
	cudaFree(d); gpuerr();
}
void _cudaFree(void* a, void* b, void* c, void* d, void* e) {
	cudaFree(a); gpuerr();
	cudaFree(b); gpuerr();
	cudaFree(c); gpuerr();
	cudaFree(d); gpuerr();
	cudaFree(e); gpuerr();
}
void _cudaFree(void* a, void* b, void* c, void* d, void* e, void* f) {
	cudaFree(a); gpuerr();
	cudaFree(b); gpuerr();
	cudaFree(c); gpuerr();
	cudaFree(d); gpuerr();
	cudaFree(e); gpuerr();
	cudaFree(f); gpuerr();
}
void _cudaFree(void* a, void* b, void* c, void* d, void* e, void* f, void* g) {
	cudaFree(a); gpuerr();
	cudaFree(b); gpuerr();
	cudaFree(c); gpuerr();
	cudaFree(d); gpuerr();
	cudaFree(e); gpuerr();
	cudaFree(g); gpuerr();
}

void _cudaFreeHost(void* a, void* b) {
	cudaFreeHost(a); gpuerr();
	cudaFreeHost(b); gpuerr();
}

void _cudaFreeHost(void* a, void* b, void* c) {
	cudaFreeHost(a); gpuerr();
	cudaFreeHost(b); gpuerr();
	cudaFreeHost(c); gpuerr();
}

void _cudaFreeHost(void* a, void* b, void* c, void* d) {
	cudaFreeHost(a); gpuerr();
	cudaFreeHost(b); gpuerr();
	cudaFreeHost(c); gpuerr();
	cudaFreeHost(d); gpuerr();
}

template <typename T>
void _cudaFreeHost2D(T** a, int n) {
	for (int i = 0; i < n; i++) {
		cudaFreeHost(a[i]); gpuerr();
	}
	cudaFreeHost(a); gpuerr();
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
	cudaMallocHost(&temp, tempBytes); gpuerr();
	cudaMemcpy(temp, arr, tempBytes, cudaMemcpyDeviceToHost); gpuerr();
	return temp;
}

template <typename T>
T* host_to_device(T* arr, int n) {
	T* temp;
	size_t tempBytes = sizeof(T) * n;
	cudaMalloc(&temp, tempBytes); gpuerr();
	cudaMemcpy(temp, arr, tempBytes, cudaMemcpyHostToDevice); gpuerr();
	return temp;
}

template <typename T>
T* shrink(T* arr, int n) {
	T* temp;
	size_t tempBytes = sizeof(T) * n;
	cudaMalloc(&temp, tempBytes); gpuerr();
	cudaMemcpy(temp, arr, tempBytes, cudaMemcpyDeviceToDevice); gpuerr();
	cudaFree(arr); gpuerr();
	return temp;
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
	if (n > 0) {
		cudaFreeHost(arr2); gpuerr();
	}
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
	if (n > 0) {
		cudaFreeHost(arr2); gpuerr();
	}
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
	if (n > 0) {
		cudaFreeHost(arr2); gpuerr();
	}
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
	cudaFreeHost(arr2); gpuerr();
}

void print_seqinfo_arr(SeqInfo* arr, int n) {
	printf("[ ");
	SeqInfo* arr2 = device_to_host(arr, n);
	for (int i = 0; i < n; i++) {
		printf("(%d %d %d)", arr2[i].frequency, arr2[i].repertoire, arr2[i].originalIndex);
		if (i != n - 1)
			printf(", ");
	}
	printf(" ] n=%d\n", n);
	cudaFreeHost(arr2); gpuerr();
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

void print_tl(const char* stream, size_t tl) {
	if (verboseGlobal)
		printf("stream %s completed with total length: %'lu\n", stream, tl);
}

void print_bandwidth(int chunkLen, int bandwidth, const char* process) {
	if (!verboseGlobal)
		return;
	printf("process %s started with bandwidth %'d / %'d\n",
	       process, chunkLen, bandwidth);
}

void print_v(const char* message) {
	if (verboseGlobal)
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