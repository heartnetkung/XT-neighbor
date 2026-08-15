/**
 * @file
 * Evidence for why cal_histogram uses its own kernel instead of
 * cub::DeviceHistogram::HistogramEven. See the comment on histogram_kernel in
 * src/cub.cu; this file reproduces the numbers quoted there.
 *
 * Deliberately standalone and not wired into CMakeLists.txt: it needs CUB >= 2.0.0
 * (older CUB computes the bins incorrectly, so timing it is meaningless), while the
 * project itself still builds against older toolkits. Build it directly:
 *
 *   nvcc -O3 -std=c++14 -arch=sm_61 -o histogram_bench histogram_bench.cu
 *
 * With a toolkit older than CUDA 12.0, point it at a standalone CCCL checkout:
 *
 *   nvcc -O3 -std=c++14 -arch=sm_61 -I cccl/cub -I cccl/thrust \
 *        -I cccl/libcudacxx/include -o histogram_bench histogram_bench.cu
 *
 * Two sweeps are reported. The first varies the bin count, which is what decides the
 * winner: CUB privatizes bins into shared memory only up to MAX_PRIVATIZED_SMEM_BINS
 * (256), and above that keeps numThreadBlocks copies of the histogram in global
 * memory, so it falls off as soon as those copies stop fitting in L2. The second
 * varies the input distribution at a fixed bin count, to confirm that atomic collisions
 * on a hot bin are not what drives the result.
 */
#include <cstdio>
#include <cub/device/device_histogram.cuh>

#define divide_ceil(a,b) (((a)+(b)-1)/(b))

const int N = 2000000, SEQ_LEN = 30000000, REP = 20;

/** the kernel from src/cub.cu, copied so this file stays independent of the build */
__global__
void histogram_kernel(int* input, int* output, int minValue, int maxValue, int nLevel, int n) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid >= n)
		return;

	int value = input[tid];
	if ((value < minValue) || (value >= maxValue))
		return;

	size_t range = (size_t)maxValue - (size_t)minValue;
	size_t offset = (size_t)value - (size_t)minValue;
	size_t bin = (offset * (size_t)nLevel) / range;
	if (bin >= (size_t)nLevel)
		bin = nLevel - 1;

	atomicAdd(&output[bin], 1);
}

/** input shapes, all producing values in [0, SEQ_LEN) */
const char* DISTRIBUTIONS[] = {"uniform", "sorted", "runs of 100", "10% one bin",
                               "50% one bin", "100% one bin", "heavy tail"
                              };

void fill(int* a, int mode) {
	for (int i = 0; i < N; i++) {
		size_t v;
		switch (mode) {
		case 0: v = ((size_t)i * 2654435761u) % (size_t)SEQ_LEN; break;
		case 1: v = ((size_t)i * (size_t)SEQ_LEN) / (size_t)N; break;
		case 2: v = ((size_t)(i / 100) * 2654435761u) % (size_t)SEQ_LEN; break;
		case 3: v = (i % 10 == 0) ? 12345 : (((size_t)i * 2654435761u) % (size_t)SEQ_LEN); break;
		case 4: v = (i % 2 == 0) ? 12345 : (((size_t)i * 2654435761u) % (size_t)SEQ_LEN); break;
		case 5: v = 12345; break;
		case 6: v = (size_t)((double)SEQ_LEN / (1.0 + (double)(i % 4096) * (double)(i % 4096))); break;
		default: v = 0;
		}
		a[i] = (int)v;
	}
}

/** milliseconds per call for each implementation, plus cub's scratch requirement */
void time_both(int* input_d, int nLevel, float &cubMs, float &customMs, size_t &bufferSize) {
	int* output_d;
	void* buffer = NULL;
	cudaEvent_t t0, t1;

	cudaMalloc(&output_d, sizeof(int) * nLevel);
	bufferSize = 0;
	cub::DeviceHistogram::HistogramEven(buffer, bufferSize, input_d, output_d,
	                                    nLevel + 1, 0, SEQ_LEN, N);
	cudaMalloc(&buffer, bufferSize);
	cudaEventCreate(&t0);
	cudaEventCreate(&t1);

	/*warm up both paths so neither pays for lazy module loading*/
	cub::DeviceHistogram::HistogramEven(buffer, bufferSize, input_d, output_d, nLevel + 1, 0, SEQ_LEN, N);
	cudaMemset(output_d, 0, sizeof(int) * nLevel);
	histogram_kernel <<< divide_ceil(N, 256), 256 >>>(input_d, output_d, 0, SEQ_LEN, nLevel, N);
	cudaDeviceSynchronize();

	cudaEventRecord(t0);
	for (int i = 0; i < REP; i++)
		cub::DeviceHistogram::HistogramEven(buffer, bufferSize, input_d, output_d, nLevel + 1, 0, SEQ_LEN, N);
	cudaEventRecord(t1);
	cudaEventSynchronize(t1);
	cudaEventElapsedTime(&cubMs, t0, t1);
	cubMs /= REP;

	/*cub zeroes the output itself, so the memset belongs to the custom path's cost*/
	cudaEventRecord(t0);
	for (int i = 0; i < REP; i++) {
		cudaMemset(output_d, 0, sizeof(int) * nLevel);
		histogram_kernel <<< divide_ceil(N, 256), 256 >>>(input_d, output_d, 0, SEQ_LEN, nLevel, N);
	}
	cudaEventRecord(t1);
	cudaEventSynchronize(t1);
	cudaEventElapsedTime(&customMs, t0, t1);
	customMs /= REP;

	cudaFree(buffer);
	cudaFree(output_d);
}

int main() {
	int levels[] = {64, 256, 257, 1024, 4096, 16384, 65536, 262144, 1048576};
	int *input, *input_d;
	float cubMs, customMs;
	size_t bufferSize;
	cudaDeviceProp prop;

	cudaGetDeviceProperties(&prop, 0);
	printf("CUB %d on %s: %d SMs, L2 = %d KiB\n\n", CUB_VERSION, prop.name,
	       prop.multiProcessorCount, prop.l2CacheSize >> 10);

	cudaMallocHost(&input, sizeof(int) * N);
	cudaMalloc(&input_d, sizeof(int) * N);

	printf("bin count sweep, uniform input, %d samples\n", N);
	printf("%9s %10s %11s %8s %13s %8s\n", "bins", "cub(ms)", "custom(ms)", "ratio", "cub scratch", "path");
	fill(input, 0);
	cudaMemcpy(input_d, input, sizeof(int) * N, cudaMemcpyHostToDevice);
	for (int i = 0; i < 9; i++) {
		time_both(input_d, levels[i], cubMs, customMs, bufferSize);
		printf("%9d %10.3f %11.3f %7.2fx %10zu KiB %8s\n", levels[i], cubMs, customMs,
		       cubMs / customMs, bufferSize >> 10, (levels[i] <= 256) ? "smem" : "global");
	}

	for (int li = 6; li <= 8; li += 2) {
		printf("\ndistribution sweep at %d bins\n", levels[li]);
		printf("%14s %10s %11s %8s\n", "distribution", "cub(ms)", "custom(ms)", "ratio");
		for (int mode = 0; mode < 7; mode++) {
			fill(input, mode);
			cudaMemcpy(input_d, input, sizeof(int) * N, cudaMemcpyHostToDevice);
			time_both(input_d, levels[li], cubMs, customMs, bufferSize);
			printf("%14s %10.3f %11.3f %7.2fx\n", DISTRIBUTIONS[mode], cubMs, customMs, cubMs / customMs);
		}
	}

	cudaFree(input_d);
	cudaFreeHost(input);
	return 0;
}
