#include <cub/device/device_scan.cuh>
#include <cub/device/device_merge_sort.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_select.cuh>
#include <cub/device/device_reduce.cuh>
#include "codec.cu"

/**
 * @file
 * Listing of all GPU parallel primitives that use CUB library
 * (everything other than map and expand). Follows Facade design pattern.
 */

struct Int3Comparator {
	CUB_RUNTIME_FUNCTION __forceinline__ __device__
	bool operator()(const Int3 &lhs, const Int3 &rhs) {
		/*intentionally sort the second int first as it makes histograms more evenly distributed*/
		if (lhs.entry[1] != rhs.entry[1])
			return lhs.entry[1] < rhs.entry[1];
		if (lhs.entry[0] != rhs.entry[0])
			return lhs.entry[0] < rhs.entry[0];
		// make sure it's irreflexive https://en.cppreference.com/w/cpp/concepts/strict_weak_order
		if (lhs.entry[2] == rhs.entry[2])
			return false;
		return lhs.entry[2] < rhs.entry[2];
	}
};

struct Int2Comparator {
	CUB_RUNTIME_FUNCTION __forceinline__ __device__
	bool operator()(const Int2 &lhs, const Int2 &rhs) {
		if (lhs.x != rhs.x)
			return lhs.x < rhs.x;
		// make sure it's irreflexive https://en.cppreference.com/w/cpp/concepts/strict_weak_order
		if (lhs.y == rhs.y)
			return false;
		return lhs.y < rhs.y;
	}
};

struct IntMax {
	CUB_RUNTIME_FUNCTION __forceinline__ __device__
	int operator()(const int &a, const int &b) const {
		return (b > a) ? b : a;
	}
};

struct Sum {
	CUB_RUNTIME_FUNCTION __forceinline__ __device__
	size_t operator()(const size_t &a, const size_t &b) const {
		return a + b;
	}
};

struct SeqInfoComparator {
	char* allStr = NULL;
	unsigned int* offsets = NULL;

	CUB_RUNTIME_FUNCTION __forceinline__ __device__
	bool operator()(const SeqInfo &el1, const SeqInfo &el2) {
		unsigned int start1 = offsets[el1.originalIndex], start2 = offsets[el2.originalIndex];
		int len1 = offsets[el1.originalIndex + 1] - start1;
		int len2 = offsets[el2.originalIndex + 1] - start2;
		int shorterLen = (len1 < len2) ? len1 : len2;

		for (int i = 0; i < shorterLen; i++) {
			char c1 = allStr[start1 + i], c2 = allStr[start2 + i];
			if (c1 != c2)
				return c1 < c2;
		}

		// exact equal but irreflexive property is needed
		// https://en.cppreference.com/w/cpp/concepts/strict_weak_order
		if (len1 == len2)
			return false;
		return len1 < len2;
	}
};

template <typename T>
void inclusive_sum(T* input, T* output, int n) {
	void *buffer = NULL;
	size_t bufferSize = 0;
	cub::DeviceScan::InclusiveSum(buffer, bufferSize, input, output, n); gpuerr();
	cudaMalloc(&buffer, bufferSize); gpuerr(); /*<1% memory*/
	cub::DeviceScan::InclusiveSum(buffer, bufferSize, input, output, n); gpuerr();
	cudaFree(buffer); gpuerr();
}

template <typename T>
void inclusive_sum(T* input, int n) {
	inclusive_sum(input, input, n);
}

template <typename T>
void sort_key_values(Int3* keys, T* values, int n) {
	void *buffer = NULL;
	size_t bufferSize = 0;
	Int3Comparator op;
	cub::DeviceMergeSort::SortPairs(buffer, bufferSize, keys, values, n, op); gpuerr();
	cudaMalloc(&buffer, bufferSize); gpuerr(); /*16x memory*/
	cub::DeviceMergeSort::SortPairs(buffer, bufferSize, keys, values, n, op); gpuerr();
	cudaFree(buffer); gpuerr();
}

void sort_key_values2(Int2* keys, size_t* values, int n) {
	void *buffer = NULL;
	size_t bufferSize = 0;
	Int2Comparator op;
	cub::DeviceMergeSort::SortPairs(buffer, bufferSize, keys, values, n, op); gpuerr();
	cudaMalloc(&buffer, bufferSize); gpuerr(); /*16x memory*/
	cub::DeviceMergeSort::SortPairs(buffer, bufferSize, keys, values, n, op); gpuerr();
	cudaFree(buffer); gpuerr();
}

void sort_int2(Int2* input, int n) {
	void *buffer = NULL;
	size_t bufferSize = 0;
	Int2Comparator op;
	cub::DeviceMergeSort::SortKeys(buffer, bufferSize, input, n, op); gpuerr();
	cudaMalloc(&buffer, bufferSize); gpuerr(); /*8x memory*/
	cub::DeviceMergeSort::SortKeys(buffer, bufferSize, input, n, op); gpuerr();
	cudaFree(buffer); gpuerr();
}

void sort_info(SeqInfo* input, char* allStr, unsigned int* offsets, int n) {
	void *buffer = NULL;
	size_t bufferSize = 0;
	SeqInfoComparator op = {.allStr = allStr, .offsets = offsets};
	cub::DeviceMergeSort::SortKeys(buffer, bufferSize, input, n, op); gpuerr();
	cudaMalloc(&buffer, bufferSize); gpuerr(); /*8x memory*/
	cub::DeviceMergeSort::SortKeys(buffer, bufferSize, input, n, op); gpuerr();
	cudaFree(buffer); gpuerr();
}

template <typename T>
void unique_counts(T* keys, int* output, T* uniqueOut, int* outputLen, int n) {
	void *buffer = NULL;
	size_t bufferSize = 0;
	cub::DeviceRunLengthEncode::Encode(
	    buffer, bufferSize, keys, uniqueOut, output, outputLen, n); gpuerr();
	cudaMalloc(&buffer, bufferSize); gpuerr(); /*~5% memory*/
	cub::DeviceRunLengthEncode::Encode(
	    buffer, bufferSize, keys, uniqueOut, output, outputLen, n); gpuerr();
	cudaFree(buffer); gpuerr();
}

template <typename T>
void unique_counts(T* keys, int* output, int* outputLen, int n) {
	T* dummy;
	cudaMalloc(&dummy, sizeof(T)*n); gpuerr();
	unique_counts(keys, output, dummy, outputLen, n);
	cudaFree(dummy); gpuerr();
}

void unique(Int2* input, Int2* output, int* outputLen, int n) {
	void *buffer = NULL;
	size_t bufferSize = 0;
	cub::DeviceSelect::Unique(buffer, bufferSize, input, output, outputLen, n); gpuerr();
	cudaMalloc(&buffer, bufferSize); gpuerr(); /*~1% memory*/
	cub::DeviceSelect::Unique(buffer, bufferSize, input, output, outputLen, n); gpuerr();
	cudaFree(buffer); gpuerr();
}

template <typename T1>
void flag(T1* input1, char* flags, T1* output1, int* outputLen, int n) {
	void *buffer = NULL;
	size_t bufferSize = 0;
	cub::DeviceSelect::Flagged(buffer, bufferSize, input1, flags, output1, outputLen, n); gpuerr();
	cudaMalloc(&buffer, bufferSize); gpuerr(); /*~2% memory*/
	cub::DeviceSelect::Flagged(buffer, bufferSize, input1, flags, output1, outputLen, n); gpuerr();
	cudaFree(buffer); gpuerr();
}

/**
 * per-thread bucketing kernel backing cal_histogram. Bins are computed in size_t so
 * that offset * nLevel is exact for every nLevel and range we use.
 *
 * Used instead of cub::DeviceHistogram::HistogramEven, for a reason that is about the
 * shape of our workload rather than about any CUB version. CUB privatizes bins into
 * shared memory only while the bin count is <= MAX_PRIVATIZED_SMEM_BINS, which is 256
 * (an enum on DispatchHistogram itself, not part of the tuning policy, so no GPU
 * generation changes it). Above 256 bins that strategy is switched off and CUB issues
 * the same global atomics this kernel does, except into numThreadBlocks separate
 * copies of the histogram, plus an init pass to zero them and an aggregation pass to
 * sum them. Once those copies stop fitting in L2 it loses badly. We run
 * histogramSize = 65536, or 1048576 when seqLen > 10M (see initMemory), i.e. 256x to
 * 4096x past the regime CUB's histogram is built for -- its API speaks in pixels,
 * rows, and RGBA channels, where 256 bins is the natural maximum.
 *
 * Measured on CUB 2.3.2, 2M samples, MX250 (3 SMs, 512 KiB L2), ms/call:
 *
 *   bins        cub    this     bins        cub    this
 *   64        0.308   1.237     16384     4.127   0.820  <- cub scratch > L2
 *   256       0.987   0.993     65536    11.087   0.787
 *   1024      0.992   0.989     262144   13.259   9.621
 *   4096      0.926   0.985     1048576  15.240  12.578
 *
 * The crossover is the privatized working set (numThreadBlocks * nLevel * 4 bytes)
 * passing L2, and it moves the wrong way on bigger GPUs: the copy count scales with SM
 * count while this kernel always touches one histogram. Concentrated input does not
 * rescue CUB either -- across uniform, sorted, run-length, single-hot-bin and
 * heavy-tail inputs it stays 1.2x-3.3x slower at 1048576 bins. Same-address atomics
 * cost this kernel at most ~1.6x, since a warp's collisions are aggregated before they
 * reach L2. See benchmark/histogram_bench.cu to reproduce, including on newer
 * hardware (only sm_90 gets its own histogram tuning; sm_50 through sm_89 all share
 * Policy500).
 *
 * Correctness was the historical reason and still rules out old CUB: before 2.0.0
 * HistogramEven derived its bin width with a truncating integer division, so any range
 * that is not an exact multiple of the bin count sized every bin wrong and samples
 * above the last boundary were incremented out of range (NVIDIA/cub#489, #479; fixed
 * by #487). Passing the levels as float to dodge that loses precision above 2^24,
 * which the seqLen > 10M configuration reaches. CUB >= 2.0.0 computes bins with the
 * same integer formula used here and is exact; it is simply slower.
*/
template <typename T>
__global__
void histogram_kernel(T* input, int* output, T minValue, T maxValue, int nLevel, int n) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid >= n)
		return;

	T value = input[tid];
	if ((value < minValue) || (value >= maxValue))
		return;

	size_t range = (size_t)maxValue - (size_t)minValue;
	size_t offset = (size_t)value - (size_t)minValue;
	size_t bin = (offset * (size_t)nLevel) / range;
	if (bin >= (size_t)nLevel)
		bin = nLevel - 1;

	atomicAdd(&output[bin], 1);
}

template <typename T>
void cal_histogram(T* input, int* output, int nLevel, T minValue, T maxValue, int n) {
	cudaMemset(output, 0, sizeof(int) * nLevel); gpuerr();

	int nThreads = 256;
	int nBlocks = divide_ceil(n, nThreads);
	if (nBlocks == 0)
		nBlocks = 1;
	histogram_kernel <<< nBlocks, nThreads >>>(input, output, minValue, maxValue, nLevel, n); gpuerr();
}

template <typename T>
void inclusive_sum_by_key(int* keyIn, T* valueInOut, int n) {
	void *buffer = NULL;
	size_t bufferSize = 0;
	cub::DeviceScan::InclusiveSumByKey(
	    buffer, bufferSize, keyIn, valueInOut, valueInOut, n); gpuerr();
	cudaMalloc(&buffer, bufferSize); gpuerr(); /*2% memory*/
	cub::DeviceScan::InclusiveSumByKey(
	    buffer, bufferSize, keyIn, valueInOut, valueInOut, n); gpuerr();
	cudaFree(buffer); gpuerr();
}

void max_by_key(int* keyIn, int* valueIn, int* valueOut, int* outputLen, int n) {
	void *buffer = NULL;
	size_t bufferSize = 0;
	int* dummy;
	IntMax op;

	cudaMalloc(&dummy, sizeof(int)*n); gpuerr();
	cub::DeviceReduce::ReduceByKey(buffer, bufferSize, keyIn,
	                               dummy, valueIn, valueOut, outputLen, op, n); gpuerr();
	cudaMalloc(&buffer, bufferSize); gpuerr(); /*3% memory*/
	cub::DeviceReduce::ReduceByKey(buffer, bufferSize, keyIn,
	                               dummy, valueIn, valueOut, outputLen, op, n); gpuerr();
	_cudaFree(buffer, dummy); gpuerr();
}

void sum_by_key(Int2* keyIn, Int2* keyOut, size_t* valueIn, size_t* valueOut, int* outputLen, int n) {
	void *buffer = NULL;
	size_t bufferSize = 0;
	Sum op;

	cub::DeviceReduce::ReduceByKey(buffer, bufferSize, keyIn,
	                               keyOut, valueIn, valueOut, outputLen, op, n); gpuerr();
	cudaMalloc(&buffer, bufferSize); gpuerr();
	cub::DeviceReduce::ReduceByKey(buffer, bufferSize, keyIn,
	                               keyOut, valueIn, valueOut, outputLen, op, n); gpuerr();
	_cudaFree(buffer); gpuerr();
}