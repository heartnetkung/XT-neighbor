#include <cub/device/device_scan.cuh>
#include <cub/device/device_merge_sort.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_select.cuh>
#include <cub/device/device_histogram.cuh>
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
	int operator()(const size_t &a, const size_t &b) const {
		return a + b;
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

void sort_key_values(Int3* keys, int* values, int n) {
	void *buffer = NULL;
	size_t bufferSize = 0;
	Int3Comparator op;
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

template <typename T>
void unique_counts(T* keys, int* output, int* outputLen, int n) {
	void *buffer = NULL;
	size_t bufferSize = 0;
	T* dummy;
	cudaMalloc(&dummy, sizeof(T)*n); gpuerr();
	cub::DeviceRunLengthEncode::Encode(
	    buffer, bufferSize, keys, dummy, output, outputLen, n); gpuerr();
	cudaMalloc(&buffer, bufferSize); gpuerr(); /*~5% memory*/
	cub::DeviceRunLengthEncode::Encode(
	    buffer, bufferSize, keys, dummy, output, outputLen, n); gpuerr();
	cudaFree(buffer); gpuerr();
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
	size_t bufferSize = 0, bufferSize2 = 0;
	cub::DeviceSelect::Flagged(buffer, bufferSize, input1, flags, output1, outputLen, n); gpuerr();
	cudaMalloc(&buffer, bufferSize); gpuerr(); /*~2% memory*/
	cub::DeviceSelect::Flagged(buffer, bufferSize, input1, flags, output1, outputLen, n); gpuerr();
	cudaFree(buffer); gpuerr();
}

template <typename T>
void cal_histogram(T* input, int* output, int nLevel, T minValue, T maxValue, int n) {
	void *buffer = NULL;
	size_t bufferSize = 0;
	cub::DeviceHistogram::HistogramEven(buffer, bufferSize,
	                                    input, output, nLevel + 1, minValue, maxValue, n); gpuerr();
	cudaMalloc(&buffer, bufferSize); gpuerr(); /*5-10% memory*/
	cub::DeviceHistogram::HistogramEven(buffer, bufferSize,
	                                    input, output, nLevel + 1, minValue, maxValue, n); gpuerr();
	cudaFree(buffer); gpuerr();
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