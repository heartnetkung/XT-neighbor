#include <cub/device/device_scan.cuh>
#include <cub/device/device_merge_sort.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_select.cuh>
#include <cub/device/device_histogram.cuh>
#include <cub/device/device_reduce.cuh>
#include "codec.cu"

struct Int3Comparator {
	CUB_RUNTIME_FUNCTION __forceinline__ __device__
	bool operator()(const Int3 &lhs, const Int3 &rhs) {
		if (lhs.entry[0] != rhs.entry[0])
			return lhs.entry[0] < rhs.entry[0];
		if (lhs.entry[1] != rhs.entry[1])
			return lhs.entry[1] < rhs.entry[1];
		return lhs.entry[2] < rhs.entry[2];
	}
};

struct Int2Comparator {
	CUB_RUNTIME_FUNCTION __forceinline__ __device__
	bool operator()(const Int2 &lhs, const Int2 &rhs) {
		if (lhs.x != rhs.x)
			return lhs.x < rhs.x;
		return lhs.y < rhs.y;
	}
};

struct SizeTMax {
	CUB_RUNTIME_FUNCTION __forceinline__ __device__
	size_t operator()(const size_t &a, const size_t &b) const {
		return (b > a) ? b : a;
	}
};

template <typename T>
void inclusive_sum(T* input, int n) {
	void *buffer = NULL;
	size_t bufferSize = 0;
	cub::DeviceScan::InclusiveSum(buffer, bufferSize, input, input, n);
	cudaMalloc(&buffer, bufferSize);
	cub::DeviceScan::InclusiveSum(buffer, bufferSize, input, input, n);
	cudaFree(buffer);
}

void inclusive_sum(int* input, size_t* output, int n) {
	void *buffer = NULL;
	size_t bufferSize = 0;
	cub::DeviceScan::InclusiveSum(buffer, bufferSize, input, output, n);
	cudaMalloc(&buffer, bufferSize);
	cub::DeviceScan::InclusiveSum(buffer, bufferSize, input, output, n);
	cudaFree(buffer);
}

void sort_key_values(Int3* keys, int* values, int n) {
	void *buffer = NULL;
	size_t bufferSize = 0;
	Int3Comparator op;
	cub::DeviceMergeSort::SortPairs(buffer, bufferSize, keys, values, n, op);
	cudaMalloc(&buffer, bufferSize);
	cub::DeviceMergeSort::SortPairs(buffer, bufferSize, keys, values, n, op);
	cudaFree(buffer);
}

void sort_int2(Int2* input, int n) {
	void *buffer = NULL;
	size_t bufferSize = 0;
	Int2Comparator op;
	cub::DeviceMergeSort::SortKeys(buffer, bufferSize, input, n, op);
	cudaMalloc(&buffer, bufferSize);
	cub::DeviceMergeSort::SortKeys(buffer, bufferSize, input, n, op);
	cudaFree(buffer);
}

template <typename T>
void unique_counts(T* keys, int* output, int* outputLen, int n) {
	void *buffer = NULL;
	size_t bufferSize = 0;
	T* dummy;
	cudaMalloc(&dummy, sizeof(T)*n);
	cub::DeviceRunLengthEncode::Encode(
	    buffer, bufferSize, keys, dummy, output, outputLen, n);
	cudaMalloc(&buffer, bufferSize);
	cub::DeviceRunLengthEncode::Encode(
	    buffer, bufferSize, keys, dummy, output, outputLen, n);
	cudaFree(buffer);
	cudaFree(dummy);
}

void unique(Int2* input, Int2* output, int* outputLen, int n) {
	void *buffer = NULL;
	size_t bufferSize = 0;
	cub::DeviceSelect::Unique(buffer, bufferSize, input, output, outputLen, n);
	cudaMalloc(&buffer, bufferSize);
	cub::DeviceSelect::Unique(buffer, bufferSize, input, output, outputLen, n);
	cudaFree(buffer);
}

template <typename T1, typename T2>
void double_flag(T1* input1, T2* input2, char* flags, T1* output1, T2* output2, int* outputLen, int n) {
	void *buffer = NULL, *buffer2 = NULL;
	size_t bufferSize = 0, bufferSize2 = 0;
	cub::DeviceSelect::Flagged(buffer, bufferSize, input1, flags, output1, outputLen, n);
	cudaMalloc(&buffer, bufferSize);
	cub::DeviceSelect::Flagged(buffer, bufferSize, input1, flags, output1, outputLen, n);
	cub::DeviceSelect::Flagged(buffer2, bufferSize2, input2, flags, output2, outputLen, n);
	cudaMalloc(&buffer2, bufferSize2);
	cub::DeviceSelect::Flagged(buffer2, bufferSize2, input2, flags, output2, outputLen, n);
	_cudaFree(buffer, buffer2);
}

template <typename T>
void histogram(T* input, int* output, int nLevel, T maxValue, int n) {
	void *buffer = NULL;
	size_t bufferSize = 0;
	unsigned int minValue = 0;
	cub::DeviceHistogram::HistogramEven(buffer, bufferSize,
	                                    input, output, nLevel + 1, minValue, maxValue, n);
	cudaMalloc(&buffer, bufferSize);
	cub::DeviceHistogram::HistogramEven(buffer, bufferSize,
	                                    input, output, nLevel + 1, minValue, maxValue, n);
	cudaFree(buffer);
}

void inclusive_sum_by_key(int* keyIn, int* valueIn, size_t* valueOut, int n) {
	void *buffer = NULL;
	size_t bufferSize = 0;
	cub::DeviceScan::InclusiveSumByKey(
	    buffer, bufferSize, keyIn, valueIn, valueOut, n);
	cudaMalloc(&buffer, bufferSize);
	cub::DeviceScan::InclusiveSumByKey(
	    buffer, bufferSize, keyIn, valueIn, valueOut, n);
	cudaFree(buffer);
}

void max_by_key(int* keyIn, size_t* valueIn, size_t* valueOut, int* outputLen, int n) {
	void *buffer = NULL;
	size_t bufferSize = 0;
	int* dummy;
	SizeTMax op;

	printf("1.1\n");
	cudaMalloc(&dummy, sizeof(int)*n);
	printf("1.2\n");
	cub::DeviceReduce::ReduceByKey(buffer, bufferSize, keyIn,
	                               dummy, valueIn, valueOut, outputLen, op, n);
	printf("1.3\n");
	cudaMalloc(&buffer, bufferSize);
	printf("1.4\n");
	cub::DeviceReduce::ReduceByKey(buffer, bufferSize, keyIn,
	                               dummy, valueIn, valueOut, outputLen, op, n);
	printf("1.5\n");
	_cudaFree(buffer, dummy);
}
