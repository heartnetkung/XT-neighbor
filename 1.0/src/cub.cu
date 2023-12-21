#include <cub/device/device_scan.cuh>
#include <cub/device/device_merge_sort.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_select.cuh>
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

void inclusive_sum(int* input, int n) {
	void *buffer = NULL;
	size_t bufferSize = 0;
	cub::DeviceScan::InclusiveSum(buffer, bufferSize, input, input, n);
	cudaMalloc(&buffer, bufferSize);
	cub::DeviceScan::InclusiveSum(buffer, bufferSize, input, input, n);
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

void unique_counts(Int3* keys, int* output, int* outputLen, int n) {
	void *buffer = NULL;
	size_t bufferSize = 0;
	Int3* dummy;
	cudaMalloc(&dummy, sizeof(Int3)*n);
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

void double_flag(Int2* input1, char* input2, char* flags, Int2* output1, char* output2, int* outputLen, int n) {
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
