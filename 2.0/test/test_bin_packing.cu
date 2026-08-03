#include <stdio.h>
#include "test_util.cu"
#include "../src/xtn_inner.cu"

TEST(bin_packing_offsets, {
	int len = 3, nLevel = 3;

	int* histogramInput;
	cudaMallocHost(&histogramInput, sizeof(int) * len * nLevel);
	histogramInput[0] = 1; histogramInput[1] = 2; histogramInput[2] = 3;
	histogramInput[3] = 2; histogramInput[4] = 3; histogramInput[5] = 4;
	histogramInput[6] = 4; histogramInput[7] = 1; histogramInput[8] = 1;
	int* histogramInput_d = host_to_device(histogramInput, len * nLevel);

	int* deviceInt;
	cudaMalloc(&deviceInt, sizeof(int));
	MemoryContext ctx;
	ctx.maxThroughputExponent = 4;
	ctx.histogramSize = nLevel;

	int** output;
	int offsetLen =  solve_bin_packing_offsets(histogramInput_d, output, len, deviceInt, ctx);

	int expectedOffsetLen = 2;
	int expectedOut[][2] = {{3, 6}, {5, 9}, {5, 6}};

	check(offsetLen == expectedOffsetLen);
	for (int i = 0; i < len; i++)
		check_arr(expectedOut[i], output[i], offsetLen);
})

TEST(bin_packing_offsets_single_row, {
	// reproduces the real-run scenario: n=1 row, and the whole row's total
	// falls well under a single bin (maxThroughputExponent large relative to the data),
	// so every column should collapse into exactly one bin-packing group.
	int len = 1, nLevel = 5;

	int* histogramInput;
	cudaMallocHost(&histogramInput, sizeof(int) * len * nLevel);
	histogramInput[0] = 3; histogramInput[1] = 0; histogramInput[2] = 5;
	histogramInput[3] = 2; histogramInput[4] = 0;
	int* histogramInput_d = host_to_device(histogramInput, len * nLevel);

	int* deviceInt;
	cudaMalloc(&deviceInt, sizeof(int));
	MemoryContext ctx;
	ctx.maxThroughputExponent = 10;
	ctx.histogramSize = nLevel;

	int** output;
	int offsetLen = solve_bin_packing_offsets(histogramInput_d, output, len, deviceInt, ctx);

	int expectedOffsetLen = 1;
	int expectedOut[][1] = {{10}};

	check(offsetLen == expectedOffsetLen);
	for (int i = 0; i < len; i++)
		check_arr(expectedOut[i], output[i], offsetLen);
})

TEST(histogram_large_scale_no_loss, {
	// mirrors stream_handler2/3's real cal_histogram call: ~1.69M in-range
	// int values binned into seqLen=50000 buckets over [0, seqLen). Every
	// input value is valid, so the bin counts must sum back to n exactly.
	// cub::DeviceHistogram::HistogramEven silently drops samples at this
	// exact scale (large bin count x large sample count), which is why
	// cal_histogram uses its own atomic kernel instead.
	int seqLen = 50000;
	int n = 1687864;

	int* lesserIndex;
	cudaMallocHost(&lesserIndex, sizeof(int) * n);
	for (int i = 0; i < n; i++)
		lesserIndex[i] = i % seqLen;
	int* lesserIndex_d = host_to_device(lesserIndex, n);

	int histogramSize = seqLen;
	int* histogram_d;
	cudaMalloc(&histogram_d, sizeof(int) * histogramSize);
	cal_histogram(lesserIndex_d, histogram_d, histogramSize, 0, seqLen, n);

	int* histogram;
	cudaMallocHost(&histogram, sizeof(int) * histogramSize);
	cudaMemcpy(histogram, histogram_d, sizeof(int) * histogramSize, cudaMemcpyDeviceToHost);

	long total = 0;
	for (int i = 0; i < histogramSize; i++)
		total += histogram[i];

	check(total == n);
})

TEST(histogram_large_index_no_precision_loss, {
	// index values well past 2^24 (the largest integer a float can represent
	// exactly), mirroring a real seqLen=30,000,000 run. A float-cast
	// histogram implementation silently mis-bins values in this range
	// (confirmed against a real run); cal_histogram's plain size_t integer
	// arithmetic must not.
	int seqLen = 30000000;
	int n = 2000000;

	int* lesserIndex;
	cudaMallocHost(&lesserIndex, sizeof(int) * n);
	for (int i = 0; i < n; i++)
		lesserIndex[i] = seqLen - 1 - (i % seqLen);
	int* lesserIndex_d = host_to_device(lesserIndex, n);

	int histogramSize = 1048576;
	int* histogram_d;
	cudaMalloc(&histogram_d, sizeof(int) * histogramSize);
	cal_histogram(lesserIndex_d, histogram_d, histogramSize, 0, seqLen, n);

	int* histogram;
	cudaMallocHost(&histogram, sizeof(int) * histogramSize);
	cudaMemcpy(histogram, histogram_d, sizeof(int) * histogramSize, cudaMemcpyDeviceToHost);

	long total = 0;
	for (int i = 0; i < histogramSize; i++)
		total += histogram[i];

	check(total == n);
})

TEST(bin_packing_lowerbounds, {
	int len = 3, nLevel = 3, seqLen = 35;

	int* histogramInput;
	cudaMallocHost(&histogramInput, sizeof(int) * len * nLevel);
	histogramInput[0] = 1; histogramInput[1] = 2; histogramInput[2] = 3;
	histogramInput[3] = 2; histogramInput[4] = 3; histogramInput[5] = 4;
	histogramInput[6] = 4; histogramInput[7] = 1; histogramInput[8] = 1;
	int* histogramInput_d = host_to_device(histogramInput, len * nLevel);

	int* deviceInt;
	cudaMalloc(&deviceInt, sizeof(int));
	MemoryContext ctx;
	ctx.maxThroughputExponent = 4;
	ctx.histogramSize = nLevel;

	int* output;
	int offsetLen = solve_bin_packing_lowerbounds(histogramInput_d, output, len, seqLen, deviceInt, ctx);

	int expectedOffsetLen = 2;
	int expectedOut[] = {22, 34};

	check(offsetLen == expectedOffsetLen);
	check_arr(output, expectedOut, offsetLen);
})
