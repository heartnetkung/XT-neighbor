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
	// input value is valid, so the bin counts must sum back to n exactly. Note
	// the range divides evenly into the bin count here, so this is a scale
	// regression guard rather than a reproducer of the truncating-bin-width bug
	// -- histogram_large_index_no_precision_loss covers that.
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
	// mirrors a real seqLen=30,000,000 run, and is the case CUB < 2.0.0 got
	// wrong: 30000000 does not divide evenly into 1048576 bins, so its
	// truncating bin width (28 instead of 28.61) pushed the top ~2% of values
	// past the last bin (NVIDIA/cub#489). Index values here also exceed 2^24,
	// the largest integer a float represents exactly, so the float-level
	// workaround for that bug mis-bins them too. histogram_kernel's size_t
	// arithmetic is exact at this scale and must keep every sample.
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

TEST(histogram_matches_cpu_reference, {
	// the two tests above only check that every sample lands somewhere; this one
	// checks that each one lands in the right bin, against a host oracle derived
	// from the bin edges rather than from the kernel's own (offset * nLevel) / range,
	// so a wrong formula on either side surfaces as a mismatch. Bin i of an even
	// histogram holds the offsets [ceil(i * range / nLevel), ceil((i+1) * range /
	// nLevel)). Same awkward configuration as above: the range is not a multiple of
	// the bin count and the indexes run past 2^24.
	int seqLen = 30000000, n = 100000, histogramSize = 1048576;
	size_t range = seqLen;

	int* lesserIndex;
	cudaMallocHost(&lesserIndex, sizeof(int) * n);
	for (int i = 0; i < n; i++)
		lesserIndex[i] = (int)(((size_t)i * 293) % (size_t)seqLen);
	int* lesserIndex_d = host_to_device(lesserIndex, n);

	int* expected;
	cudaMallocHost(&expected, sizeof(int) * histogramSize);
	for (int i = 0; i < histogramSize; i++)
		expected[i] = 0;
	for (int i = 0; i < n; i++) {
		/*largest bin whose lower edge is still <= the value*/
		int lo = 0, hi = histogramSize - 1;
		while (lo < hi) {
			int mid = (lo + hi + 1) / 2;
			size_t edge = ((size_t)mid * range + histogramSize - 1) / histogramSize;
			if ((size_t)lesserIndex[i] >= edge)
				lo = mid;
			else
				hi = mid - 1;
		}
		expected[lo]++;
	}

	int* histogram_d;
	cudaMalloc(&histogram_d, sizeof(int) * histogramSize);
	cal_histogram(lesserIndex_d, histogram_d, histogramSize, 0, seqLen, n);

	check_device_arr(histogram_d, expected, histogramSize);
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
