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
