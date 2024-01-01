#include <stdio.h>
#include "test_util.cu"
#include "../src/xtn_inner.cu"

TEST(bin_packing, {
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

	int** output;
	int offsetLen =  solve_bin_packing(histogramInput_d, output, len, nLevel, deviceInt, ctx);

	int expectedOffsetLen = 2;
	int expectedOut[][2] = {{3, 6}, {5, 9}, {5, 6}};

	check(offsetLen == expectedOffsetLen);
	for (int i = 0; i < len; i++)
		for (int j = 0; j < offsetLen; j++)
			check(expectedOut[i][j] == output[i][j]);
})

TEST(cal_lowerbounds, {
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

	int* output;
	int offsetLen =  cal_lowerbounds(histogramInput_d, output, len, nLevel, seqLen, deviceInt, ctx);

	for (int i = 0; i < offsetLen; i++)
		printf("%d ", output[i]);
	printf("offsetLen: %d", offsetLen);


	// int expectedOffsetLen = 2;
	// int expectedOut[][2] = {{3, 6}, {5, 9}, {5, 6}};

	// check(offsetLen == expectedOffsetLen);
	// for (int i = 0; i < len; i++)
	// 	for (int j = 0; j < offsetLen; j++)
	// 		check(expectedOut[i][j] == output[i][j]);
})
