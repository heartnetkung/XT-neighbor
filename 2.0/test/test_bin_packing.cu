#include <stdio.h>
#include "test_util.cu"
#include "../src/xtn_inner.cu"

TEST(bin_packing, {
	int len = 3, n_level = 3;

	int* histogramInput;
	cudaMallocHost(&histogramInput, sizeof(int) * len * n_level);
	histogramInput[0] = 1; histogramInput[1] = 2; histogramInput[2] = 3;
	histogramInput[3] = 2; histogramInput[4] = 3; histogramInput[5] = 4;
	histogramInput[6] = 4; histogramInput[7] = 1; histogramInput[8] = 1;
	int* histogramInput_d = host_to_device(histogramInput, len * n_level);

	int* deviceInt;
	cudaMalloc((void**)&deviceInt, sizeof(int));

	size_t* output;
	size_t outputLen =  solve_bin_packing(histogramInput, output, 4, len, n_level, int* deviceInt)

	// printf("outputLen: %lu\n", outputLen);
	// print_size_t_arr(output);
})
