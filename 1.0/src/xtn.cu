#include <stdio.h>
#include <stdlib.h>
#include "xtn_inner.cu"


void xtn_perform(XTNArgs args, Int3* seq1, XTNOutput* output) {
	int distance = args.distance, verbose = args.verbose, seq1Len = args.seq1Len;
	int* deviceInt;
	cudaMalloc((void**)&deviceInt, sizeof(int));

	//=====================================
	// step 1: transfer input to GPU
	//=====================================
	Int3* seq1Device = host_to_device(seq1, seq1Len);

	print_tp(verbose, "1", seq1Len);

	//=====================================
	// step 2: generate deletion combinations
	//=====================================
	Int3* combinationKeys;
	int* combinationValues;
	int combinationLen =
	    gen_combinations(seq1Device, distance, combinationKeys, combinationValues, seq1Len);

	print_tp(verbose, "2", combinationLen);

	//=====================================
	// step 3.1: cal group by offsets
	//=====================================
	int* combinationValueOffsets, *pairOffsets;
	int offsetLen =
	    cal_offsets(combinationKeys, combinationValues, combinationValueOffsets,
	                pairOffsets, combinationLen, deviceInt);

	print_tp(verbose, "3.1", offsetLen);

	//=====================================
	// step 3.2: perform group by
	//=====================================
	Int2* pairs;
	int pairLen = gen_pairs(combinationValues, combinationValueOffsets,
	                        pairOffsets, pairs, offsetLen, deviceInt);

	print_tp(verbose, "3.2", pairLen);

	//=====================================
	// step 4: postprocessing
	//=====================================
	Int2* pairOut;
	char* distanceOut;
	int outputLen = postprocessing(seq1Device, pairs, distance,
	                               pairOut, distanceOut,
	                               pairLen, deviceInt, seq1Len);

	print_tp(verbose, "3.2", outputLen);

	//=====================================
	// step 5: transfer output to GPU
	//=====================================
	output->len = outputLen;
	output->indexPairs = device_to_host(pairOut, outputLen);
	output->pairwiseDistances = device_to_host(distanceOut, outputLen);

	//=====================================
	// step 6: deallocate
	//=====================================
	_cudaFree(deviceInt, seq1Device, combinationKeys, combinationValues, combinationValueOffsets, pairs);
	_cudaFree(pairOut, distanceOut);

}


void xtn_free(XTNOutput *output) {
	if (output->indexPairs) {
		cudaFreeHost(output->indexPairs);
		output->indexPairs = NULL;
	}
	if (output->pairwiseDistances) {
		cudaFreeHost(output->pairwiseDistances);
		output->pairwiseDistances = NULL;
	}
}