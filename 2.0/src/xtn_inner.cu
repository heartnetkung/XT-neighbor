#include "generate_combination.cu"
#include "cub.cu"
#include "kernel.cu"
#include "codec.cu"
#include "stream.cu"
#include <limits.h>

const int NUM_THREADS = 256;
const int HISTOGRAM_SIZE = 16; //TODO
const unsigned int UINT_MIN = 0;

//=====================================
// Private Functions
//=====================================

int cal_offsets(Int3* inputKeys, int* &inputOffsets, int* &outputLengths, int n, int* buffer) {
	// cal inputOffsets
	cudaMalloc(&inputOffsets, sizeof(int)*n); gpuerr();
	unique_counts(inputKeys, inputOffsets, buffer, n); gpuerr();
	int nUnique = transfer_last_element(buffer, 1); gpuerr();

	// cal outputLengths
	int nUniqueBlock = divide_ceil(nUnique, NUM_THREADS);
	cudaMalloc(&outputLengths, sizeof(int)*nUnique); gpuerr();
	cal_pair_len <<< nUniqueBlock, NUM_THREADS>>>(inputOffsets, outputLengths, nUnique); gpuerr();
	inclusive_sum(inputOffsets, nUnique); gpuerr();
	return nUnique;
}

int gen_pairs(int* input, int* inputOffsets, int* outputLengths, Int2* &output, int n) {
	int* outputOffsets;

	// cal outputOffsets
	cudaMalloc(&outputOffsets, n * sizeof(int));
	inclusive_sum(outputLengths, outputOffsets, n); gpuerr();
	int outputLen = transfer_last_element(outputOffsets, n); gpuerr();

	//generate pairs
	int nBlock = divide_ceil(n, NUM_THREADS);
	cudaMalloc(&output, sizeof(Int2)*outputLen); gpuerr();
	generate_pairs <<< nBlock, NUM_THREADS>>>(input, output,
	        inputOffsets, outputOffsets, n); gpuerr();

	cudaFree(outputOffsets);
	return outputLen;
}

int gen_smaller_index(int* input, int* inputOffsets, int* outputLengths, int* &output, int n) {
	int* outputOffsets;

	// cal outputOffsets
	cudaMalloc(&outputOffsets, n * sizeof(int));
	inclusive_sum(outputLengths, outputOffsets, n); gpuerr();
	int outputLen = transfer_last_element(outputOffsets, n); gpuerr();

	//generate pairs
	int nBlock = divide_ceil(n, NUM_THREADS);
	cudaMalloc(&output, sizeof(Int2)*outputLen); gpuerr();
	generate_smaller_index <<< nBlock, NUM_THREADS>>>(input, output,
	        inputOffsets, outputOffsets, n); gpuerr();

	cudaFree(outputOffsets);
	return outputLen;
}

int postprocessing(Int3* seq, Int2* input, int distance,
                   Int2* &pairOutput, char* &distanceOutput,
                   int n, int* buffer, int seqLen) {
	Int2* uniquePairs;
	char* uniqueDistances, *flags;

	// filter duplicate
	cudaMalloc(&uniquePairs, sizeof(Int2)*n); gpuerr();
	sort_int2(input, n); gpuerr();
	unique(input, uniquePairs, buffer, n); gpuerr();

	// cal levenshtein
	int uniqueLen = transfer_last_element(buffer, 1); gpuerr();
	int byteRequirement = sizeof(char) * uniqueLen;
	int uniqueLenBlock = divide_ceil(uniqueLen, NUM_THREADS);
	cudaMalloc(&flags, byteRequirement); gpuerr();
	cudaMalloc(&uniqueDistances, byteRequirement); gpuerr();
	cudaMalloc(&distanceOutput, byteRequirement); gpuerr();
	cudaMalloc(&pairOutput, sizeof(Int2)*uniqueLen); gpuerr();
	cal_levenshtein <<< uniqueLenBlock, NUM_THREADS>>>(
	    seq, uniquePairs, distance, uniqueDistances, flags, uniqueLen, seqLen); gpuerr();

	// filter levenshtein
	double_flag(uniquePairs, uniqueDistances, flags, pairOutput,
	            distanceOutput, buffer, uniqueLen); gpuerr();
	_cudaFree(uniquePairs, uniqueDistances, flags); gpuerr();
	return transfer_last_element(buffer, 1);
}

void make_output(Int2* pairOut, char* distanceOut, size_t len, XTNOutput &output) {
	output.indexPairs = device_to_host(pairOut, len); gpuerr();
	output.pairwiseDistances = device_to_host(distanceOut, len); gpuerr();
	output.len = len;
}

void gen_next_chunk(Chunk<Int3> keyInput, Chunk<int> valueInput,
                    Chunk<Int3> &keyOutput, Chunk<int> &valueOutput,
                    int* valueOffsets, int offsetLen, int lowerbound, int* buffer) {
	char* flags;
	cudaMalloc(&flags, sizeof(char)*valueInput.len); gpuerr();
	cudaMemset(flags, 1, sizeof(char)*valueInput.len); gpuerr();
	int inputBlocks = divide_ceil(offsetLen, NUM_THREADS);

	flag_lowerbound <<< inputBlocks, NUM_THREADS>>>(
	    valueInput.ptr, valueOffsets, flags, lowerbound, offsetLen); gpuerr();
	double_flag(keyInput.ptr, valueInput.ptr, flags, keyOutput.ptr, valueOutput.ptr,
	            buffer, keyInput.len); gpuerr();

	int outputLen = transfer_last_element(buffer, 1); gpuerr();
	keyOutput.len = outputLen;
	valueOutput.len = outputLen;
	cudaFree(flags); gpuerr();
}

int solve_next_bin(int* chunksizes, int start, int maxReadableSize, int n) {
	int ans = 0, len = 0;
	for (int i = start; i < n; i++) {
		int currentChunkSize = chunksizes[i];
		if (len + currentChunkSize > maxReadableSize)
			break;
		len += currentChunkSize;
		ans++;
	}
	return ans;
}

//=====================================
// Public Functions
//=====================================

int solve_bin_packing(int* histograms, int** &offsetOutput,
                      int maxProcessingExponent, int n, int nLevel, int* buffer) {
	int* rowIndex, *assignment, *output1d, *output1dPtr;

	int len2d = n * nLevel;
	int inputBlocks = divide_ceil(n, NUM_THREADS);
	int inputBlocks2 = divide_ceil(nLevel, NUM_THREADS);
	cudaMalloc(&rowIndex, sizeof(int) * len2d); gpuerr();
	cudaMalloc(&assignment, sizeof(int) * len2d); gpuerr();
	cudaMalloc(&output1d, sizeof(int) * len2d); gpuerr();
	cudaMallocHost(&offsetOutput, sizeof(int*) * n); gpuerr();

	//solve bin packing
	make_row_index <<< inputBlocks, NUM_THREADS>>>(rowIndex, n, nLevel);
	inclusive_sum_by_key(rowIndex, histograms, len2d); gpuerr();
	gen_assignment <<< inputBlocks2, NUM_THREADS >>>(
	    histograms, assignment, maxProcessingExponent, n, nLevel); gpuerr();
	max_by_key(assignment, histograms, output1d, buffer, len2d); gpuerr();

	//make output
	int outputLen = transfer_last_element(buffer, 1); gpuerr();
	if (outputLen % n != 0)
		print_err("bin_packing outputLen is not divisible by inputLen");
	int offsetLen = outputLen / n;
	output1dPtr = output1d;
	for (int i = 0; i < n; i++) {
		offsetOutput[i] = device_to_host(output1dPtr, offsetLen); gpuerr();
		output1dPtr += offsetLen;
	}

	_cudaFree(rowIndex, assignment, output1d); gpuerr();
	return offsetLen;
}

void stream_handler1(Chunk<Int3> input, Int3* &deletionsOutput, int* &indexOutput,
                     int* &histogramOutput, int &outputLen, int distance) {
	// boilerplate
	int *combinationOffsets;
	unsigned int *histogramValue;
	int inputBlocks = divide_ceil(input.len, NUM_THREADS);

	// cal combinationOffsets
	cudaMalloc(&combinationOffsets, sizeof(int)*input.len);	gpuerr();
	cal_combination_len <<< inputBlocks, NUM_THREADS >>>(
	    input.ptr, distance, combinationOffsets, input.len); gpuerr();
	inclusive_sum(combinationOffsets, input.len); gpuerr();
	outputLen = transfer_last_element(combinationOffsets, input.len); gpuerr();

	// generate combinations
	cudaMalloc(&deletionsOutput, sizeof(Int3)*outputLen); gpuerr();
	cudaMalloc(&indexOutput, sizeof(int)*outputLen); gpuerr();
	gen_combination <<< inputBlocks, NUM_THREADS >>> (
	    input.ptr, combinationOffsets, distance,
	    deletionsOutput, indexOutput, input.len); gpuerr();

	// generate histogram
	int outputBlocks = divide_ceil(outputLen , NUM_THREADS);
	cudaMalloc(&histogramValue, sizeof(unsigned int)*outputLen);
	cudaMalloc(&histogramOutput, sizeof(int)*HISTOGRAM_SIZE);
	select_int3 <<< outputBlocks, NUM_THREADS>>>(
	    deletionsOutput, histogramValue, outputLen);
	cal_histogram(histogramValue, histogramOutput, HISTOGRAM_SIZE, UINT_MIN, UINT_MAX, outputLen);
	sort_key_values(deletionsOutput, indexOutput, outputLen);

	// boilerplate
	_cudaFree(combinationOffsets, histogramValue); gpuerr();
}

void stream_handler2(Chunk<Int3> &keyInOut, Chunk<int> &valueInOut, int* &histogramOutput,
                     int distance, int seqLen, int memoryConstraint, int* buffer) {
	int* inputOffsets, *valueLengths, *histogram, *indexes, *valueLengthsHost;

	printf("1\n");
	sort_key_values(keyInOut.ptr, valueInOut.ptr, keyInOut.len); gpuerr();
	printf("2\n");
	int offsetLen =
	    cal_offsets(keyInOut.ptr, inputOffsets, valueLengths, keyInOut.len, buffer); gpuerr();
	printf("3\n");

	int start = 0, nChunk;
	int* inputOffsetsPtr = inputOffsets, *valueLengthsPtr = valueLengths;
	int nBlock = divide_ceil(HISTOGRAM_SIZE , NUM_THREADS);
	valueLengthsHost = device_to_host(valueLengths, offsetLen);
	printf("4\n");

	//histogram loop
	while ((nChunk = solve_next_bin(valueLengthsHost, start, memoryConstraint, offsetLen)) > 0) {
		printf("5\n");
		int chunkLen = gen_smaller_index(valueInOut.ptr, inputOffsetsPtr, valueLengthsPtr, indexes, nChunk);
		printf("6\n");
		cal_histogram(indexes, histogram, HISTOGRAM_SIZE , 0, seqLen, chunkLen);
		printf("7\n");
		vector_add <<< nBlock, NUM_THREADS>>>(histogramOutput, histogram, HISTOGRAM_SIZE);
		printf("8\n");

		start += nChunk;
		inputOffsetsPtr += nChunk;
		valueLengthsPtr += nChunk;
		_cudaFree(indexes, histogram); gpuerr();
		printf("9\n");
	}
	_cudaFree(inputOffsets, valueLengths); gpuerr();
	cudaFreeHost(valueLengthsHost); gpuerr();
}

// void stream_handler3(Chunk<Int3> keyInput, Chunk<int> valueInput,
//                      Chunk<Int3> &keyOutput, Chunk<int> &valueOutput, Int2* &pairOutput,
//                      int* &histogramOutput, int lowerbound, int* buffer) {
// 	int* combinationValueOffsets, *pairOffsets;
// 	int offsetLen =
// 	    cal_offsets(keyInput.ptr, valueInput.ptr, combinationValueOffsets,
// 	                pairOffsets, keyInput.len, buffer);
// 	int pairLen =
// 	    gen_pairs(valueInput.ptr, combinationValueOffsets,
// 	              pairOffsets, pairOutput, offsetLen, buffer);

// 	// generate histogram
// 	// take lower bound into account

// 	gen_next_chunk(keyInput, valueInput, keyOutput, valueOutput,
// 	               combinationValueOffsets, offsetLen, lowerbound, buffer);
// 	_cudaFree(combinationValueOffsets, pairOffsets);
// }

void stream_handler4(Chunk<Int2> pairInput, XTNOutput &output, Int3* seq1,
                     int seq1Len, int distance, int* buffer) {
	Int2* pairOut;
	char* distanceOut;
	int outputLen =
	    postprocessing(seq1, pairInput.ptr, distance, pairOut, distanceOut,
	                   pairInput.len, buffer, seq1Len);

	make_output(pairOut, distanceOut, outputLen, output);
	_cudaFree(pairOut, distanceOut);
}