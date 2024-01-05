#include "generate_combination.cu"
#include "cub.cu"
#include "kernel.cu"
#include "codec.cu"
#include "stream.cu"
#include <limits.h>

/**
 * @file
 * @brief The core algorithm on the high-level abstraction
 * concerning data manupulation operations in all 4 stream.
 */

const int NUM_THREADS = 256;
const unsigned int UINT_MIN = 0;

//=====================================
// Private Functions
//=====================================

int NUM_BLOCK(int len) {
	return divide_ceil(len, NUM_THREADS);
}

int cal_offsets(Int3* inputKeys, int* &inputOffsets, int* &outputLengths, int n, int* buffer) {
	// cal inputOffsets
	cudaMalloc(&inputOffsets, sizeof(int)*n); gpuerr();
	unique_counts(inputKeys, inputOffsets, buffer, n); gpuerr();
	int nUnique = transfer_last_element(buffer, 1); gpuerr();

	// cal outputLengths
	cudaMalloc(&outputLengths, sizeof(int)*nUnique); gpuerr();
	cal_pair_len <<< NUM_BLOCK(nUnique), NUM_THREADS>>>(inputOffsets, outputLengths, nUnique); gpuerr();
	inclusive_sum(inputOffsets, nUnique); gpuerr();
	return nUnique;
}

int cal_offsets_lowerbound(Int3* inputKeys, int* inputValues, int* &inputOffsets,
                           int* &outputLengths, int lowerbound, int n, int* buffer) {
	// cal inputOffsets
	cudaMalloc(&inputOffsets, sizeof(int)*n); gpuerr();
	unique_counts(inputKeys, inputOffsets, buffer, n); gpuerr();
	int nUnique = transfer_last_element(buffer, 1); gpuerr();
	inclusive_sum(inputOffsets, nUnique); gpuerr();

	// cal outputLengths
	cudaMalloc(&outputLengths, sizeof(int)*nUnique); gpuerr();
	cal_pair_len_lowerbound <<< NUM_BLOCK(nUnique), NUM_THREADS>>>(
	    inputValues, inputOffsets, outputLengths, lowerbound, nUnique); gpuerr();
	return nUnique;
}

int gen_pairs(int* input, int* inputOffsets, int* outputLengths, Int2* &output,
              int* &lesserIndex, int lowerbound, int carry, int n) {
	int* outputOffsets;

	// cal outputOffsets
	cudaMalloc(&outputOffsets, n * sizeof(int)); gpuerr();
	inclusive_sum(outputLengths, outputOffsets, n); gpuerr();
	int outputLen = transfer_last_element(outputOffsets, n); gpuerr();

	//generate pairs
	cudaMalloc(&output, sizeof(Int2)*outputLen); gpuerr();
	cudaMalloc(&lesserIndex, sizeof(int)*outputLen); gpuerr();
	generate_pairs <<< NUM_BLOCK(n), NUM_THREADS>>>(input, output,
	        inputOffsets, outputOffsets, lesserIndex, lowerbound, carry, n); gpuerr();
	sort_int2(output, outputLen);

	print_gpu_memory();
	cudaFree(outputOffsets); gpuerr();
	return outputLen;
}

int gen_smaller_index(int* input, int* inputOffsets, int* outputLengths,
                      int* &output, int carry, int n) {
	int* outputOffsets;

	// cal outputOffsets
	cudaMalloc(&outputOffsets, n * sizeof(int)); gpuerr();
	inclusive_sum(outputLengths, outputOffsets, n); gpuerr();
	int outputLen = transfer_last_element(outputOffsets, n); gpuerr();

	//generate pairs
	cudaMalloc(&output, sizeof(int)*outputLen); gpuerr();
	generate_smaller_index <<< NUM_BLOCK(n), NUM_THREADS>>>(input, output,
	        inputOffsets, outputOffsets, carry, n); gpuerr();

	print_gpu_memory();
	cudaFree(outputOffsets); gpuerr();
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
	print_int2_arr(uniquePairs, 20);
	print_int2_arr(uniquePairs + uniqueLen - 20, 20);
	printf("uniqueLen %'d \n", uniqueLen);
	size_t byteRequirement = sizeof(char) * uniqueLen;
	cudaMalloc(&flags, byteRequirement); gpuerr();
	cudaMalloc(&uniqueDistances, byteRequirement); gpuerr();
	cudaMalloc(&distanceOutput, byteRequirement); gpuerr();
	cudaMalloc(&pairOutput, sizeof(Int2)*uniqueLen); gpuerr();
	cal_levenshtein <<< NUM_BLOCK(uniqueLen), NUM_THREADS>>>(
	    seq, uniquePairs, distance, uniqueDistances, flags, uniqueLen, seqLen); gpuerr();

	// filter levenshtein
	double_flag(uniquePairs, uniqueDistances, flags, pairOutput,
	            distanceOutput, buffer, uniqueLen); gpuerr();

	print_gpu_memory();
	_cudaFree(uniquePairs, uniqueDistances, flags); gpuerr();
	int outputLen = transfer_last_element(buffer, 1); gpuerr();
	return outputLen;
}

void make_output(Int2* pairOut, char* distanceOut, int len, XTNOutput &output) {
	output.indexPairs = device_to_host(pairOut, len); gpuerr();
	output.pairwiseDistances = device_to_host(distanceOut, len); gpuerr();
	output.len = len;
}

void gen_next_chunk(Chunk<Int3> &keyInOut, Chunk<int> &valueInOut,
                    int* valueOffsets, int offsetLen, int lowerbound, int* buffer) {
	char* flags;
	Int3* keyOut;
	int* valueOut;

	cudaMalloc(&flags, sizeof(char)*valueInOut.len); gpuerr();
	cudaMemset(flags, 1, sizeof(char)*valueInOut.len); gpuerr();
	cudaMalloc(&keyOut, sizeof(Int3)*valueInOut.len); gpuerr();
	cudaMalloc(&valueOut, sizeof(int)*valueInOut.len); gpuerr();

	flag_lowerbound <<< NUM_BLOCK(offsetLen), NUM_THREADS>>>(
	    valueInOut.ptr, valueOffsets, flags, lowerbound, offsetLen); gpuerr();
	double_flag(keyInOut.ptr, valueInOut.ptr, flags, keyOut, valueOut,
	            buffer, valueInOut.len); gpuerr();

	int outputLen = transfer_last_element(buffer, 1); gpuerr();
	print_gpu_memory();
	_cudaFree(flags); gpuerr();
	keyInOut.ptr = keyOut;
	keyInOut.len = outputLen;
	valueInOut.ptr = valueOut;
	valueInOut.len = outputLen;
}

int solve_next_bin(int* chunksizes, int start, int maxSize, int n) {
	int ans = 0, len = 0;
	for (int i = start; i < n; i++) {
		int currentChunkSize = chunksizes[i];
		if (len + currentChunkSize > maxSize)
			break;
		len += currentChunkSize;
		ans++;
	}
	return ans;
}

//=====================================
// Public Functions
//=====================================

int solve_bin_packing_lowerbounds(int* histograms, int* &lowerboundsOutput,
                                  int n, int seqLen, int* buffer, MemoryContext ctx) {
	int* rowIndex, *output, *key, *value;
	size_t* histogramIntermediate;

	int nLevel = ctx.histogramSize, len2d = n * nLevel;
	cudaMalloc(&rowIndex, sizeof(int) * len2d); gpuerr();
	cudaMalloc(&output, sizeof(int) * nLevel); gpuerr();
	cudaMalloc(&key, sizeof(int) * nLevel); gpuerr();
	cudaMalloc(&value, sizeof(int) * nLevel); gpuerr();
	cudaMalloc(&histogramIntermediate, sizeof(size_t) * len2d); gpuerr();

	make_row_index <<< NUM_BLOCK(n), NUM_THREADS>>>(rowIndex, n, nLevel); gpuerr();
	inclusive_sum_by_key(rowIndex, histograms, histogramIntermediate, len2d); gpuerr();
	gen_bounds <<< NUM_BLOCK(nLevel), NUM_THREADS >>>(
	    histogramIntermediate, key, value, 10 /*ctx.maxThroughputExponent*/, seqLen, n, nLevel); gpuerr();
	print_int_arr(key, nLevel);
	max_by_key(key, value, output, buffer, nLevel); gpuerr();

	int outputLen = transfer_last_element(buffer, 1); gpuerr();
	lowerboundsOutput = device_to_host(output, outputLen); gpuerr();

	print_gpu_memory();
	_cudaFree(rowIndex, output, key, value, histogramIntermediate); gpuerr();
	return outputLen;
}

int solve_bin_packing_offsets(int* histograms, int** &offsetOutput,
                              int n, int* buffer, MemoryContext ctx) {
	int* rowIndex, *assignment, *output1d;
	int offsetLen;

	int nLevel = ctx.histogramSize, len2d = n * nLevel;
	cudaMalloc(&rowIndex, sizeof(int) * len2d); gpuerr();
	cudaMalloc(&assignment, sizeof(int) * len2d); gpuerr();
	cudaMalloc(&output1d, sizeof(int) * len2d); gpuerr();
	cudaMallocHost(&offsetOutput, sizeof(int*) * n); gpuerr();

	//solve bin packing
	make_row_index <<< NUM_BLOCK(n), NUM_THREADS>>>(rowIndex, n, nLevel); gpuerr();
	inclusive_sum_by_key(rowIndex, histograms, len2d); gpuerr();
	gen_assignment <<< NUM_BLOCK(nLevel), NUM_THREADS >>>(
	    histograms, assignment, ctx.maxThroughputExponent, n, nLevel); gpuerr();
	max_by_key(assignment, histograms, output1d, buffer, len2d); gpuerr();

	//make output
	int outputLen = transfer_last_element(buffer, 1); gpuerr();

	if (outputLen % n == 0) {
		offsetLen = outputLen / n;
		int* output1dPtr = output1d;
		for (int i = 0; i < n; i++) {
			offsetOutput[i] = device_to_host(output1dPtr, offsetLen); gpuerr();
			output1dPtr += offsetLen;
		}
	} else if (outputLen == 1) {
		for (int i = 0; i < n; i++)
			offsetOutput[i] = device_to_host( histograms + (i * nLevel) + nLevel - 1, 1); gpuerr();
		offsetLen = 1;
	} else
		print_err("bin_packing outputLen is not divisible by inputLen");

	print_gpu_memory();
	_cudaFree(rowIndex, assignment, output1d); gpuerr();
	return offsetLen;
}

void stream_handler1(Chunk<Int3> input, Int3* &deletionsOutput, int* &indexOutput,
                     std::vector<int*> &histogramOutput, int &outputLen, int distance, MemoryContext ctx) {
	int *combinationOffsets;
	unsigned int *histogramValue;
	int* histogram;

	// cal combinationOffsets
	cudaMalloc(&combinationOffsets, sizeof(int)*input.len);	gpuerr();
	cal_combination_len <<< NUM_BLOCK(input.len), NUM_THREADS >>>(
	    input.ptr, distance, combinationOffsets, input.len); gpuerr();
	inclusive_sum(combinationOffsets, input.len); gpuerr();
	outputLen = transfer_last_element(combinationOffsets, input.len); gpuerr();

	// generate combinations
	cudaMalloc(&deletionsOutput, sizeof(Int3)*outputLen); gpuerr();
	cudaMalloc(&indexOutput, sizeof(int)*outputLen); gpuerr();
	cudaMalloc(&histogramValue, sizeof(unsigned int)*outputLen); gpuerr();
	gen_combination <<< NUM_BLOCK(input.len), NUM_THREADS >>> (
	    input.ptr, combinationOffsets, distance,
	    deletionsOutput, indexOutput, histogramValue, input.len); gpuerr();

	// generate histogram
	sort_key_values(deletionsOutput, indexOutput, outputLen); gpuerr();
	cudaMalloc(&histogram, sizeof(int)*ctx.histogramSize);	gpuerr();
	cal_histogram(histogramValue, histogram, ctx.histogramSize, UINT_MIN, UINT_MAX, outputLen); gpuerr();
	histogramOutput.push_back(histogram);

	printf("total allocation 1 %'lu\n",
	       sizeof(int)*input.len + sizeof(Int3)*outputLen + sizeof(int)*outputLen + sizeof(unsigned int)*outputLen);
	print_gpu_memory();
	_cudaFree(combinationOffsets, histogramValue); gpuerr();
}

void stream_handler2(Chunk<Int3> &keyInOut, Chunk<int> &valueInOut, std::vector<int*> &histogramOutput,
                     int distance, int seqLen, int* buffer, MemoryContext ctx) {
	int* inputOffsets, *valueLengths, *indexes, *valueLengthsHost, *histogram;

	sort_key_values(keyInOut.ptr, valueInOut.ptr, keyInOut.len); gpuerr();
	int offsetLen =
	    cal_offsets(keyInOut.ptr, inputOffsets, valueLengths, keyInOut.len, buffer);

	int start = 0, carry = 0, nChunk;
	int* inputOffsetsPtr = inputOffsets, *valueLengthsPtr = valueLengths;
	valueLengthsHost = device_to_host(valueLengths, offsetLen); gpuerr();

	size_t nChunkSum = 0;
	//histogram loop
	while ((nChunk = solve_next_bin(valueLengthsHost, start, ctx.bandwidth2, offsetLen)) > 0) {

		int chunkLen = gen_smaller_index(
		                   valueInOut.ptr, inputOffsetsPtr, valueLengthsPtr, indexes, carry, nChunk);
		print_bandwidth(chunkLen, ctx.bandwidth2, "2b");
		cudaMalloc(&histogram, sizeof(int)*ctx.histogramSize);	gpuerr();
		cal_histogram(indexes, histogram, ctx.histogramSize , 0, seqLen, chunkLen); gpuerr();
		histogramOutput.push_back(histogram);

		carry = transfer_last_element(inputOffsetsPtr, nChunk);
		start += nChunk;
		inputOffsetsPtr += nChunk;
		valueLengthsPtr += nChunk;
		nChunkSum += nChunk;
		print_gpu_memory();
		cudaFree(indexes); gpuerr();
	}

	_cudaFree(inputOffsets, valueLengths); gpuerr();
	cudaFreeHost(valueLengthsHost); gpuerr();
}

void stream_handler3(Chunk<Int3> &keyInOut, Chunk<int> &valueInOut, void callback(Int2*, int),
                     std::vector<int*> &histogramOutput, int lowerbound, int seqLen, int* buffer, MemoryContext ctx) {
	int* inputOffsets, *valueLengths, *valueLengthsHost, *lesserIndex, *histogram;
	Int2* pairOutput;

	int offsetLen = cal_offsets_lowerbound(
	                    keyInOut.ptr, valueInOut.ptr, inputOffsets,
	                    valueLengths, lowerbound, keyInOut.len, buffer);

	int start = 0, carry = 0, nChunk;
	int* inputOffsetsPtr = inputOffsets, *valueLengthsPtr = valueLengths;
	valueLengthsHost = device_to_host(valueLengths, offsetLen); gpuerr();

	// generate pairs
	while ((nChunk = solve_next_bin(valueLengthsHost, start, ctx.bandwidth2, offsetLen)) > 0) {
		int chunkLen = gen_pairs(valueInOut.ptr, inputOffsetsPtr, valueLengthsPtr,
		                         pairOutput, lesserIndex, lowerbound, carry, nChunk);
		print_bandwidth(chunkLen, ctx.bandwidth2, "3b");
		callback(pairOutput, chunkLen);
		cudaMalloc(&histogram, sizeof(int)*ctx.histogramSize);	gpuerr();
		cal_histogram(lesserIndex, histogram, ctx.histogramSize , 0, seqLen, chunkLen); gpuerr();
		histogramOutput.push_back(histogram);

		carry = transfer_last_element(inputOffsetsPtr, nChunk);
		start += nChunk;
		inputOffsetsPtr += nChunk;
		valueLengthsPtr += nChunk;

		print_gpu_memory();
		_cudaFree(pairOutput, lesserIndex); gpuerr();
	}

	gen_next_chunk(keyInOut, valueInOut, inputOffsets, offsetLen, lowerbound, buffer);

	_cudaFree(inputOffsets, valueLengths); gpuerr();
	cudaFreeHost(valueLengthsHost); gpuerr();
}

void stream_handler4(Chunk<Int2> pairInput, XTNOutput & output, Int3 * seq1,
                     int seq1Len, int distance, int* buffer) {
	Int2* pairOut;
	char* distanceOut;
	int outputLen =
	    postprocessing(seq1, pairInput.ptr, distance, pairOut, distanceOut,
	                   pairInput.len, buffer, seq1Len);

	make_output(pairOut, distanceOut, outputLen, output);
	_cudaFree(pairOut, distanceOut); gpuerr();
}