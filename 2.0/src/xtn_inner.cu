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
              int* &lesserIndex, int lowerbound, int n) {
	int* outputOffsets;

	// cal outputOffsets
	cudaMalloc(&outputOffsets, n * sizeof(int)); gpuerr();
	inclusive_sum(outputLengths, outputOffsets, n); gpuerr();
	int outputLen = transfer_last_element(outputOffsets, n); gpuerr();

	//generate pairs
	cudaMalloc(&output, sizeof(Int2)*outputLen); gpuerr();
	cudaMalloc(&lesserIndex, sizeof(int)*outputLen); gpuerr();
	generate_pairs <<< NUM_BLOCK(n), NUM_THREADS>>>(input, output,
	        inputOffsets, outputOffsets, lesserIndex, lowerbound, n); gpuerr();

	cudaFree(outputOffsets); gpuerr();
	return outputLen;
}

int gen_smaller_index(int* input, int* inputOffsets, int* outputLengths, int* &output, int n) {
	int* outputOffsets;

	// cal outputOffsets
	cudaMalloc(&outputOffsets, n * sizeof(int)); gpuerr();
	inclusive_sum(outputLengths, outputOffsets, n); gpuerr();
	int outputLen = transfer_last_element(outputOffsets, n); gpuerr();

	//generate pairs
	cudaMalloc(&output, sizeof(int)*outputLen); gpuerr();
	generate_smaller_index <<< NUM_BLOCK(n), NUM_THREADS>>>(input, output,
	        inputOffsets, outputOffsets, n); gpuerr();

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
	int byteRequirement = sizeof(char) * uniqueLen;
	cudaMalloc(&flags, byteRequirement); gpuerr();
	cudaMalloc(&uniqueDistances, byteRequirement); gpuerr();
	cudaMalloc(&distanceOutput, byteRequirement); gpuerr();
	cudaMalloc(&pairOutput, sizeof(Int2)*uniqueLen); gpuerr();
	cal_levenshtein <<< NUM_BLOCK(uniqueLen), NUM_THREADS>>>(
	    seq, uniquePairs, distance, uniqueDistances, flags, uniqueLen, seqLen); gpuerr();

	// filter levenshtein
	double_flag(uniquePairs, uniqueDistances, flags, pairOutput,
	            distanceOutput, buffer, uniqueLen); gpuerr();
	_cudaFree(uniquePairs, uniqueDistances, flags); gpuerr();
	int outputLen = transfer_last_element(buffer, 1); gpuerr();
	return outputLen;
}

void make_output(Int2* pairOut, char* distanceOut, size_t len, XTNOutput &output) {
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
	_cudaFree(flags, keyInOut.ptr, valueInOut.ptr); gpuerr();
	keyInOut.ptr = keyOut;
	keyInOut.len = outputLen;
	valueInOut.ptr = valueOut;
	valueInOut.len = outputLen;
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
                      int n, int nLevel, int* buffer, MemoryContext ctx) {
	int* rowIndex, *assignment, *output1d, *output1dPtr;

	int len2d = n * nLevel;
	cudaMalloc(&rowIndex, sizeof(int) * len2d); gpuerr();
	cudaMalloc(&assignment, sizeof(int) * len2d); gpuerr();
	cudaMalloc(&output1d, sizeof(int) * len2d); gpuerr();
	cudaMallocHost(&offsetOutput, sizeof(int*) * n); gpuerr();

	//solve bin packing
	make_row_index <<< NUM_BLOCK(n), NUM_THREADS>>>(rowIndex, n, nLevel);
	inclusive_sum_by_key(rowIndex, histograms, len2d); gpuerr();
	gen_assignment <<< NUM_BLOCK(nLevel), NUM_THREADS >>>(
	    histograms, assignment, ctx.maxThroughputExponent, n, nLevel); gpuerr();
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
                     int* &histogramOutput, int &outputLen, int distance, MemoryContext ctx) {
	int *combinationOffsets;
	unsigned int *histogramValue;

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
	cudaMalloc(&histogramOutput, sizeof(int)*ctx.histogramSize); gpuerr();
	cal_histogram(histogramValue, histogramOutput, ctx.histogramSize, UINT_MIN, UINT_MAX, outputLen); gpuerr();
	sort_key_values(deletionsOutput, indexOutput, outputLen); gpuerr();

	_cudaFree(combinationOffsets, histogramValue); gpuerr();
}

void stream_handler2(Chunk<Int3> &keyInOut, Chunk<int> &valueInOut, int* &histogramOutput,
                     int distance, int seqLen, int* buffer, MemoryContext ctx) {
	int* inputOffsets, *valueLengths, *histogram, *indexes, *valueLengthsHost;

	sort_key_values(keyInOut.ptr, valueInOut.ptr, keyInOut.len); gpuerr();
	int offsetLen =
	    cal_offsets(keyInOut.ptr, inputOffsets, valueLengths, keyInOut.len, buffer);

	int start = 0, nChunk;
	int* inputOffsetsPtr = inputOffsets, *valueLengthsPtr = valueLengths;
	valueLengthsHost = device_to_host(valueLengths, offsetLen); gpuerr();
	int nBlock = divide_ceil(ctx.histogramSize, NUM_THREADS);
	cudaMalloc(&histogram, sizeof(int)*ctx.histogramSize); gpuerr();

	//histogram loop
	while ((nChunk = solve_next_bin(valueLengthsHost, start, ctx.maxThroughput, offsetLen)) > 0) {
		int chunkLen = gen_smaller_index(valueInOut.ptr, inputOffsetsPtr, valueLengthsPtr, indexes, nChunk);
		cal_histogram(indexes, histogram, ctx.histogramSize , 0, seqLen, chunkLen); gpuerr();
		vector_add <<< nBlock, NUM_THREADS>>>(histogramOutput, histogram, ctx.histogramSize); gpuerr();

		start += nChunk;
		inputOffsetsPtr += nChunk;
		valueLengthsPtr += nChunk;
		cudaFree(indexes); gpuerr();
	}

	_cudaFree(inputOffsets, valueLengths, histogram); gpuerr();
	cudaFreeHost(valueLengthsHost); gpuerr();
}

void stream_handler3(Chunk<Int3> &keyInOut, Chunk<int> &valueInOut, D2Stream<Int2> &pairOut,
                     int* &histogramOutput, int lowerbound, int seqLen, int* buffer, MemoryContext ctx) {
	int* inputOffsets, *valueLengths, *valueLengthsHost, *histogram, *lesserIndex;
	Int2* pairOutput;

	int offsetLen = cal_offsets_lowerbound(
	                    keyInOut.ptr, valueInOut.ptr, inputOffsets,
	                    valueLengths, lowerbound, keyInOut.len, buffer);

	int start = 0, nChunk;
	int* inputOffsetsPtr = inputOffsets, *valueLengthsPtr = valueLengths;
	valueLengthsHost = device_to_host(valueLengths, offsetLen); gpuerr();
	int nBlock = divide_ceil(ctx.histogramSize, NUM_THREADS);
	cudaMalloc(&histogram, sizeof(int)*ctx.histogramSize); gpuerr();

	// generate pairs
	while ((nChunk = solve_next_bin(valueLengthsHost, start, ctx.maxThroughput, offsetLen)) > 0) {
		int chunkLen = gen_pairs(valueInOut.ptr, inputOffsetsPtr, valueLengthsPtr,
		                         pairOutput, lesserIndex, lowerbound, nChunk);
		pairOut.write(pairOutput, chunkLen);
		cal_histogram(lesserIndex, histogram, ctx.histogramSize , 0, seqLen, chunkLen); gpuerr();
		vector_add <<< nBlock, NUM_THREADS>>>(histogramOutput, histogram, ctx.histogramSize); gpuerr();

		start += nChunk;
		inputOffsetsPtr += nChunk;
		valueLengthsPtr += nChunk;
		_cudaFree(pairOutput, lesserIndex); gpuerr();
	}

	gen_next_chunk(keyInOut, valueInOut, inputOffsets, offsetLen, lowerbound, buffer);

	_cudaFree(inputOffsets, valueLengths, histogram); gpuerr();
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