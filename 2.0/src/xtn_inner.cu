#include "generate_combination.cu"
#include "cub.cu"
#include "kernel.cu"
#include "codec.cu"
#include "stream.cu"
#include <limits.h>

/**
 * @file
 * The core algorithm on the high-level abstraction
 * concerning data manupulation operations in all 4 stream.
 */

const int NUM_THREADS = 256;
const unsigned int UINT_MIN = 0;

//=====================================
// Private Functions
//=====================================

/**
 * private function.
*/
int NUM_BLOCK(int len) {
	int ans = divide_ceil(len, NUM_THREADS);
	if (ans == 0) return 1;
	return ans;
}

/**
 * private function.
*/
int cal_offsets(Int3* inputKeys, int* &inputOffsets, int* &outputLengths, int n, int* buffer) {
	// cal inputOffsets
	cudaMalloc(&inputOffsets, sizeof(int)*n); gpuerr();
	unique_counts(inputKeys, inputOffsets, buffer, n);
	int nUnique = transfer_last_element(buffer, 1);

	// cal outputLengths
	cudaMalloc(&outputLengths, sizeof(int)*nUnique); gpuerr();
	cal_pair_len <<< NUM_BLOCK(nUnique), NUM_THREADS>>>(inputOffsets, outputLengths, nUnique); gpuerr();
	inclusive_sum(inputOffsets, nUnique);
	return nUnique;
}

/**
 * private function.
*/
int cal_offsets_lowerbound(Int3* inputKeys, int* inputValues, int* &inputOffsets,
                           int* &outputLengths, int lowerbound, int n, int* buffer) {
	// cal inputOffsets
	cudaMalloc(&inputOffsets, sizeof(int)*n); gpuerr();
	unique_counts(inputKeys, inputOffsets, buffer, n);
	int nUnique = transfer_last_element(buffer, 1);
	inclusive_sum(inputOffsets, nUnique);

	// cal outputLengths
	cudaMalloc(&outputLengths, sizeof(int)*nUnique); gpuerr();
	cal_pair_len_lowerbound <<< NUM_BLOCK(nUnique), NUM_THREADS>>>(
	    inputValues, inputOffsets, outputLengths, lowerbound, nUnique); gpuerr();
	return nUnique;
}

/**
 * private function.
*/
int gen_pairs(int* input, int* inputOffsets, int* outputLengths, Int2* &output,
              int* &lesserIndex, int lowerbound, int carry, int n, int seqLen) {
	int* outputOffsets;

	// cal outputOffsets
	cudaMalloc(&outputOffsets, n * sizeof(int)); gpuerr();
	inclusive_sum(outputLengths, outputOffsets, n);
	int outputLen = transfer_last_element(outputOffsets, n);

	//generate pairs
	_cudaMalloc(output, lesserIndex, outputLen);
	generate_pairs <<< NUM_BLOCK(n), NUM_THREADS>>>(input, output,
	        inputOffsets, outputOffsets, lesserIndex, lowerbound, carry, n); gpuerr();
	sort_int2(output, outputLen);

	cudaFree(outputOffsets); gpuerr();
	return outputLen;
}

/**
 * private function.
*/
int gen_smaller_index(int* input, int* inputOffsets, int* outputLengths,
                      int* &output, int carry, int n) {
	int* outputOffsets;

	// cal outputOffsets
	cudaMalloc(&outputOffsets, n * sizeof(int)); gpuerr();
	inclusive_sum(outputLengths, outputOffsets, n);
	int outputLen = transfer_last_element(outputOffsets, n);

	//generate pairs
	cudaMalloc(&output, sizeof(int)*outputLen); gpuerr();
	generate_smaller_index <<< NUM_BLOCK(n), NUM_THREADS>>>(input, output,
	        inputOffsets, outputOffsets, carry, n); gpuerr();

	cudaFree(outputOffsets); gpuerr();
	return outputLen;
}

/**
 * private function.
*/
int deduplicate(Int2* input, Int2* output, int n, int* buffer) {
	sort_int2(input, n);
	unique(input, output, buffer, n);
	return transfer_last_element(buffer, 1);
}

/**
 * private function.
*/
int postprocessing(Int3* seq, Int2* uniquePairs, int distance, char measure,
                   Int2* &pairOutput, char* &distanceOutput,
                   int uniqueLen, int* buffer, int seqLen) {
	char* uniqueDistances, *flags;

	// cal levenshtein
	_cudaMalloc(flags, uniqueDistances, uniqueLen);
	cal_distance <<< NUM_BLOCK(uniqueLen), NUM_THREADS>>>(
	    seq, uniquePairs, distance, measure, uniqueDistances,
	    flags, uniqueLen, seqLen); gpuerr();

	// filter levenshtein
	_cudaMalloc(pairOutput, distanceOutput, uniqueLen);
	flag(uniquePairs, flags, pairOutput, buffer, uniqueLen);
	flag(uniqueDistances, flags, distanceOutput, buffer, uniqueLen);

	_cudaFree(uniqueDistances, flags);
	return transfer_last_element(buffer, 1);
}

/**
 * private function.
*/
void make_output(Int2* pairOut, char* distanceOut, int len, XTNOutput &output) {
	output.indexPairs = device_to_host(pairOut, len);
	output.pairwiseDistances = device_to_host(distanceOut, len);
	output.len = len;
}

/**
 * private function.
*/
void gen_next_chunk(Chunk<Int3> &keyInOut, Chunk<int> &valueInOut,
                    int* valueOffsets, int offsetLen, int lowerbound, int* buffer) {
	char* flags;
	Int3* keyOut;
	int* valueOut;

	_cudaMalloc(flags, keyOut, valueOut, valueInOut.len);
	cudaMemset(flags, 1, sizeof(char)*valueInOut.len); gpuerr();

	flag_lowerbound <<< NUM_BLOCK(offsetLen), NUM_THREADS>>>(
	    valueInOut.ptr, valueOffsets, flags, lowerbound, offsetLen); gpuerr();
	flag(keyInOut.ptr, flags, keyOut, buffer, keyInOut.len);
	flag(valueInOut.ptr, flags, valueOut, buffer, valueInOut.len);

	int outputLen = transfer_last_element(buffer, 1);
	_cudaFree(flags);
	keyInOut.ptr = keyOut;
	keyInOut.len = outputLen;
	valueInOut.ptr = valueOut;
	valueInOut.len = outputLen;
}

//=====================================
// Public Functions
//=====================================

/**
 * solve bin packing on spot when precalculation is impossible.
 *
 * @param chunksizes array of chunk sizes
 * @param start offset of the latest processed chunk
 * @param maxSize bin size
 * @param n array length of chunkSizes
*/
int solve_next_bin(int* chunksizes, int start, int maxSize, int n) {
	int ans = 0;
	size_t len = 0;
	int currentChunkSize = -1;
	for (int i = start; i < n; i++) {
		currentChunkSize = chunksizes[i];
		if (currentChunkSize < 0)
			printf("solve_next_bin negative chunk size: %d\n", currentChunkSize);
		if (len + currentChunkSize > maxSize)
			break;
		len += currentChunkSize;
		ans++;
	}
	if ((ans == 0) && (start != n)) {
		printf("solve_next_bin exceeding maxSize: %d / %d\n", chunksizes[start], maxSize);
		ans = 1;
	}
	return ans;
}

/**
 * solve bin packing for lower bound calculation.
 *
 * @param histograms flatten histogram matrix
 * @param lowerboundsOutput output assignment
 * @param n row count of histograms
 * @param seqLen array length of sequences
 * @param buffer reusable 4-byte buffer
 * @param ctx memory constraints and info
*/
int solve_bin_packing_lowerbounds(int* histograms, int* &lowerboundsOutput,
                                  int n, int seqLen, int* buffer, MemoryContext ctx) {
	int* rowIndex, *output, *key, *value;
	size_t* histogramIntermediate;

	int nLevel = ctx.histogramSize, len2d = n * nLevel;
	_cudaMalloc(output, key, value, nLevel);
	_cudaMalloc(rowIndex, histogramIntermediate, len2d);

	make_row_index <<< NUM_BLOCK(n), NUM_THREADS>>>(rowIndex, n, nLevel); gpuerr();
	toSizeT <<< NUM_BLOCK(len2d), NUM_THREADS>>>(histograms, histogramIntermediate, len2d); gpuerr();
	inclusive_sum_by_key(rowIndex, histogramIntermediate, len2d);
	gen_bounds <<< NUM_BLOCK(nLevel), NUM_THREADS >>>(
	    histogramIntermediate, key, value, ctx.maxThroughputExponent, seqLen, n, nLevel); gpuerr();
	max_by_key(key, value, output, buffer, nLevel);

	int outputLen = transfer_last_element(buffer, 1);
	lowerboundsOutput = device_to_host(output, outputLen);

	_cudaFree(rowIndex, output, key, value, histogramIntermediate);
	return outputLen;
}

/**
 * solve bin packing for 2 dimensional buffer.
 *
 * @param histograms flatten histogram matrix
 * @param offsetOutput output assignment
 * @param n row count of histograms
 * @param buffer reusable 4-byte buffer
 * @param ctx memory constraints and info
*/
int solve_bin_packing_offsets(int* histograms, int** &offsetOutput,
                              int n, int* buffer, MemoryContext ctx) {
	int* rowIndex, *assignment, *output1d;
	int offsetLen;

	int nLevel = ctx.histogramSize, len2d = n * nLevel;
	_cudaMalloc(rowIndex, assignment, output1d, len2d);
	cudaMallocHost(&offsetOutput, sizeof(int*) * n); gpuerr();

	//solve bin packing
	make_row_index <<< NUM_BLOCK(n), NUM_THREADS>>>(rowIndex, n, nLevel); gpuerr();
	inclusive_sum_by_key(rowIndex, histograms, len2d);
	gen_assignment <<< NUM_BLOCK(nLevel), NUM_THREADS >>>(
	    histograms, assignment, ctx.maxThroughputExponent, n, nLevel); gpuerr();
	max_by_key(assignment, histograms, output1d, buffer, len2d);

	//make output
	int outputLen = transfer_last_element(buffer, 1);

	if (outputLen % n == 0) {
		offsetLen = outputLen / n;
		int* output1dPtr = output1d;
		for (int i = 0; i < n; i++) {
			offsetOutput[i] = device_to_host(output1dPtr, offsetLen);
			output1dPtr += offsetLen;
		}
	} else if (outputLen == 1) {
		for (int i = 0; i < n; i++)
			offsetOutput[i] = device_to_host( histograms + (i * nLevel) + nLevel - 1, 1);
		offsetLen = 1;
	} else
		print_err("bin_packing outputLen is not divisible by inputLen");

	_cudaFree(rowIndex, assignment, output1d);
	return offsetLen;
}

/**
 * handle all GPU operations in stream 1.
 *
 * @param input input CDR3 sequences
 * @param deletionsOutput generated deletion forms of CDR3 sequence
 * @param indexOutput original index of each deletion forms
 * @param histogramOutput histogram of deletion forms
 * @param outputLen number of output deletion forms
 * @param distance distance threshold
 * @param carry index offset of the chunk
 * @param ctx memory context
*/
void stream_handler1(Chunk<Int3> input, Int3* &deletionsOutput, int* &indexOutput,
                     std::vector<int*> &histogramOutput, int &outputLen,
                     int distance, int &carry, MemoryContext ctx) {
	int *combinationOffsets;
	unsigned int *histogramValue;
	int* histogram;

	// cal combinationOffsets
	cudaMalloc(&combinationOffsets, sizeof(int)*input.len);	gpuerr();
	cal_combination_len <<< NUM_BLOCK(input.len), NUM_THREADS >>>(
	    input.ptr, distance, combinationOffsets, input.len); gpuerr();
	inclusive_sum(combinationOffsets, input.len);
	outputLen = transfer_last_element(combinationOffsets, input.len);

	// generate combinations
	_cudaMalloc(deletionsOutput, indexOutput, histogramValue, outputLen);
	gen_combination <<< NUM_BLOCK(input.len), NUM_THREADS >>> (
	    input.ptr, combinationOffsets, distance, deletionsOutput,
	    indexOutput, carry, histogramValue, input.len); gpuerr();
	carry += input.len;
	cudaFree(combinationOffsets); gpuerr();

	// generate histogram
	sort_key_values(deletionsOutput, indexOutput, outputLen);
	cudaMalloc(&histogram, sizeof(int)*ctx.histogramSize); gpuerr();
	cal_histogram(histogramValue, histogram, ctx.histogramSize, UINT_MIN, UINT_MAX, outputLen);
	histogramOutput.push_back(histogram);

	cudaFree(histogramValue); gpuerr();
}

/**
 * handle all GPU operations in stream 2.
 *
 * @param keyInOut deletion forms of CDR3 sequence
 * @param valueInOut index of each deletion forms
 * @param histogramOutput histogram of indexes
 * @param throughput2B throughput counting variable
 * @param distance distance threshold
 * @param seqLen number of input CDR3 sequences
 * @param buffer integer buffer
 * @param ctx memory context
*/
void stream_handler2(Chunk<Int3> &keyInOut, Chunk<int> &valueInOut, std::vector<int*> &histogramOutput,
                     size_t &throughput2B, int distance, int seqLen, int* buffer, MemoryContext ctx) {
	int* inputOffsets, *valueLengths, *indexes, *valueLengthsHost, *histogram;

	sort_key_values(keyInOut.ptr, valueInOut.ptr, keyInOut.len);
	int offsetLen =
	    cal_offsets(keyInOut.ptr, inputOffsets, valueLengths, keyInOut.len, buffer);

	int start = 0, carry = 0, nChunk;
	int* inputOffsetsPtr = inputOffsets, *valueLengthsPtr = valueLengths;
	valueLengthsHost = device_to_host(valueLengths, offsetLen);

	//histogram loop
	while ((nChunk = solve_next_bin(valueLengthsHost, start, ctx.bandwidth2, offsetLen)) > 0) {

		int chunkLen = gen_smaller_index(
		                   valueInOut.ptr, inputOffsetsPtr, valueLengthsPtr, indexes, carry, nChunk);

		throughput2B += chunkLen;
		print_bandwidth(chunkLen, ctx.bandwidth2, "2b");
		cudaMalloc(&histogram, sizeof(int)*ctx.histogramSize);	gpuerr();
		cal_histogram(indexes, histogram, ctx.histogramSize , 0, seqLen, chunkLen);
		histogramOutput.push_back(histogram);

		carry = transfer_last_element(inputOffsetsPtr, nChunk);
		start += nChunk;
		inputOffsetsPtr += nChunk;
		valueLengthsPtr += nChunk;
		cudaFree(indexes); gpuerr();
	}

	_cudaFree(inputOffsets, valueLengths);
	cudaFreeHost(valueLengthsHost); gpuerr();
}

/**
 * handle all GPU operations in stream 3.
 *
 * @param keyInOut deletion forms of CDR3 sequence
 * @param valueInOut index of each deletion forms
 * @param callback pair outputs
 * @param histogramOutput histogram of generating pairs
 * @param lowerbound bound of generating pairs
 * @param seqLen number of input CDR3 sequences
 * @param buffer integer buffer
 * @param ctx memory context
*/
void stream_handler3(Chunk<Int3> &keyInOut, Chunk<int> &valueInOut, void callback(Int2*, int),
                     std::vector<int*> &histogramOutput, int lowerbound, int seqLen,
                     int* buffer, MemoryContext ctx) {
	int* inputOffsets, *valueLengths, *valueLengthsHost, *lesserIndex, *histogram;
	Int2* pairOutput;

	int offsetLen = cal_offsets_lowerbound(
	                    keyInOut.ptr, valueInOut.ptr, inputOffsets,
	                    valueLengths, lowerbound, keyInOut.len, buffer);

	int start = 0, carry = 0, nChunk;
	int* inputOffsetsPtr = inputOffsets, *valueLengthsPtr = valueLengths;
	valueLengthsHost = device_to_host(valueLengths, offsetLen);

	// generate pairs
	while ((nChunk = solve_next_bin(valueLengthsHost, start, ctx.bandwidth2, offsetLen)) > 0) {
		int chunkLen = gen_pairs(valueInOut.ptr, inputOffsetsPtr, valueLengthsPtr,
		                         pairOutput, lesserIndex, lowerbound, carry, nChunk, seqLen);
		print_bandwidth(chunkLen, ctx.bandwidth2, "3b");
		callback(pairOutput, chunkLen);
		cudaMalloc(&histogram, sizeof(int)*ctx.histogramSize);	gpuerr();
		cal_histogram(lesserIndex, histogram, ctx.histogramSize , 0, seqLen, chunkLen);
		histogramOutput.push_back(histogram);

		carry = transfer_last_element(inputOffsetsPtr, nChunk);
		start += nChunk;
		inputOffsetsPtr += nChunk;
		valueLengthsPtr += nChunk;

		_cudaFree(pairOutput, lesserIndex);
	}

	gen_next_chunk(keyInOut, valueInOut, inputOffsets, offsetLen, lowerbound, buffer);

	_cudaFree(inputOffsets, valueLengths);
	cudaFreeHost(valueLengthsHost); gpuerr();
}

/**
 * handle all GPU operations in stream 4 nn mode.
 *
 * @param pairInput nearest neighbor pairs
 * @param output returning output
 * @param seq input CDR3 sequences
 * @param seqLen number of input CDR3 sequences
 * @param distance distance threshold
 * @param measure type of measurement (levenshtein/hamming)
 * @param buffer integer buffer
*/
void stream_handler4_nn(Chunk<Int2> pairInput, XTNOutput &output, Int3* seq,
                        int seqLen, int distance, char measure, int* buffer) {
	Int2* pairOut, *uniquePairs;
	char* distanceOut;

	cudaMalloc(&uniquePairs, sizeof(Int2)*pairInput.len); gpuerr();
	int uniqueLen = deduplicate(pairInput.ptr, uniquePairs, pairInput.len, buffer);
	int outputLen =
	    postprocessing(seq, uniquePairs, distance, measure, pairOut, distanceOut,
	                   uniqueLen, buffer, seqLen);
	make_output(pairOut, distanceOut, outputLen, output);
	_cudaFree(uniquePairs, pairOut, distanceOut);
}

/**
 * handle all GPU operations in stream 4 overlap mode.
 *
 * @param pairInput nearest neighbor pairs
 * @param output returning output
 * @param seq input CDR3 sequences
 * @param seqInfo information of each sequence
 * @param seqOffset offset of seqInfo array
 * @param seqLen number of input CDR3 sequences
 * @param distance distance threshold
 * @param measure type of measurement (levenshtein/hamming)
 * @param buffer integer buffer
*/
void stream_handler4_overlap(Chunk<Int2> pairInput, std::vector<XTNOutput> &allOutputs, Int3* seq,
                             SeqInfo* seqInfo, int* seqOffset, int seqLen, int distance,
                             char measure, int* buffer, MemoryContext ctx) {
	Int2* pairOut, *pairOut2, *uniquePairs, *pairOut3;
	size_t* freqOut, *freqOut2;
	int* valueLengths, *valueLengthsHost;
	char* flags;

	// find pairOut3
	cudaMalloc(&uniquePairs, sizeof(Int2)*pairInput.len); gpuerr();
	int uniqueLen = deduplicate(pairInput.ptr, uniquePairs, pairInput.len, buffer);
	_cudaMalloc(flags, pairOut3, uniqueLen);
	cal_distance <<< NUM_BLOCK(uniqueLen), NUM_THREADS>>>(
	    seq, uniquePairs, distance, measure, NULL,
	    flags, uniqueLen, seqLen); gpuerr();
	flag(uniquePairs, flags, pairOut3, buffer, uniqueLen);
	_cudaFree(uniquePairs, flags);
	uniqueLen = transfer_last_element(buffer, 1);

	// cal value Lengths
	cudaMalloc(&valueLengths, sizeof(int)*uniqueLen); gpuerr();
	cal_pair_len_nondiag <<< NUM_BLOCK(uniqueLen), NUM_THREADS>>>(
	    pairOut3, seqOffset, valueLengths, uniqueLen); gpuerr();
	valueLengthsHost = device_to_host(valueLengths, uniqueLen);

	int start = 0, nChunk;
	Int2* pairPtr = pairOut3;
	int* valueLengthsPtr = valueLengths;
	while ((nChunk = solve_next_bin(valueLengthsHost, start, ctx.bandwidth2, uniqueLen)) > 0) {
		// cal valueOffsets
		int* valueOffsets;
		cudaMalloc(&valueOffsets, nChunk * sizeof(int)); gpuerr();
		inclusive_sum(valueLengthsPtr, valueOffsets, nChunk);
		int outputLen = transfer_last_element(valueOffsets, nChunk);

		// cal repertoire
		_cudaMalloc(pairOut, freqOut, outputLen);
		pair2rep <<< NUM_BLOCK(outputLen), NUM_THREADS>>>(
		    pairOut3, pairOut, freqOut, seqInfo, seqOffset, valueOffsets, outputLen); gpuerr();
		_cudaFree(pairOut3, valueOffsets);
		sort_key_values2(pairOut, freqOut, outputLen);
		_cudaMalloc(pairOut2, freqOut2, outputLen);
		sum_by_key(pairOut, pairOut2, freqOut, freqOut2, buffer, outputLen);
		_cudaFree(pairOut, freqOut);

		// wrap up
		int finalLen = transfer_last_element(buffer, 1);
		XTNOutput newValue = {
			.indexPairs = shrink(pairOut2, finalLen),
			.pairwiseFrequencies = shrink(freqOut2, finalLen),
			.len = finalLen
		};
		allOutputs.push_back(newValue);

		// increment
		start += nChunk;
		pairPtr += nChunk;
		valueLengthsPtr += nChunk;
	}
	cudaFreeHost(valueLengthsHost); gpuerr();
}

XTNOutput mergeOutput(std::vector<XTNOutput> allOutputs, int* buffer) {
	int totalLen = 0;
	for (auto &element : allOutputs)
		totalLen += element.len;

	Int2* indexBuffer, *indexOut;
	size_t* freqBuffer, *freqOut;
	_cudaMalloc(indexBuffer, freqBuffer, totalLen);

	Int2* indexBufferP = indexBuffer;
	size_t* freqBufferP = freqBuffer;
	for (auto &element : allOutputs) {
		if (element.len == 0)
			continue;
		cudaMemcpy(indexBufferP, element.indexPairs,
		           sizeof(Int2)*element.len, cudaMemcpyDeviceToDevice); gpuerr();
		cudaMemcpy(freqBufferP, element.pairwiseFrequencies,
		           sizeof(size_t)*element.len, cudaMemcpyDeviceToDevice); gpuerr();
		_cudaFree(element.indexPairs, element.pairwiseFrequencies);
		indexBufferP += element.len;
		freqBufferP += element.len;
	}

	sort_key_values2(indexBuffer, freqBuffer, totalLen);
	_cudaMalloc(indexOut, freqOut, totalLen);
	sum_by_key(indexBuffer, indexOut, freqBuffer, freqOut, buffer, totalLen);

	XTNOutput ans = {
		.indexPairs = device_to_host(indexOut, totalLen),
		.pairwiseFrequencies = device_to_host(freqOut, totalLen),
		.len = transfer_last_element(buffer, 1)
	};
	_cudaFree(indexOut, freqOut);
	return ans;
}

/**
 * deduplicate input and initialize output variable
 *
 * @param seq input sequence
 * @param seqOut deduplicated input sequence
 * @param infoInOut information of each input sequence
 * @param infoOffsetOut offset of each info for each unique sequence
 * @param output final output of the function
 * @param seqLen number of input sequence
 * @param buffer integer buffer
*/
int overlap_mode_init(Int3* seq, Int3* &seqOut, SeqInfo* &infoInOut, int* &infoOffsetOut,
                      std::vector<XTNOutput> &allOutputs, int seqLen, int* buffer) {
	int* outputOffset;
	Int2* indexPairs, *indexPairs2;
	size_t* pairwiseFreq, *pairwiseFreq2;

	// create grouping
	sort_key_values(seq, infoInOut, seqLen);
	_cudaMalloc(infoOffsetOut, seqOut, seqLen);
	unique_counts(seq, infoOffsetOut, seqOut, buffer, seqLen);
	int uniqueLen = transfer_last_element(buffer, 1);

	// cal offset
	cudaMalloc(&outputOffset, uniqueLen * sizeof(int)); gpuerr();
	cal_pair_len_diag <<< NUM_BLOCK(uniqueLen), NUM_THREADS>>>(infoOffsetOut, outputOffset, uniqueLen); gpuerr();
	inclusive_sum(outputOffset, uniqueLen);
	inclusive_sum(infoOffsetOut, uniqueLen);

	// init output
	int outputLen = transfer_last_element(outputOffset, uniqueLen);
	_cudaMalloc(indexPairs, pairwiseFreq, outputLen);
	init_overlap_output <<< NUM_BLOCK(uniqueLen), NUM_THREADS>>>(infoInOut, indexPairs,
	        pairwiseFreq, infoOffsetOut, outputOffset, uniqueLen); gpuerr();

	// merge output
	sort_key_values2(indexPairs, pairwiseFreq, outputLen);
	_cudaMalloc(indexPairs2, pairwiseFreq2, outputLen);
	sum_by_key(indexPairs, indexPairs2, pairwiseFreq, pairwiseFreq2, buffer, outputLen);
	_cudaFree(outputOffset, indexPairs, pairwiseFreq);

	// wrap up
	int finalLen = transfer_last_element(buffer, 1);
	XTNOutput newOutput = {
		.indexPairs = shrink(indexPairs2, finalLen),
		.pairwiseFrequencies = shrink(pairwiseFreq2, finalLen),
		.len = finalLen
	};
	allOutputs.push_back(newOutput);
	return uniqueLen;
}