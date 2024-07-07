#include "xtn_inner.cu"

//=====================================
// private functions
//=====================================

__device__
char* _allStr = NULL; /*global variable for callback*/
__device__
unsigned int* _allStrOffset = NULL; /*global variable for callback*/
__global__
void _setGlobalVar(char* allStr, unsigned int* allStrOffset) {
	_allStr = allStr;
	_allStrOffset = allStrOffset;
}

__device__
bool SeqInfo::operator==(const SeqInfo& t) const {
	unsigned int start1 = _allStrOffset[originalIndex], start2 = _allStrOffset[t.originalIndex];
	int len1 = _allStrOffset[originalIndex + 1] - start1;
	int len2 = _allStrOffset[t.originalIndex + 1] - start2;
	if (len1 != len2) return false;

	for (int i = 0; i < len1; i++) {
		char c1 = _allStr[start1 + i], c2 = _allStr[start2 + i];
		if (c1 != c2)
			return false;
	}
	return true;
}

/**
 * deduplicate the sequence on the full length level and generate related data
 *
 * @param allStr container of all sequences
 * @param allStrOffsets start/end position of each sequence
 * @param info information of each input sequence, will be sorted as the side-effect of this operation
 * @param seqOut deduplicated sequence in Int3 form
 * @param infoLenOut , has the same length as seqOut
 * @param seqLen number of input sequence
 * @param buffer integer buffer
 * @return the length of seqOut
*/
int deduplicate_full_length(char* allStr, unsigned int* allStrOffsets, SeqInfo* &info,
                            Int3* &seqOut, int* &infoLenOut, int seqLen, int* buffer) {
	SeqInfo* uniqueSeqInfo;

	_setGlobalVar <<< 1, 1>>>(allStr, allStrOffsets);
	sort_info(info, allStr, allStrOffsets, seqLen);

	_cudaMalloc(infoLenOut, uniqueSeqInfo, seqLen);
	unique_counts(info, infoLenOut, uniqueSeqInfo, buffer, seqLen);
	int uniqueLen = transfer_last_element(buffer, 1);

	cudaMalloc(&seqOut, sizeof(Int3)*uniqueLen); gpuerr();
	toInt3 <<< NUM_BLOCK(uniqueLen), NUM_THREADS>>>(
	    allStr, allStrOffsets, uniqueSeqInfo, seqOut, uniqueLen); gpuerr();
	cudaFree(uniqueSeqInfo); gpuerr();
	return uniqueLen;
}

/**
 * deduplicate input and initialize output variable
 *
 * @param allStr container of all sequences
 * @param allStrOffsets start/end position of each sequence
 * @param seqOut return sequence in Int3 form
 * @param info information of each input sequence, will be sorted as the side-effect of this operation
 * @param infoOffsetOut offset of each info for each unique sequence
 * @param allOutputs container of generated result
 * @param seqLen number of input sequence
 * @param buffer integer buffer
 * @return the length of seqOut
*/
int overlap_mode_init(char* allStr, unsigned int* allStrOffsets, Int3* &seqOut, SeqInfo* &info,
                      int* &infoOffsetOut, std::vector<XTNOutput> &allOutputs, int seqLen, int* buffer) {
	int* outputOffset;
	Int2* indexPairs, *indexPairs2;
	size_t* pairwiseFreq, *pairwiseFreq2;

	int uniqueLen = deduplicate_full_length(allStr, allStrOffsets, info, seqOut, outputOffset, seqLen, buffer);
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
	XTNOutput newOutput;
	newOutput.indexPairs = shrink(indexPairs2, finalLen);
	newOutput.pairwiseFrequencies = shrink(pairwiseFreq2, finalLen);
	newOutput.len = finalLen;
	allOutputs.push_back(newOutput);
	return uniqueLen;
}

/**
 * handle all GPU operations in stream 4 overlap mode.
 *
 * @param pairInput nearest neighbor pairs
 * @param allOutputs container of generated results
 * @param seq input CDR3 sequences
 * @param seqInfo information of each sequence
 * @param seqOffset offset of seqInfo array
 * @param seqLen number of input CDR3 sequences
 * @param distance distance threshold
 * @param measure type of measurement (levenshtein/hamming)
 * @param buffer integer buffer
 * @param ctx memory context
*/
void stream_handler4_overlap(Chunk<Int2> pairInput, std::vector<XTNOutput> &allOutputs, char* allStr, unsigned int* allStrOffsets,
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
	    allStr, allStrOffsets, uniquePairs, distance, measure, NULL,
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
		print_bandwidth(outputLen, ctx.bandwidth2, "4b");

		// cal repertoire
		_cudaMalloc(pairOut, freqOut, outputLen);
		pair2rep <<< NUM_BLOCK(nChunk), NUM_THREADS>>>(
		    pairPtr, pairOut, freqOut, seqInfo, seqOffset, valueOffsets, nChunk); gpuerr();
		cudaFree(valueOffsets); gpuerr();
		sort_key_values2(pairOut, freqOut, outputLen);
		_cudaMalloc(pairOut2, freqOut2, outputLen);
		sum_by_key(pairOut, pairOut2, freqOut, freqOut2, buffer, outputLen);
		_cudaFree(pairOut, freqOut);

		// wrap up
		int finalLen = transfer_last_element(buffer, 1);
		XTNOutput newValue;
		newValue.indexPairs = shrink(pairOut2, finalLen);
		newValue.pairwiseFrequencies = shrink(freqOut2, finalLen);
		newValue.len = finalLen;
		allOutputs.push_back(newValue);

		// increment
		start += nChunk;
		pairPtr += nChunk;
		valueLengthsPtr += nChunk;
	}
	cudaFreeHost(valueLengthsHost); gpuerr();
	cudaFree(pairOut3); gpuerr();
}

/**
 * merge all outputs by grouping the index keys and summing the frequency values.
 *
 * @param allOutputs container of generated results
 * @param buffer integer buffer
 * @return merged results
*/
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

	XTNOutput ans;
	ans.indexPairs = device_to_host(indexOut, totalLen);
	ans.pairwiseFrequencies = device_to_host(freqOut, totalLen);
	ans.len = transfer_last_element(buffer, 1);
	_cudaFree(indexOut, freqOut);
	return ans;
}
