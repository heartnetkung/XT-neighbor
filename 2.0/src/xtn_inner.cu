#include "generate_combination.cu"
#include "cub.cu"
#include "kernel.cu"
#include "codec.cu"
#include "stream.cu"

const int NUM_THREADS = 256;

int cal_offsets(Int3* inputKeys, int* inputValues, int* &inputOffsets, int* &outputOffsets, int n, int* buffer) {
	// cal valueOffsets
	cudaMalloc(&inputOffsets, sizeof(int)*n); gpuerr();
	// assume sorted
	// sort_key_values(inputKeys, inputValues, n); gpuerr();
	unique_counts(inputKeys, inputOffsets, buffer, n); gpuerr();

	// cal pairOffsets
	int nUnique = transfer_last_element(buffer, 1); gpuerr();
	int nUniqueBlock = divideCeil(nUnique, NUM_THREADS);
	cudaMalloc(&outputOffsets, sizeof(int)*nUnique); gpuerr();
	cal_pair_len <<< nUniqueBlock, NUM_THREADS>>>(inputOffsets, outputOffsets, nUnique); gpuerr();
	inclusive_sum(inputOffsets, nUnique); gpuerr();
	inclusive_sum(outputOffsets, nUnique); gpuerr();
	return nUnique;
}

int gen_pairs(int* input, int* inputOffsets, int* outputOffsets, Int2* &output, int n, int* buffer) {
	// generate pairs
	int outputLen = transfer_last_element(outputOffsets, n); gpuerr();
	int nBlock = divideCeil(n, NUM_THREADS);
	cudaMalloc(&output, sizeof(Int2)*outputLen); gpuerr();
	generate_pairs <<< nBlock, NUM_THREADS>>>(input, output,
	        inputOffsets, outputOffsets, n); gpuerr();
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
	int uniqueLenBlock = divideCeil(uniqueLen, NUM_THREADS);
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
	cudaMemset(flags, 1, valueInput.len); gpuerr();
	int inputBlocks = divideCeil(offsetLen, NUM_THREADS);

	flag_lowerbound <<< inputBlocks, NUM_THREADS>>>(
	    valueInput.ptr, valueOffsets, flags, lowerbound, offsetLen); gpuerr();
	double_flag(keyInput.ptr, valueInput.ptr, flags, keyOutput.ptr, valueOutput.ptr,
	            buffer, keyInput.len); gpuerr();

	int outputLen = transfer_last_element(buffer, 1); gpuerr();
	keyOutput.len = outputLen;
	valueOutput.len = outputLen;
	cudaFree(flags);
}


void stream_handler1(Chunk<Int3> input, Chunk<Int3> &output1,
                     Chunk<int> &output2, int distance) {
	int *combinationOffsets;
	int n = input.len;
	int inputBlocks = divideCeil(n, NUM_THREADS);

	// cal combinationOffsets
	cudaMalloc((void**)&combinationOffsets, sizeof(int)*n);	gpuerr();
	cal_combination_len <<< inputBlocks, NUM_THREADS >>>(
	    input.ptr, distance, combinationOffsets, n); gpuerr();
	inclusive_sum(combinationOffsets, n); gpuerr();
	int outputLen = transfer_last_element(combinationOffsets, n); gpuerr();

	// generate combinations
	cudaMalloc(&output1.ptr, sizeof(Int3)*outputLen); gpuerr();
	cudaMalloc(&output2.ptr, sizeof(int)*outputLen); gpuerr();
	gen_combination <<< inputBlocks, NUM_THREADS >>> (
	    input.ptr, combinationOffsets, distance, output1.ptr, output2.ptr, n); gpuerr();

	cudaFree(combinationOffsets); gpuerr();
	output1.len = outputLen;
	output2.len = outputLen;
}

void stream_handler2() {

}

void stream_handler3(Chunk<Int3> keyInput, Chunk<int> valueInput,
                     Chunk<Int3> &keyOutput, Chunk<int> &valueOutput,
                     XTNOutput &output, Int3* seq1, int seq1Len,
                     int distance, int lowerbound, int* buffer) {

	int* combinationValueOffsets, *pairOffsets;
	int offsetLen =
	    cal_offsets(keyInput.ptr, valueInput.ptr, combinationValueOffsets,
	                pairOffsets, keyInput.len, buffer);

	Int2* pairs;
	int pairLen = gen_pairs(valueInput.ptr, combinationValueOffsets,
	                        pairOffsets, pairs, offsetLen, buffer);

	Int2* pairOut;
	char* distanceOut;
	int outputLen = postprocessing(seq1, pairs, distance,
	                               pairOut, distanceOut,
	                               pairLen, buffer, seq1Len);

	make_output(pairOut, distanceOut, outputLen, output);
	gen_next_chunk(keyInput, valueInput, keyOutput, valueOutput,
	               combinationValueOffsets, offsetLen, lowerbound, buffer);
	_cudaFree(combinationValueOffsets, pairOffsets, pairs, pairOut, distanceOut);
}