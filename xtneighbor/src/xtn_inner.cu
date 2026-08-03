#include "generate_combination.cu"
#include "cub.cu"
#include "kernel.cu"
#include "codec.cu"

const int NUM_THREADS = 256;

int gen_combinations(Int3* seq, int distance, Int3* &outputKeys, int* &outputValues, int n) {
	int *combinationOffsets;
	int seq1LenBlocks = divideCeil(n, NUM_THREADS);

	// cal combinationOffsets
	cudaMalloc((void**)&combinationOffsets, sizeof(int)*n);	gpuerr();
	cal_combination_len <<< seq1LenBlocks, NUM_THREADS >>>(
	    seq, distance, combinationOffsets, n); gpuerr();
	inclusive_sum(combinationOffsets, n); gpuerr();
	int outputLen = transfer_last_element(combinationOffsets, n); gpuerr();

	// generate combinations
	cudaMalloc(&outputKeys, sizeof(Int3)*outputLen); gpuerr();
	cudaMalloc(&outputValues, sizeof(int)*outputLen); gpuerr();
	gen_combination <<< seq1LenBlocks, NUM_THREADS >>> (
	    seq, combinationOffsets, distance, outputKeys, outputValues, n); gpuerr();

	cudaFree(combinationOffsets); gpuerr();
	return outputLen;
}

int cal_offsets(Int3* inputKeys, int* inputValues, int* &inputOffsets, int* &outputOffsets, int n, int* buffer) {
	// cal valueOffsets
	cudaMalloc(&inputOffsets, sizeof(int)*n); gpuerr();
	sort_key_values(inputKeys, inputValues, n); gpuerr();
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
	int uniqueLen = transfer_last_element(buffer, 1);
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