#include "codec.cu"
#include <limits.h>

const size_t MAX = INT_MAX;

/**
 * @file
 * A collection of most GPU parallel primitives that is implemented as CUDA kernel
 * (most map and expand operations). Follows Facade design pattern.
 */

/**
 * transfer last element of the GPU array to main memory.
 * @param deviceArr the GPU array
 * @param n array length
*/
template <typename T>
T transfer_last_element(T* deviceArr, int n) {
	T ans[1];
	cudaMemcpy(ans, deviceArr + n - 1, sizeof(T), cudaMemcpyDeviceToHost); gpuerr();
	cudaDeviceSynchronize(); gpuerr();
	return ans[0];
}

/**
 * precalculate the number of positions required in the output array of generate combination operation.
 *
 * @param input sequences to generate combination
 * @param distance Levenshtein threshold
 * @param output position output for each sequence
 * @param n array length of input and output
*/
__global__
void cal_combination_len(Int3* input, int distance, int* output, int n) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid >= n)
		return;

	int len = len_decode(input[tid]);
	int newValue = 1 + len;
	if (distance == 2)
		newValue += len * (len - 1) / 2;
	// distance larger than 2 is not supported

	output[tid] = newValue;
}

/**
 * precalculate the number of positions required in the output array of generate pair operation.
 *
 * @param input group size
 * @param output position requirement
 * @param n array length of input and output
*/
__global__
void cal_pair_len(int* input, int* output, int n) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid >= n)
		return;

	size_t intermediate = input[tid];
	intermediate = intermediate * (intermediate - 1) / 2;
	if (intermediate > MAX)
		printf("cal_pair_len overflow\n");
	output[tid] = intermediate;
}

/**
 * precalculate the number of positions required in the output array of generate pair operation with lower bound constratint.
 *
 * @param indexes value of seqIndexes to generate pair
 * @param inputOffsets group offsets
 * @param outputLengths output position requirement
 * @param lowerbound the processing limit for the indexes
 * @param n array length
*/
__global__
void cal_pair_len_lowerbound(int* indexes, int* inputOffsets, int* outputLengths, int lowerbound, int n) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid >= n)
		return;

	int start = tid == 0 ? 0 : inputOffsets[tid - 1];
	int end = inputOffsets[tid];
	int invalidCount = 0;
	for (int i = start; i < end; i++)
		if (indexes[i] > lowerbound)
			invalidCount++;

	size_t intermediate = end - start;
	intermediate = ((intermediate * (intermediate - 1)) - (invalidCount * (invalidCount - 1)) ) / 2;
	if (intermediate > MAX)
		printf("cal_pair_len_lowerbound overflow\n");
	outputLengths[tid] = intermediate;
}

/**
 * combinatorially generate pairs of indexes within the same group.
 *
 * @param indexes value of seqIndexes to generate pair
 * @param outputs pairs output
 * @param inputOffsets precalculated group offsets
 * @param outputOffsets precalculated output position requirement
 * @param lesserIndex by partial output for histogram
 * @param lowerbound the processing limit for the indexes
 * @param carry latest offset from previous chunk in the stream
 * @param n array length of inputOffsets and outputOffsets
*/
__global__
void generate_pairs(int* indexes, Int2* outputs, int* inputOffsets, int* outputOffsets,
                    int* lesserIndex, int lowerbound, int carry, int n) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid >= n)
		return;

	int start = tid == 0 ? carry : inputOffsets[tid - 1];
	int end = inputOffsets[tid];
	int outputIndex = tid == 0 ? 0 : outputOffsets[tid - 1];
	int outputEnd = outputOffsets[tid];

	for (int i = start; i < end; i++) {
		for (int j = i + 1; j < end; j++) {
			Int2 newValue;
			if (indexes[i] < indexes[j]) {
				if (indexes[i] > lowerbound)
					continue;
				newValue.x = indexes[i];
				newValue.y = indexes[j];
			} else {
				if (indexes[j] > lowerbound)
					continue;
				newValue.x = indexes[j];
				newValue.y = indexes[i];
			}
			if (outputIndex < outputEnd) {
				outputs[outputIndex] = newValue;
				lesserIndex[outputIndex++] = newValue.x;
			}
			else
				printf("[1]potential error on generate pairs\n");
		}
	}
}

/**
 * combinatorially generate pairs of indexes within the same group but record only the partial output.
 *
 * @param indexes value of seqIndexes to generate pair
 * @param outputs smaller index output
 * @param inputOffsets precalculated group offsets
 * @param outputOffsets precalculated output position requirement
 * @param carry latest offset from previous chunk in the stream
 * @param n array length of inputOffsets and outputOffsets
*/
__global__
void generate_smaller_index(int* indexes, int* outputs, int* inputOffsets,
                            int* outputOffsets, int carry, int n) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid >= n)
		return;

	int start = tid == 0 ? carry : inputOffsets[tid - 1];
	int end = inputOffsets[tid];
	int outputIndex = tid == 0 ? 0 : outputOffsets[tid - 1];
	int outputEnd = outputOffsets[tid];

	for (int i = start; i < end; i++) {
		for (int j = i + 1; j < end; j++) {
			if (outputIndex < outputEnd)
				outputs[outputIndex++] = indexes[i] < indexes[j] ? indexes[i] : indexes[j];
			else
				printf("[2]potential error on generate pairs\n");
		}
	}
}

#define MIN3(a, b, c) ((a) < (b) ? ((a) < (c) ? (a) : (c)) : ((b) < (c) ? (b) : (c)))

/**
 * calculate Levenshtein distance of 2 strings in GPU.
 * @param x1 first string
 * @param x2 second string
*/
__device__
char levenshtein(Int3 x1, Int3 x2) {
	char s1len = (char)len_decode(x1), s2len = (char)len_decode(x2);
	char x, y, lastdiag, olddiag;
	char s1[MAX_INPUT_LENGTH];
	char s2[MAX_INPUT_LENGTH];
	char column[MAX_INPUT_LENGTH + 1];

	for (int i = 0; i < MAX_INPUT_LENGTH; i++) {
		char c = (x1.entry[i / 6] >> (27 - 5 * (i % 6))) & 0x1F;
		if (c == 0)
			break;
		s1[i] = BEFORE_A_CHAR + c;
	}
	for (int i = 0; i < MAX_INPUT_LENGTH; i++) {
		char c = (x2.entry[i / 6] >> (27 - 5 * (i % 6))) & 0x1F;
		if (c == 0)
			break;
		s2[i] = BEFORE_A_CHAR + c;
	}

	for (y = 1; y <= s1len; y++)
		column[y] = y;
	for (x = 1; x <= s2len; x++) {
		column[0] = x;
		for (y = 1, lastdiag = x - 1; y <= s1len; y++) {
			olddiag = column[y];
			column[y] = MIN3(column[y] + 1, column[y - 1] + 1, lastdiag + (s1[y - 1] == s2[x - 1] ? 0 : 1));
			lastdiag = olddiag;
		}
	}
	return column[s1len];
}

/**
 * calculate Hamming distance of 2 strings in GPU.
 * @param x1 first string
 * @param x2 second string
*/
__device__
char hamming(Int3 x1, Int3 x2) {
	char s1len = (char)len_decode(x1), s2len = (char)len_decode(x2);
	if (s1len != s2len)
		return 77;

	char ans = 0;
	for (int i = 0; i < s1len; i++) {
		char c1 = (x1.entry[i / 6] >> (27 - 5 * (i % 6))) & 0x1F;
		char c2 = (x2.entry[i / 6] >> (27 - 5 * (i % 6))) & 0x1F;
		if (c1 != c2)
			ans++;
	}
	return ans;
}

/**
 * calculate distances of strings from given pairs and flag ones exceeding the threshold.
 *
 * @param seq sequence input
 * @param index pairs of sequence to calculate
 * @param distance Levenshtein/Hamming distance threshold
 * @param measure enum representing Levenshtein/Hamming
 * @param distanceOutput output distance
 * @param flagOutput array output flag
 * @param n array length of index
 * @param seqLen array length of seq
*/
__global__
void cal_distance(Int3* seq, Int2* index, int distance, char measure,
                     char* distanceOutput, char* flagOutput, int n, int seqLen) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid >= n)
		return;

	Int2 indexPair = index[tid];
	if (indexPair.x == indexPair.y) {
		flagOutput[tid] =  0;
		distanceOutput[tid] = 99;
		return;
	}

	if (indexPair.x >= seqLen || indexPair.y >= seqLen) {
		printf("curious case! %d %d\n", indexPair.x, indexPair.y);
		flagOutput[tid] =  0;
		distanceOutput[tid] = 88;
		return;
	}

	char newOutput = measure == LEVENSHTEIN ?
	                 levenshtein(seq[indexPair.x], seq[indexPair.y]) :
	                 hamming(seq[indexPair.x], seq[indexPair.y]);
	distanceOutput[tid] = newOutput;
	flagOutput[tid] =  newOutput <= distance;
}

/**
 * expand operation part of solving bin packing for 2D buffer.
 *
 * @param matrix statistics of all chunks where each row record the histogram count of each chunk and nRow=nChunk
 * @param output assignment of each chunk to the bins
 * @param nBit bin capacity expressed in log2 form
 * @param nRow number of rows of the matrix
 * @param nColumn number of columns of the matrix
*/
__global__
void gen_assignment(int* matrix, int* output, int nBit, int nRow, int nColumn) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid >= nColumn)
		return;

	size_t ans = 0;
	for (int i = 0; i < nRow; i++)
		ans += matrix[i * nColumn + tid];
	ans = (ans >> nBit);
	if (ans > MAX)
		printf("gen_assignment overflow\n");
	for (int i = 0; i < nRow; i++)
		output[i * nColumn + tid] = ans;
}

/**
 * expand operation part of solving bin packing for lower bound.
 *
 * @param matrix statistics of all chunks where each row record the histogram count of each chunk and nRow=nChunk
 * @param keyOut the regrouping of each bin
 * @param valueOut the upper bound of each grouped bin
 * @param nBit bin capacity expressed in log2 form
 * @param valueMax last sequence index
 * @param nRow number of rows of the matrix
 * @param nColumn number of columns of the matrix
*/
__global__
void gen_bounds(size_t* matrix, int* keyOut, int* valueOut, int nBit, int valueMax, int nRow, int nColumn) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid >= nColumn)
		return;

	size_t intermediate = valueMax;
	valueOut[tid] = intermediate * (tid + 1) / nColumn - 1;

	size_t ans = 0;
	for (int i = 0; i < nRow; i++)
		ans += matrix[i * nColumn + tid];
	ans = (ans >> nBit);
	if (ans > MAX)
		printf("gen_bounds overflow");
	keyOut[tid] = ans;
}

/**
 * flag data to be removed after the lower bound has been processed. This includes both useless group and processed rows.
 *
 * @param valueInput seqIndex input
 * @param valueOffsets group offset
 * @param output flag output
 * @param lowerbound the lowerbound used
 * @param n array length of valueOffsets
*/
__global__
void flag_lowerbound(int* valueInput, int* valueOffsets, char* output, int lowerbound, int n) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid >= n)
		return;

	int start = tid == 0 ? 0 : valueOffsets[tid - 1];
	int end = valueOffsets[tid];
	int validCount = 0;

	for (int i = start; i < end; i++) {
		if (valueInput[i] > lowerbound)
			validCount++;
		else
			output[i] = 0;
	}

	if (validCount < 2)
		for (int i = start; i < end; i++)
			output[i] = 0;
}

/**
 * utility to generate keys for matrix processing.
 *
 * @param output key output with range 0 to n-1 each repeating nRepeat time
 * @param n number of rows
 * @param nRepeat number of columns
*/
__global__
void make_row_index(int* output, int n, int nRepeat) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid >= n)
		return;

	for (int i = tid * nRepeat; i < tid * nRepeat + nRepeat; i++)
		output[i] = tid;
}


__global__
void toSizeT(int* input, size_t* output, int n) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid >= n)
		return;
	output[tid] = input[tid];
}