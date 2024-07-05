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
	int newValue = 1 + len; // d=1

	// d>=2
	if ((distance > 1) && (len > 1))
		newValue += len * (len - 1) / 2;

	// d>2 case
	if ((distance > 2) && (len > 2)) {
		int latestValue = len * (len - 1) / 2;
		for (int i = 3; i <= distance; i++) {
			latestValue = latestValue * (len - i + 1) / i;
			if (latestValue <= 0)
				break;
			newValue += latestValue;
		}
	}
	output[tid] = newValue;
}

/**
 * precalculate the output range of generate pair operation.
 *
 * @param inputRange range of each input group
 * @param outputRange range of each output group
 * @param n array length of inputRange and outputRange
*/
__global__
void cal_pair_len(int* inputRange, int* outputRange, int n) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid >= n)
		return;

	size_t intermediate = inputRange[tid];
	intermediate = intermediate * (intermediate - 1) / 2;
	if (intermediate > MAX)
		printf("cal_pair_len overflow\n");
	outputRange[tid] = intermediate;
}

/**
 * precalculate the output range of pair generation in diagonal position for overlap mode.
 *
 * @param inputRange range of each input group
 * @param outputRange range of each output group
 * @param n array length of inputRange and outputRange
*/
__global__
void cal_pair_len_diag(int* inputRange, int* outputRange, int n) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid >= n)
		return;

	int len = inputRange[tid];
	outputRange[tid] = len * (len + 1) / 2;
}

/**
 * precalculate the output range of pair generation in diagonal position for overlap mode.
 *
 * @param pairs pairs of sequences
 * @param seqOffset range of each input group
 * @param outputRange range of each output group
 * @param n array length of inputRange and outputRange
*/
__global__
void cal_pair_len_nondiag(Int2* pairs, int* seqOffset, int* outputRange, int n) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid >= n)
		return;

	Int2 pair = pairs[tid];
	int len1 = (pair.x == 0) ? seqOffset[pair.x] : (seqOffset[pair.x] - seqOffset[pair.x - 1]);
	int len2 = (pair.y == 0) ? seqOffset[pair.y] : (seqOffset[pair.y] - seqOffset[pair.y - 1]);
	outputRange[tid] = len1 * len2;
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
 * calculate Levenshtein distance of 2 strings in GPU where len1,len2<=18.
 * @param allStr database of all sequences
 * @param start1 start index of string one
 * @param start2 start index of string two
 * @param len1 length of string one
 * @param len2 length of string two
*/
__device__
char levenshtein_static(char* allStr, unsigned int start1, unsigned int start2, int len1, int len2) {
	char x, y, lastdiag, olddiag;
	char column[MAX_COMPRESSED_LENGTH + 1];

	for (y = 1; y <= len1; y++)
		column[y] = y;
	for (x = 1; x <= len2; x++) {
		column[0] = x;
		for (y = 1, lastdiag = x - 1; y <= len1; y++) {
			olddiag = column[y];
			column[y] = MIN3(column[y] + 1, column[y - 1] + 1, lastdiag +
			                 (allStr[start1 + y - 1] == allStr[start2 + x - 1] ? 0 : 1));
			lastdiag = olddiag;
		}
	}
	return column[len1];
}

/**
 * calculate Levenshtein distance of 2 strings in GPU where len1>18 or len2>18.
 * @param allStr database of all sequences
 * @param start1 start index of string one
 * @param start2 start index of string two
 * @param len1 length of string one
 * @param len2 length of string two
*/
__device__
char levenshtein(char* allStr, unsigned int start1, unsigned int start2, int len1, int len2) {
	char x, y, lastdiag, olddiag;
	char* column = new char[(len1 > len2) ? len1 : len2];

	for (y = 1; y <= len1; y++)
		column[y] = y;
	for (x = 1; x <= len2; x++) {
		column[0] = x;
		for (y = 1, lastdiag = x - 1; y <= len1; y++) {
			olddiag = column[y];
			column[y] = MIN3(column[y] + 1, column[y - 1] + 1, lastdiag +
			                 (allStr[start1 + y - 1] == allStr[start2 + x - 1] ? 0 : 1));
			lastdiag = olddiag;
		}
	}
	free(column);
	return column[len1];
}

/**
 * calculate Hamming distance of 2 strings in GPU.
 * @param allStr database of all sequences
 * @param start1 start index of string one
 * @param start2 start index of string two
 * @param len1 length of string one
 * @param len2 length of string two
*/
__device__
char hamming(char* allStr, unsigned int start1, unsigned int start2, int len1, int len2) {
	if (len1 != len2)
		return 77;

	char ans = 0;
	for (int i = 0; i < len1; i++)
		if (allStr[start1 + i] != allStr[start2 + i])
			ans++;
	return ans;
}

/**
 * calculate distances of strings from given pairs and flag ones exceeding the threshold.
 *
 * @param seq sequence input
 * @param index pairs of sequence to calculate
 * @param distance Levenshtein/Hamming distance threshold
 * @param measure enum representing Levenshtein/Hamming
 * @param distanceOutput output distance, maybe null
 * @param flagOutput array output flag
 * @param n array length of index
 * @param seqLen array length of seq
*/
__global__
void cal_distance(char* allStr, unsigned int* offsets, Int2* index, int distance, char measure,
                  char* distanceOutput, char* flagOutput, int n, int seqLen) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid >= n)
		return;

	Int2 indexPair = index[tid];
	if (indexPair.x == indexPair.y) {
		flagOutput[tid] =  0;
		return;
	}

	if (indexPair.x >= seqLen || indexPair.y >= seqLen) {
		printf("curious case! %d %d\n", indexPair.x, indexPair.y);
		flagOutput[tid] =  0;
		return;
	}

	unsigned int start1 = offsets[indexPair.x], start2 = offsets[indexPair.y];
	int len1 = offsets[indexPair.x + 1] - offsets[indexPair.x];
	int len2 = offsets[indexPair.y + 1] - offsets[indexPair.y];
	char newOutput;
	if (measure == HAMMING)
		newOutput = hamming(allStr, start1, start2, len1, len2);
	else if ((len1 > MAX_COMPRESSED_LENGTH) || (len2 > MAX_COMPRESSED_LENGTH))
		newOutput = levenshtein(allStr, start1, start2, len1, len2);
	else
		newOutput = levenshtein_static(allStr, start1, start2, len1, len2);

	if (distanceOutput != NULL)
		distanceOutput[tid] = newOutput;
	flagOutput[tid] =  newOutput <= distance;
}

/**
 * turning pairs and frequencies from sequence format to repertoire format.
 *
 * @param pairs pair result from nearest neighbor search
 * @param indexOut repertiore pair output
 * @param freqOut frequency output
 * @param seqInfo information of each CDR3 sequence
 * @param inputOffsets seqInfo Offset
 * @param outputOffsets output range of indexOut and freqOut
 * @param n number of pairs
*/
__global__
void pair2rep(Int2* pairs, Int2* indexOut, size_t* freqOut, SeqInfo* seqInfo,
              int* inputOffsets, int* outputOffsets, int n) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid >= n)
		return;

	Int2 pair = pairs[tid];
	int start1 = (pair.x == 0) ? 0 : inputOffsets[pair.x - 1];
	int end1 = inputOffsets[pair.x];
	int start2 = inputOffsets[pair.y - 1]; // pair.y >= pair.x >=0 and pair.x!=pair.y
	int end2 = inputOffsets[pair.y];
	int outputIndex = tid == 0 ? 0 : outputOffsets[tid - 1];

	// produce output
	for (int i = start1; i < end1; i++) {
		SeqInfo infoI = seqInfo[i];
		for (int j = start2; j < end2; j++) {
			SeqInfo infoJ = seqInfo[j];
			if (infoI.repertoire > infoJ.repertoire)
				indexOut[outputIndex] = {.x = infoJ.repertoire, .y = infoI.repertoire};
			else
				indexOut[outputIndex] = {.x = infoI.repertoire, .y = infoJ.repertoire};
			if (infoI.repertoire == infoJ.repertoire)
				// same repertoire must be counted twice
				freqOut[outputIndex++] = ((size_t)infoI.frequency) * infoJ.frequency * 2;
			else
				freqOut[outputIndex++] = ((size_t)infoI.frequency) * infoJ.frequency;
		}
	}
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

/**
 * utility to cast types.
 *
 * @param input input array
 * @param output output array
 * @param n number of rows
*/
__global__
void toSizeT(int* input, size_t* output, int n) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid >= n)
		return;
	output[tid] = input[tid];
}

/**
 * generate initial output of overlap mode.
 *
 * @param info information of each sequence
 * @param indexOut repertoire pair output
 * @param freqOut frequency output for the pair
 * @param inputOffsets index range of input to operate on
 * @param outputOffsets index range of output to operate on
 * @param n length of inputOffset and outputOffset
*/
__global__
void init_overlap_output(SeqInfo* info, Int2* indexOut, size_t* freqOut,
                         int* inputOffsets, int* outputOffsets, int n) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid >= n)
		return;

	int start = tid == 0 ? 0 : inputOffsets[tid - 1];
	int end = inputOffsets[tid];
	int outputIndex = tid == 0 ? 0 : outputOffsets[tid - 1];

	for (int i = start; i < end; i++) {
		SeqInfo infoI = info[i];
		for (int j = i; j < end; j++) {
			SeqInfo infoJ = info[j];
			indexOut[outputIndex] = {.x = infoI.repertoire, .y = infoJ.repertoire};
			freqOut[outputIndex++] = ((size_t)infoI.frequency) * infoJ.frequency;
		}
	}
}

__global__
void toInt3(char* inputs, unsigned int* offsets, Int3* output, int n) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid >= n)
		return;

	unsigned int start = offsets[tid], end = offsets[tid + 1];
	Int3 ans;
	if (end > MAX_COMPRESSED_LENGTH)
		end = MAX_COMPRESSED_LENGTH;
	for (int i = start; i < end; i++) {
		int value = inputs[i] - BEFORE_A_CHAR;
		ans.entry[i / 6] |= value << (27 - 5 * (i % 6));
	}
	output[tid] = ans;
}