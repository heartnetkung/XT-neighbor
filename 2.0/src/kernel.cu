#include "codec.cu"

int transfer_last_element(int* deviceArr, int n) {
	int ans[1];
	cudaMemcpy(ans, deviceArr + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	return ans[0];
}

__global__
void cal_combination_len(Int3* input, int distance, int* output, int n) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid >= n)
		return;

	int len = len_decode(input[tid]);
	int newValue = 1 + len;
	switch (distance < len ? distance : len) {
	case 4:
		newValue += len * (len - 1) * (len - 2) * (len - 3) / 24;
	case 3:
		newValue += len * (len - 1) * (len - 2) / 6;
	case 2:
		newValue += len * (len - 1) / 2;
	}
	output[tid] = newValue;
}

__global__
void cal_pair_len(int* input, int* output, int n) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid >= n)
		return;

	output[tid] = input[tid] * (input[tid] - 1) / 2;
}

__global__
void generate_pairs(int* indexes, Int2* outputs, int* inputOffsets, int* outputOffsets, int n) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid >= n)
		return;

	int start = tid == 0 ? 0 : inputOffsets[tid - 1];
	int end = inputOffsets[tid];
	int outputIndex = tid == 0 ? 0 : outputOffsets[tid - 1];
	int outputEnd = outputOffsets[tid];

	for (int i = start; i < end; i++) {
		for (int j = i + 1; j < end; j++) {
			Int2 newValue;
			if (indexes[i] < indexes[j]) {
				newValue.x = indexes[i];
				newValue.y = indexes[j];
			} else {
				newValue.x = indexes[j];
				newValue.y = indexes[i];
			}
			if (outputIndex++ < outputEnd)
				outputs[outputIndex] = newValue;
			else
				printf("potential error on generate pairs\n");
		}
	}
}

__global__
void generate_smaller_index(int* indexes, int* outputs, int* inputOffsets, int* outputOffsets, int n) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid >= n)
		return;

	int start = tid == 0 ? 0 : inputOffsets[tid - 1];
	int end = inputOffsets[tid];
	int outputIndex = tid == 0 ? 0 : outputOffsets[tid - 1];
	int outputEnd = outputOffsets[tid];

	for (int i = start; i < end; i++) {
		for (int j = i + 1; j < end; j++) {
			if (outputIndex < outputEnd)
				outputs[outputIndex++] = indexes[i] < indexes[j] ? indexes[i] : indexes[j];
			else
				printf("potential error on generate pairs\n");
		}
	}
}

#define MIN3(a, b, c) ((a) < (b) ? ((a) < (c) ? (a) : (c)) : ((b) < (c) ? (b) : (c)))

__device__
char levenshtein(Int3 x1, Int3 x2) {
	char x, y, lastdiag, olddiag;
	char s1[MAX_INPUT_LENGTH];
	char s2[MAX_INPUT_LENGTH];
	char s1len = (char)len_decode(x1), s2len = (char)len_decode(x2);
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

__global__
void cal_levenshtein(Int3* seq, Int2* index, int distance,
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

	char newOutput = levenshtein(seq[indexPair.x], seq[indexPair.y]);
	distanceOutput[tid] = newOutput;
	flagOutput[tid] =  newOutput <= distance;
}

__global__
void gen_assignment(int* matrix, int* output, int nBit, int n_row, int n_column) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid >= n_column)
		return;

	size_t ans = 0;
	for (int i = 0; i < n_row; i++)
		ans += matrix[i * n_column + tid];
	ans = (ans >> nBit);
	for (int i = 0; i < n_row; i++)
		output[i * n_column + tid] = ans;
}

__global__
void select_int3(Int3* input, unsigned int* output, int n) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid >= n)
		return;
	output[tid] = input[tid].entry[0];
}

__global__
void select_int2(Int2* input, int* output, int n) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid >= n)
		return;
	output[tid] = input[tid].x;
}

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

__global__
void make_row_index(int* output, int n, int nRepeat) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid >= n)
		return;

	for (int i = tid * nRepeat; i < tid * nRepeat + nRepeat; i++)
		output[i] = tid;
}

__global__
void vector_add(int* dest, int* src, int n) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid >= n)
		return;
	dest[tid] += src[tid];
}