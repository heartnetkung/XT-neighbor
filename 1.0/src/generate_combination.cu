#include "codec.cu"

const int MAX_DISTANCE = 2;

#ifdef TEST_ENV
__host__
#endif
__device__
void printStack(int* indexStack, int length) {
	for (int i = 1; i <= length; i++)
		printf("%d ", indexStack[i]);
	printf(">>%d\n", length);
}

__device__
void expand_values(int index, int* output, int start, int end) {
	for (int i = start; i < end; i++)
		output[i] = index;
}

#ifdef TEST_ENV
__host__
#endif
__device__
void expand_keys(Int3 seq, int distance, Int3* output, int start, int end) {
	int len = len_decode(seq);
	const int effectiveDistance = distance < len ? distance : len;

	//init stacks
	int indexStack[effectiveDistance + 1];
	Int3 seqStack[effectiveDistance + 1];
	seqStack[0] = seq;
	for (int i = 1; i <= effectiveDistance; i++) {
		indexStack[i] = i - 1;
		seqStack[i] = remove_char(seqStack[i - 1], 0);
	}

	//depth first traversal
	int stackPos = effectiveDistance;
	len--;
	for (int i = start; i < end - 1; i++) {
		// printStack(indexStack, stackPos);
		output[i] = seqStack[stackPos];

		//pop
		if (indexStack[stackPos] == len) {
			stackPos--;
			continue;
		}

		//fill
		indexStack[stackPos]++;
		seqStack[stackPos] = remove_char(seqStack[stackPos - 1], indexStack[stackPos] - stackPos + 1);

		while (indexStack[stackPos] < len) {
			if (stackPos < effectiveDistance)
				stackPos++;
			else
				break;
			indexStack[stackPos] = indexStack[stackPos - 1] + 1;
			seqStack[stackPos] = remove_char(seqStack[stackPos - 1], indexStack[stackPos] - stackPos + 1);
		}
	}

	// last step
	output[end - 1] = seqStack[0];
}

#ifdef TEST_ENV
__host__
#endif
__device__
void expand_keys2(Int3 seq, int distance, Int3* output, int start, int end) {
	int len = len_decode(seq);
	const int effectiveDistance = distance < len ? distance : len;

	//init stacks
	int indexStack[MAX_DISTANCE + 1];
	Int3 seqStack[MAX_DISTANCE + 1];
	seqStack[0] = seq;
	for (int i = 1; i <= effectiveDistance; i++) {
		indexStack[i] = i - 1;
		seqStack[i] = remove_char(seqStack[i - 1], 0);
	}

	//depth first traversal
	int stackPos = effectiveDistance;
	len--;
	for (int i = start; i < end - 1; i++) {
		// printStack(indexStack, stackPos);
		output[i] = seqStack[stackPos];

		//pop
		if (indexStack[stackPos] == len) {
			stackPos--;
			continue;
		}

		//fill
		indexStack[stackPos]++;
		seqStack[stackPos] = remove_char(seqStack[stackPos - 1], indexStack[stackPos] - stackPos + 1);

		while (indexStack[stackPos] < len) {
			if (stackPos < effectiveDistance)
				stackPos++;
			else
				break;
			indexStack[stackPos] = indexStack[stackPos - 1] + 1;
			seqStack[stackPos] = remove_char(seqStack[stackPos - 1], indexStack[stackPos] - stackPos + 1);
		}
	}

	// last step
	output[end - 1] = seqStack[0];
}

/**
 * Generate all combinations of deletion for sequences using expand primitive.
 * For a given threadId, multiple key value pairs are generated with a combination as key and threadId as value.
 *
 * This implementation doesn't use recursive for performance reason.
 *
*/
__global__
void gen_combination(Int3* seqs, int* combinationOffsets, int distance,
                     Int3* combinationKeys, int* combinationValues, int n) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid >= n)
		return;

	int start = tid == 0 ? 0 : combinationOffsets[tid - 1];
	int end = combinationOffsets[tid];

	expand_values(tid, combinationValues, start, end);
	if (distance <= 2)
		expand_keys2(seqs[tid], distance, combinationKeys, start, end);
	else
		expand_keys(seqs[tid], distance, combinationKeys, start, end);
}