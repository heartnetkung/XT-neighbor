#include "test_util.cu"
#include "../src/xtn_inner.cu"

TEST(OverlapInit, {
	XTNOutput output;
	int* buffer, *infoOffset;
	Int3 * seq1d, *seq1h, *seqOut;
	SeqInfo* infoD;

	int seqLen = 4;
	char seqs[seqLen][6] = {"CAAA", "CADA", "CAAA", "CDKD"};
	SeqInfo info[] = {
		{.frequency = 3, .repertoire = 0}, {.frequency = 4, .repertoire = 0},
		{.frequency = 5, .repertoire = 1}, {.frequency = 6, .repertoire = 1}
	};

	cudaMalloc(&buffer, sizeof(int));
	cudaMalloc(&seq1d, sizeof(Int3)*seqLen);
	cudaMallocHost(&seq1h, sizeof(Int3)*seqLen);
	cudaMalloc(&infoD, sizeof(SeqInfo)*seqLen);

	//make inputs
	for (int i = 0; i < seqLen; i++)
		seq1h[i] = str_encode(seqs[i]);
	seq1d = host_to_device(seq1h, seqLen);
	infoD = host_to_device(info, seqLen);

	int uniqueLen = overlap_mode_init(seq1d, seqOut, infoD, infoOffset,
	                                  output, seqLen, buffer);
	printf("%d %d\n", uniqueLen, output.len);
	print_int3_arr(seqOut, uniqueLen);
	print_int2_arr(output.indexPairs);
	print_size_t_arr(output.pairwiseFrequencies);
})