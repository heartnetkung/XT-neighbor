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

	int expectedUniqueLen = 3, expectedOutputLen = 3;
	check(uniqueLen == expectedUniqueLen);
	check(output.len == expectedOutputLen);

	SeqInfo expectedInfo[] = {
		{.frequency = 3, .repertoire = 0}, {.frequency = 5, .repertoire = 1},
		{.frequency = 4, .repertoire = 0}, {.frequency = 6, .repertoire = 1}
	};
	check_device_arr(infoD, expectedInfo, seqLen);

	int expectedInfoOffset[] = {2, 3, 4};
	check_device_arr(infoOffset, expectedInfoOffset, uniqueLen);

	char expectedSeqs[seqLen][6] = {"CAAA", "CADA", "CDKD"};
	Int3 expectedSeqOut[];
	cudaMallocHost(&expectedSeqOut, sizeof(Int3)*uniqueLen);
	for (int i = 0; i < uniqueLen; i++)
		expectedSeqOut[i] = str_encode(expectedSeqs[i]);
	check_device_arr(seqOut, expectedSeqOut, uniqueLen);

	Int2 expectedIndexPair[] = {{.x = 0, .y = 0}, {.x = 0, .y = 1}, {.x = 1, .y = 1}};
	check_device_arr(output.indexPairs, expectedIndexPair, output.len);

	size_t expectedPairwiseFreq[] = {25, 15, 61};
	check_device_arr(output.pairwiseFrequencies, expectedPairwiseFreq, output.len);
})