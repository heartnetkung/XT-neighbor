#include "test_util.cu"
#include "../src/xtn_overlap_inner.cu"

TEST(OverlapInit, {
	std::vector<XTNOutput> allOutputs;
	int* buffer, *infoOffsetOut;
	unsigned int* offsets_d;
	Int3 *seqOut;
	SeqInfo* info_d;
	char* allStr_d;

	int seqLen = 4;
	int totalLen = 16;
	char allStr[] = "CAAACADACAAACDKD";
	unsigned int offsets[] = {0, 4, 8, 12, 16};
	SeqInfo info[] = {
		{.frequency = 3, .repertoire = 0, .originalIndex = 0},
		{.frequency = 4, .repertoire = 0, .originalIndex = 1},
		{.frequency = 5, .repertoire = 1, .originalIndex = 2},
		{.frequency = 6, .repertoire = 1, .originalIndex = 3}
	};

	cudaMalloc(&buffer, sizeof(int));
	info_d = host_to_device(info, seqLen);
	offsets_d = host_to_device(offsets, seqLen + 1);
	allStr_d = host_to_device(allStr, totalLen);

	int uniqueLen = overlap_mode_init(allStr_d, offsets_d, seqOut, info_d, infoOffsetOut,
	                                  allOutputs, seqLen, buffer);
	XTNOutput output = allOutputs.back();
	printf("aa %d", uniqueLen);

	// int expectedUniqueLen = 3, expectedOutputLen = 3, expectedOutputCount = 1;
	// check(uniqueLen == expectedUniqueLen);
	// check(output.len == expectedOutputLen);
	// check(allOutputs.size() == expectedOutputCount);

	// SeqInfo expectedInfo[] = {
	// 	{.frequency = 3, .repertoire = 0}, {.frequency = 5, .repertoire = 1},
	// 	{.frequency = 4, .repertoire = 0}, {.frequency = 6, .repertoire = 1}
	// };
	// check_device_arr(infoD, expectedInfo, seqLen);

	// int expectedInfoOffset[] = {2, 3, 4};
	// check_device_arr(infoOffset, expectedInfoOffset, uniqueLen);

	// char expectedSeqs[seqLen][6] = {"CAAA", "CADA", "CDKD"};
	// Int3* expectedSeqOut;
	// cudaMallocHost(&expectedSeqOut, sizeof(Int3)*uniqueLen);
	// for (int i = 0; i < uniqueLen; i++)
	// 	expectedSeqOut[i] = str_encode(expectedSeqs[i]);
	// check_device_arr(seqOut, expectedSeqOut, uniqueLen);

	// Int2 expectedIndexPair[] = {{.x = 0, .y = 0}, {.x = 0, .y = 1}, {.x = 1, .y = 1}};
	// check_device_arr(output.indexPairs, expectedIndexPair, output.len);

	// size_t expectedPairwiseFreq[] = {25, 15, 61};
	// check_device_arr(output.pairwiseFrequencies, expectedPairwiseFreq, output.len);
})