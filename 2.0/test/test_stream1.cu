#include "test_util.cu"
#include "../src/xtn_inner.cu"

TEST(Stream1, {
	int seqLen = 4;
	char seqs[seqLen][6] = {"CAAA", "CADA", "CAAA", "CDKD"};
	int distance = 1;
	int carry = 0;

	//allocate inputs
	Int3 * seqd, *seqh;
	std::vector<int*> histogramOutput;
	MemoryContext ctx;

	cudaMalloc(&seqd, sizeof(Int3)*seqLen);
	cudaMallocHost(&seqh, sizeof(Int3)*seqLen);

	//make inputs
	for (int i = 0; i < seqLen; i++)
		seqh[i] = str_encode(seqs[i]);
	seqd = host_to_device(seqh, seqLen);

	//do testing
	Chunk<Int3> input;
	input.ptr = seqd;
	input.len = seqLen;
	int* indexOutput;
	Int3* deletionsOutput;
	int outputLen;
	stream_handler1(input, deletionsOutput, indexOutput, histogramOutput,
	                outputLen, distance, carry, ctx);

	//expactation
	int expectedLen = 20;
	char expectedPairs[][5] = {
		"AAA", "AAA", "ADA", "CAA", "CAA", "CAA", "CAA", "CAA", "CAA", "CAA",
		"CAAA", "CAAA", "CAD", "CADA", "CDA", "CDD", "CDK", "CDKD", "CKD", "DKD"
	};
	int expectedIndex[] = {0, 2, 1, 0, 0, 0, 1, 2, 2, 2, 0, 2, 1, 1, 1, 3, 3, 3, 3, 3};
	deletionsOutput = device_to_host(deletionsOutput, outputLen);

	//check
	check(outputLen == expectedLen);
	check_device_arr(indexOutput, expectedIndex, outputLen);
	for (int i = 0; i < expectedLen; i++)
		checkstr(expectedPairs[i], str_decode(deletionsOutput[i]));
})