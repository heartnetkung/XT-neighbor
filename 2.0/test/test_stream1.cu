#include "test_util.cu"
#include "../src/xtn_inner.cu"

TEST(Stream1, {
	int seqLen = 4;
	char seqs[seqLen][6] = {"CAAA", "CADA", "CAAA", "CDKD"};
	int distance = 1;

	//allocate inputs
	Int3 * seq1d, *seq1h;
	int* histogramOutput;
	cudaMalloc(&seq1d, sizeof(Int3)*seqLen);
	cudaMallocHost(&seq1h, sizeof(Int3)*seqLen);
	MemoryContext ctx;

	//make inputs
	for (int i = 0; i < seqLen; i++)
		seq1h[i] = str_encode(seqs[i]);
	seq1d = host_to_device(seq1h, seqLen);

	//do testing
	Chunk<Int3> input;
	input.ptr = seq1d;
	input.len = seqLen;
	int* indexOutput;
	Int3* deletionsOutput;
	int outputLen;
	stream_handler1(input, deletionsOutput, indexOutput, histogramOutput, outputLen, distance, ctx);

	//expactation
	int expectedLen = 20;
	char expectedPairs[][5] = {
		"AAA", "AAA", "ADA", "CAA", "CAA", "CAA", "CAA", "CAA", "CAA", "CAA",
		"CAAA", "CAAA", "CAD", "CADA", "CDA", "CDD", "CDK", "CDKD", "CKD", "DKD"
	};
	int expectedIndex[] = {0, 2, 1, 0, 0, 0, 1, 2, 2, 2, 0, 2, 1, 1, 1, 3, 3, 3, 3, 3};
	deletionsOutput = device_to_host(deletionsOutput, outputLen);
	indexOutput = device_to_host(indexOutput, outputLen);

	//check
	check(outputLen == expectedLen);
	for (int i = 0; i < expectedLen; i++) {
		checkstr(expectedPairs[i], str_decode(deletionsOutput[i]));
		check(expectedIndex[i] == indexOutput[i]);
	}
})