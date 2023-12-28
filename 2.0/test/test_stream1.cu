#include "test_util.cu"
#include "../src/xtn_inner.cu"

TEST(Stream1, {
	int seqLen = 4;
	char seqs[seqLen][6] = {"CAAA", "CADA", "CAAA", "CDKD"};
	int distance = 1;
	printf("1\n");

	//allocate inputs
	Int3 * seq1d, *seq1h;
	int* histogramOutput;
	cudaMalloc((void**)&seq1d, sizeof(Int3)*seqLen);
	cudaMallocHost((void**)&seq1h, sizeof(Int3)*seqLen);
	printf("2\n");

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
	stream_handler1(input, deletionsOutput, indexOutput, histogramOutput, outputLen, distance);
	printf("3\n");

	//expactation
	int expectedLen = 20;
	char expectedPairs[][5] = {
		"AAA", "CAA", "CAA", "CAA", "CAAA",
		"ADA", "CDA", "CAA", "CAD", "CADA",
		"AAA", "CAA", "CAA", "CAA", "CAAA",
		"DKD", "CKD", "CDD", "CDK", "CDKD",
	};
	int expectedIndex[] = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3};

	//check
	check(outputLen == expectedLen);
	for (int i = 0; i < expectedLen; i++) {
		printf("4\n");
		checkstr(expectedPairs[i], str_decode(deletionsOutput[i]));
		printf("5\n");
		check(expectedIndex[i] == indexOutput[i]);
	}
})