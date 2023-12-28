#include "test_util.cu"
#include "../src/xtn_inner.cu"

TEST(Stream1, {
	int seqLen = 5;
	char seqs[seqLen][6] = {"CAAA", "CADA", "CAAA", "CDKD", "CAAK"};
	int distance = 1;

	//allocate inputs
	Int3 * seq1d, *seq1h;
	int* histogramOutput;
	cudaMalloc((void**)&seq1d, sizeof(Int3)*seqLen);
	cudaMallocHost((void**)&seq1h, sizeof(Int3)*seqLen);

	//make inputs
	for (int i = 0; i < seqLen; i++)
		seq1h[i] = str_encode(seqs[i]);
	seq1d = host_to_device(seq1h, seqLen);

	//do testing
	Chunk<Int3> input;
	input.ptr = seq1d;
	input.len = seqLen;
	Chunk<int> indexOutput;
	Chunk<Int3> deletionsOutput;
	stream_handler1(input, deletionsOutput, indexOutput, histogramOutput, distance);

	print_int_arr(indexOutput.ptr, indexOutput.len);
	print_int3_arr(deletionsOutput.ptr, deletionsOutput.len);

	// //expactation
	// int expectedLen = 5;
	// Int2 expectedPairs[] = {
	// 	{.x = 0, .y = 1}, {.x = 0, .y = 2}, {.x = 0, .y = 4}, {.x = 1, .y = 2}, {.x = 2, .y = 4}
	// };
	// char expectedDistances[] = {1, 0, 1, 1, 1};

	// //check
	// check(output.len == expectedLen);
	// for (int i = 0; i < expectedLen; i++) {
	// 	check(expectedPairs[i].x == output.indexPairs[i].x);
	// 	check(expectedPairs[i].y == output.indexPairs[i].y);
	// 	check(expectedDistances[i] == output.pairwiseDistances[i]);
	// }
})