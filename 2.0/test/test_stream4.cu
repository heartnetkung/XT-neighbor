#include "test_util.cu"
#include "../src/xtn_inner.cu"

TEST(Stream4, {
	int seqLen = 5;
	char seqs[seqLen][6] = {"CAAA", "CADA", "CAAA", "CDKD", "CAAK"};
	int pairLen = 10;
	int distance = 1;

	//allocate inputs
	Int3 * seqd, *seqh;
	Int2 * pairs_d, *pairs_h;
	XTNOutput output;
	int* deviceInt;
	cudaMalloc(&deviceInt, sizeof(int));
	cudaMalloc(&seqd, sizeof(Int3)*seqLen);
	cudaMallocHost(&seqh, sizeof(Int3)*seqLen);
	cudaMalloc(&pairs_d, sizeof(Int2)*pairLen);
	cudaMallocHost(&pairs_h, sizeof(Int2)*pairLen);

	//make inputs
	for (int i = 0; i < seqLen; i++)
		seqh[i] = str_encode(seqs[i]);
	int count = 0;
	for (int i = 0; i < 5; i++)
		for (int j = i + 1; j < 5; j++)
			pairs_h[count++] = {.x = i, .y = j};
	seqd = host_to_device(seqh, seqLen);
	pairs_d = host_to_device(pairs_h, pairLen);

	//do testing
	Chunk<Int2> pairInput;
	pairInput.ptr = pairs_d;
	pairInput.len = pairLen;
	stream_handler4_nn(pairInput, output, seqd, seqLen, distance, LEVENSHTEIN, deviceInt);

	//expactation
	int expectedLen = 5;
	Int2 expectedPairs[] = {
		{.x = 0, .y = 1}, {.x = 0, .y = 2}, {.x = 0, .y = 4}, {.x = 1, .y = 2}, {.x = 2, .y = 4}
	};
	char expectedDistances[] = {1, 0, 1, 1, 1};

	//check
	check(output.len == expectedLen);
	check_arr(output.indexPairs, expectedPairs, output.len);
	check_arr(output.pairwiseDistances, expectedDistances, output.len);
})