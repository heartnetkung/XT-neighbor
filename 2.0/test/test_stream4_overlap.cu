#include "test_util.cu"
#include "../src/xtn_inner.cu"

TEST(Stream4Overlap, {
	int seqLen = 5;
	char seqs[seqLen][6] = {"CAAA", "CADA", "CAAA", "CDKD", "CAAK"};
	int freqs[6] = {3, 4, 5, 6, 7};
	int reps[1] = {5}, repCount = 1;
	int pairLen = 10;
	int distance = 1;

	//allocate inputs
	Int3 * seq1d, *seq1h;
	Int2 * pairs_d, *pairs_h;
	XTNOutput output;
	int* deviceInt, *freqs_d, *reps_d;
	cudaMalloc(&deviceInt, sizeof(int));
	cudaMalloc(&seq1d, sizeof(Int3)*seqLen);
	cudaMallocHost(&seq1h, sizeof(Int3)*seqLen);
	cudaMalloc(&pairs_d, sizeof(Int2)*pairLen);
	cudaMalloc(&freqs_d, sizeof(int)*seqLen);
	cudaMalloc(&reps_d, sizeof(int)*repCount);
	cudaMallocHost(&pairs_h, sizeof(Int2)*pairLen);

	//make inputs
	for (int i = 0; i < seqLen; i++)
		seq1h[i] = str_encode(seqs[i]);
	int count = 0;
	for (int i = 0; i < 5; i++)
		for (int j = i + 1; j < 5; j++)
			pairs_h[count++] = {.x = i, .y = j};
	seq1d = host_to_device(seq1h, seqLen);
	pairs_d = host_to_device(pairs_h, pairLen);
	freqs_d = host_to_device(freqs, seqLen);
	reps_d = host_to_device(reps, repCount);

	//do testing
	Chunk<Int2> pairInput;
	pairInput.ptr = pairs_d;
	pairInput.len = pairLen;
	stream_handler4_overlap(pairInput, output, seq1d, freqs_d, reps_d, repCount,
	                        seqLen, distance, LEVENSHTEIN, deviceInt);

	//expactation
	int expectedLen = 1;
	Int2 expectedPairs[] = {
		{.x = 0, .y = 0}
	};
	size_t expectedDistances[] = {103};
	output.indexPairs = device_to_host(output.indexPairs, output.len);
	output.pairwiseFrequencies = device_to_host(output.pairwiseFrequencies, output.len);

	//check
	check(output.len == expectedLen);
	printf("a %d %d \n", output.len, expectedLen);
	for (int i = 0; i < expectedLen; i++) {
		check(expectedPairs[i].x == output.indexPairs[i].x);
		printf("b %d %d \n", expectedPairs[i].x, output.indexPairs[i].x);
		check(expectedPairs[i].y == output.indexPairs[i].y);
		printf("c %d %d \n", expectedPairs[i].y, output.indexPairs[i].y);
		check(expectedDistances[i] == output.pairwiseFrequencies[i]);
		printf("d %d %d \n", expectedDistances[i], output.pairwiseFrequencies[i]);
	}
})