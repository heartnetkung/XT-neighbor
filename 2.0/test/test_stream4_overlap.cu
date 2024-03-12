#include "test_util.cu"
#include "../src/xtn_inner.cu"

TEST(Stream4Overlap, {
	int seqLen = 4;
	int infoLen = 5;
	// original sequence char seqs[seqLen][6] = {"CAAA", "CADA", "CAAA", "CDKD", "CAAK"};
	char seqs[seqLen][6] = {"CAAA", "CADA", "CDKD", "CAAK"};
	SeqInfo info_h[] = {
		{.frequency = 3, .repertoire = 0}, {.frequency = 5, .repertoire = 1},
		{.frequency = 4, .repertoire = 0}, {.frequency = 6, .repertoire = 1}, {.frequency = 7, .repertoire = 1}
	};
	int inputOffsets[] = {2, 3, 4, 5};
	int pairLen = 6;
	int distance = 1;

	//allocate inputs
	Int3 * seq1_d, *seq1_h;
	Int2 * pairs_d, *pairs_h;
	XTNOutput output;
	int* deviceInt, *inputOffsets_d;
	SeqInfo* info_d;
	cudaMalloc(&deviceInt, sizeof(int));
	cudaMalloc(&seq1_d, sizeof(Int3)*seqLen);
	cudaMallocHost(&seq1_h, sizeof(Int3)*seqLen);
	cudaMalloc(&pairs_d, sizeof(Int2)*pairLen);
	cudaMallocHost(&pairs_h, sizeof(Int2)*pairLen);
	cudaMalloc(&info_d, sizeof(SeqInfo)*infoLen);

	//make inputs
	for (int i = 0; i < seqLen; i++)
		seq1_h[i] = str_encode(seqs[i]);
	int count = 0;
	for (int i = 0; i < seqLen; i++)
		for (int j = i + 1; j < seqLen; j++)
			pairs_h[count++] = {.x = i, .y = j};
	seq1_d = host_to_device(seq1_h, seqLen);
	pairs_d = host_to_device(pairs_h, pairLen);
	info_d = host_to_device(info_h, infoLen);
	inputOffsets_d = host_to_device(inputOffsets, seqLen);

	//do testing
	Chunk<Int2> pairInput;
	pairInput.ptr = pairs_d;
	pairInput.len = pairLen;
	stream_handler4_overlap(pairInput, output, seq1_d, info_d, inputOffsets_d,
	                        seqLen, distance, LEVENSHTEIN, deviceInt);

	print_int2_arr(output.indexPairs, output.len);
	print_size_t_arr(output.pairwiseFrequencies, output.len);
	// //expactation
	// int expectedLen = 1;
	// Int2 expectedPairs[] = {
	// 	{.x = 0, .y = 0}
	// };
	// size_t expectedDistances[] = {206};
	// output.indexPairs = device_to_host(output.indexPairs, output.len);
	// output.pairwiseFrequencies = device_to_host(output.pairwiseFrequencies, output.len);

	// //check
	// check(output.len == expectedLen);
	// for (int i = 0; i < expectedLen; i++) {
	// 	check(expectedPairs[i].x == output.indexPairs[i].x);
	// 	check(expectedPairs[i].y == output.indexPairs[i].y);
	// 	check(expectedDistances[i] == output.pairwiseFrequencies[i]);
	// }
})