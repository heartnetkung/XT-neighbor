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
	Int3 * seq_d, *seq_h;
	Int2 * pairs_d, *pairs_h;
	std::vector<XTNOutput> allOutputs;
	int* deviceInt, *inputOffsets_d;
	SeqInfo* info_d;
	MemoryContext ctx;
	cudaMalloc(&deviceInt, sizeof(int));
	cudaMalloc(&seq_d, sizeof(Int3)*seqLen);
	cudaMallocHost(&seq_h, sizeof(Int3)*seqLen);
	cudaMalloc(&pairs_d, sizeof(Int2)*pairLen);
	cudaMallocHost(&pairs_h, sizeof(Int2)*pairLen);
	cudaMalloc(&info_d, sizeof(SeqInfo)*infoLen);
	ctx.bandwidth2 = 100;

	//make inputs
	for (int i = 0; i < seqLen; i++)
		seq_h[i] = str_encode(seqs[i]);
	int count = 0;
	for (int i = 0; i < seqLen; i++)
		for (int j = i + 1; j < seqLen; j++)
			pairs_h[count++] = {.x = i, .y = j};
	seq_d = host_to_device(seq_h, seqLen);
	pairs_d = host_to_device(pairs_h, pairLen);
	info_d = host_to_device(info_h, infoLen);
	inputOffsets_d = host_to_device(inputOffsets, seqLen);

	// do testing
	Chunk<Int2> pairInput;
	pairInput.ptr = pairs_d;
	pairInput.len = pairLen;
	stream_handler4_overlap(pairInput, allOutputs, seq_d, info_d, inputOffsets_d,
	                        seqLen, distance, LEVENSHTEIN, deviceInt, ctx);
	XTNOutput output = allOutputs.back();

	// checking
	int expectedLen = 3, expectedCount = 1;
	check(output.len == expectedLen);
	check(allOutputs.size() == expectedCount);

	Int2 expectedIndexPair[] = {{.x = 0, .y = 0}, {.x = 0, .y = 1}, {.x = 1, .y = 1}};
	check_device_arr(output.indexPairs, expectedIndexPair, output.len);

	size_t expectedFrequency[] = {24, 41, 70};
	check_device_arr(output.pairwiseFrequencies, expectedFrequency, output.len);
})