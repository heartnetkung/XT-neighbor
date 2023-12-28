#include "test_util.cu"
#include "../src/xtn_inner.cu"

TEST(Stream4, {
	int seqLen = 5;
	char seqs[seqLen][6] = {"CAAA", "CADA", "CAAA", "CDKD", "CAAK"};
	int pairLen = 10;
	int distance = 1;

	//allocate inputs
	Int3 * seq1d, *seq1h;
	Int2 * pairs_d, *pairs_h;
	XTNOutput output;
	int* deviceInt;
	cudaMalloc((void**)&deviceInt, sizeof(int));
	cudaMalloc((void**)&seq1d, sizeof(Int3)*seqLen);
	cudaMallocHost((void**)&seq1h, sizeof(Int3)*seqLen);
	cudaMalloc((void**)&pairs_d, sizeof(Int2)*pairLen);
	cudaMallocHost((void**)&pairs_h, sizeof(Int2)*pairLen);

	//make inputs
	for (int i = 0; i < seqLen; i++)
		seq1h[i] = str_encode(seqs[i]);
	int count = 0;
	for (int i = 0; i < 5; i++)
		for (int j = i + 1; j < 5; j++) {
			pairs_h[count++] = {.x = i, .y = j};
		}
	seq1d = host_to_device(seq1h, seqLen);
	pairs_d = host_to_device(pairs_h, pairLen);

	Chunk<Int2> pairInput;
	pairInput.ptr = pairs_d;
	pairInput.len = pairLen;

	stream_handler4(pairInput, output, seq1d, seqLen, distance, deviceInt)
	print_int2_arr(output.indexPairs, output.len);
	print_char_arr(output.pairwiseDistances, output.len);
	printf("len: %d\n", output.len);
})