#include "test_util.cu"
#include "../src/xtn_inner.cu"

TEST(Stream3, {
	MemoryContext ctx;
	int seqLen = 4, len = 20, lowerbound = 1;
	char pairCharInput[][5] = {
		"AAA", "AAA", "ADA", "CAA", "CAA", "CAA", "CAA", "CAA", "CAA", "CAA",
		"CAAA", "CAAA", "CAD", "CADA", "CDA", "CDD", "CDKD", "CKD", "CKD", "DKD"
	};
	int indexInput[] = {0, 2, 1, 0, 0, 0, 1, 2, 2, 2, 0, 2, 1, 1, 1, 3, 3, 3, 3, 3};
	Int3* pairInput = (Int3*)malloc(sizeof(Int3) * len);
	int* deviceInt, *histogramOutput;
	int* histogramOutputHost = (int*)calloc(ctx.histogramSize, sizeof(int));

	for (int i = 0; i < len; i++)
		pairInput[i] = str_encode(pairCharInput[i]);
	cudaMalloc(&deviceInt, sizeof(int));
	Chunk<Int3> keyIn = {.ptr = host_to_device(pairInput, len), .len = len};
	Chunk<int> valueIn = {.ptr = host_to_device(indexInput, len), .len = len};
	histogramOutput = host_to_device(histogramOutputHost, ctx.histogramSize);
	D2Stream<Int2> stream(1);

	stream_handler3(keyIn, valueIn, stream,
	                histogramOutput, lowerbound, seqLen, deviceInt, ctx);

	print_int3_arr(keyIn.ptr, keyIn.len);
	print_int_arr(valueIn.ptr, valueIn.len);
	print_int_arr(histogramOutput, ctx.histogramSize);
})