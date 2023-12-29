#include "test_util.cu"
#include "../src/xtn_inner.cu"

TEST(Stream2, {
	int seqLen = 4, len = 20, distance = 1;
	char keys[][5] = {
		"AAA", "CAA", "CAA", "CAA", "CAAA", "ADA", "CDA", "CAA", "CAD", "CADA",
		"AAA", "CAA", "CAA", "CAA", "CAAA", "DKD", "CKD", "CDD", "CKD", "CDKD"
	};
	int values[] =  {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3};
	int* histogramOutput, *deviceInt;
	MemoryContext ctx;
	ctx.maxThroughput = 100;
	int* histogramOutputHost = (int*)calloc(ctx.histogramSize, sizeof(int));

	histogramOutput = host_to_device(histogramOutputHost, ctx.histogramSize);
	cudaMalloc(&deviceInt, sizeof(int));
	Int3* keysInt3 = (Int3*)malloc(sizeof(Int3) * len);
	for (int i = 0; i < len; i++)
		keysInt3[i] = str_encode(keys[i]);

	Chunk<Int3> keyInOut = {.ptr = host_to_device(keysInt3, len), .len = len};
	Chunk<int> valueInOut = {.ptr = host_to_device(values, len), .len = len};

	printf("hello\n");
	stream_handler2(keyInOut, valueInOut, histogramOutput,
	                distance, seqLen, deviceInt, ctx);

	print_int3_arr(keyInOut.ptr, keyInOut.len);
	print_int_arr(valueInOut.ptr, valueInOut.len);
	print_int_arr(histogramOutput, ctx.histogramSize);
	// void stream_handler2(Chunk<Int3> &keyInOut, Chunk<int> &valueInOut, int* &histogramOutput,
	//                  int distance, int seqLen, int memoryConstraint, int* buffer) {
})