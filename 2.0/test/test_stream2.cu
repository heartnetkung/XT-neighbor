#include "test_util.cu"
#include "../src/xtn_inner.cu"

TEST(Stream2, {
	int seqLen = 4, len = 20, distance = 1;
	char keys[][5] = {
		"AAA", "CAA", "CAA", "CAA", "CAAA", "ADA", "CDA", "CAA", "CAD", "CADA",
		"AAA", "CAA", "CAA", "CAA", "CAAA", "DKD", "CKD", "CDD", "CKD", "CDKD"
	};
	int values[] =  {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3};
	int* deviceInt;
	std::vector<int*> histogramOutput;
	MemoryContext ctx;
	int* histogramOutputHost = (int*)calloc(ctx.histogramSize, sizeof(int));
	size_t dummy = 0;

	cudaMalloc(&deviceInt, sizeof(int));
	Int3* keysInt3 = (Int3*)malloc(sizeof(Int3) * len);
	for (int i = 0; i < len; i++)
		keysInt3[i] = str_encode(keys[i]);
	ctx.bandwidth2 = 100;

	Chunk<Int3> keyInOut = {.ptr = host_to_device(keysInt3, len), .len = len};
	Chunk<int> valueInOut = {.ptr = host_to_device(values, len), .len = len};
	stream_handler2(keyInOut, valueInOut, histogramOutput, dummy,
	                distance, seqLen, deviceInt, ctx);

	int expectedLen = 20;
	char expectedPairs[][5] = {
		"AAA", "AAA", "ADA", "CAA", "CAA", "CAA", "CAA", "CAA", "CAA", "CAA",
		"CAAA", "CAAA", "CAD", "CADA", "CDA", "CDD", "CDKD", "CKD", "CKD", "DKD"
	};
	int expectedIndex[] = {0, 2, 1, 0, 0, 0, 1, 2, 2, 2, 0, 2, 1, 1, 1, 3, 3, 3, 3, 3};
	int expectedHistogram[] = {17, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0};
	Int3* keyOut = device_to_host(keyInOut.ptr, keyInOut.len);

	check(keyInOut.len == expectedLen);
	check(valueInOut.len == expectedLen);
	for (int i = 0; i < expectedLen; i++)
		checkstr(expectedPairs[i], str_decode(keyOut[i]));
	check_device_arr(valueInOut.ptr, expectedIndex, valueInOut.len);
	check_device_arr(histogramOutput[0], expectedHistogram, ctx.histogramSize);
})