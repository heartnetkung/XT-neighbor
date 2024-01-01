#include "test_util.cu"
#include "../src/xtn_inner.cu"

TEST(Stream2, {
	printf("1\n");
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
	printf("2\n");

	cudaMalloc(&deviceInt, sizeof(int));
	Int3* keysInt3 = (Int3*)malloc(sizeof(Int3) * len);
	for (int i = 0; i < len; i++)
		keysInt3[i] = str_encode(keys[i]);
	printf("3\n");

	Chunk<Int3> keyInOut = {.ptr = host_to_device(keysInt3, len), .len = len};
	Chunk<int> valueInOut = {.ptr = host_to_device(values, len), .len = len};
	stream_handler2(keyInOut, valueInOut, histogramOutput,
	                distance, seqLen, deviceInt, ctx);
	printf("4\n");

	int expectedLen = 20;
	char expectedPairs[][5] = {
		"AAA", "AAA", "ADA", "CAA", "CAA", "CAA", "CAA", "CAA", "CAA", "CAA",
		"CAAA", "CAAA", "CAD", "CADA", "CDA", "CDD", "CDKD", "CKD", "CKD", "DKD"
	};
	int expectedIndex[] = {0, 2, 1, 0, 0, 0, 1, 2, 2, 2, 0, 2, 1, 1, 1, 3, 3, 3, 3, 3};
	int expectedHistogram[] = {17, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0};
	printf("4.1\n");
	Int3* keyOut = device_to_host(keyInOut.ptr, keyInOut.len);
	printf("4.2\n");
	int* valueOut = device_to_host(valueInOut.ptr, valueInOut.len);
	printf("4.3\n");
	int* histogramOut = device_to_host(histogramOutput[0], ctx.histogramSize);
	printf("5\n");

	check(keyInOut.len == expectedLen);
	check(valueInOut.len == expectedLen);
	printf("6\n");
	for (int i = 0; i < expectedLen; i++) {
		checkstr(expectedPairs[i], str_decode(keyOut[i]));
		check(expectedIndex[i] == valueOut[i]);
	}
	printf("7\n");
	for (int i = 0; i < ctx.histogramSize; i++)
		check(expectedHistogram[i] == histogramOut[i]);
	printf("8\n");
})