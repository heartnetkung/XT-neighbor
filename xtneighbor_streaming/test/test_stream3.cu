#include "test_util.cu"
#include "../src/xtn_inner.cu"

void callback(Int2* pairOut, int len) {
	int expectedLen = 20;
	Int2 expectedPairs[] = {
		{.x = 0, .y = 0}, {.x = 0, .y = 0}, {.x = 0, .y = 0},
		{.x = 0, .y = 1}, {.x = 0, .y = 1}, {.x = 0, .y = 1},
		{.x = 0, .y = 2}, {.x = 0, .y = 2}, {.x = 0, .y = 2},
		{.x = 0, .y = 2}, {.x = 0, .y = 2}, {.x = 0, .y = 2},
		{.x = 0, .y = 2}, {.x = 0, .y = 2}, {.x = 0, .y = 2},
		{.x = 0, .y = 2}, {.x = 0, .y = 2},
		{.x = 1, .y = 2}, {.x = 1, .y = 2}, {.x = 1, .y = 2}
	};
	check(len == expectedLen);
	check_device_arr(pairOut, expectedPairs, len);
}

TEST(Stream3, {
	MemoryContext ctx;
	int seqLen = 4, len = 20, lowerbound = 1;
	char pairCharInput[][5] = {
		"AAA", "AAA", "ADA", "CAA", "CAA", "CAA", "CAA", "CAA", "CAA", "CAA",
		"CAAA", "CAAA", "CAD", "CADA", "CDA", "CDD", "CDKD", "CKD", "CKD", "DKD"
	};
	int indexInput[] = {0, 2, 1, 0, 0, 0, 1, 2, 2, 2, 0, 2, 1, 1, 1, 3, 3, 3, 3, 3};
	Int3* pairInput = (Int3*)malloc(sizeof(Int3) * len);
	int* deviceInt;
	std::vector<int*> histogramOutput;

	for (int i = 0; i < len; i++)
		pairInput[i] = str_encode(pairCharInput[i]);
	cudaMalloc(&deviceInt, sizeof(int));
	Chunk<Int3> keyIn = {.ptr = host_to_device(pairInput, len), .len = len};
	Chunk<int> valueIn = {.ptr = host_to_device(indexInput, len), .len = len};
	ctx.bandwidth2 = 100;

	stream_handler3(keyIn, valueIn, callback,
	                histogramOutput, lowerbound, seqLen, deviceInt, ctx);

	int expectedLen = 5;
	int expectedHistogram[] = {17, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	char expectedKey[][5] = {"CAA", "CAA", "CAA", "CKD", "CKD"};
	int expectedValue[] = {2, 2, 2, 3, 3};

	check(keyIn.len == expectedLen);
	check(valueIn.len == expectedLen);
	check_device_arr(valueIn.ptr, expectedValue, valueIn.len);
	check_device_arr(histogramOutput[0], expectedHistogram, ctx.histogramSize);

	Int3* keyOut = device_to_host(keyIn.ptr, keyIn.len);
	for (int i = 0; i < expectedLen; i++)
		checkstr(expectedKey[i], str_decode(keyOut[i]));
})