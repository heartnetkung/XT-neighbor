#include "test_util.cu"
#include "../src/xtn_inner.cu"

void callback(Int2* pairOut, int len) {
	int expectedLen = 20;
	Int2 expectedPairs[] = {
		{.x = 0, .y = 2}, {.x = 0, .y = 0}, {.x = 0, .y = 0}, {.x = 0, .y = 1}, {.x = 0, .y = 2},
		{.x = 0, .y = 2}, {.x = 0, .y = 2}, {.x = 0, .y = 0}, {.x = 0, .y = 1}, {.x = 0, .y = 2},
		{.x = 0, .y = 2}, {.x = 0, .y = 2}, {.x = 0, .y = 1}, {.x = 0, .y = 2}, {.x = 0, .y = 2},
		{.x = 0, .y = 2}, {.x = 1, .y = 2}, {.x = 1, .y = 2}, {.x = 1, .y = 2}, {.x = 0, .y = 2}
	};
	Int2* pairOut2 = device_to_host(pairOut, len);

	check(len == expectedLen);
	printf("ee %d\n", len == expectedLen);
	for (int i = 0; i < expectedLen; i++) {
		check(pairOut2[i].x == expectedPairs[i].x);
		check(pairOut2[i].y == expectedPairs[i].y);
		printf("ff %d %d\n", pairOut2[i].x == expectedPairs[i].x, pairOut2[i].y == expectedPairs[i].y);
	}
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

	Int3* keyOut = device_to_host(keyIn.ptr, keyIn.len);
	int* valueOut = device_to_host(valueIn.ptr, valueIn.len);
	int* histogramOutput2 = device_to_host(histogramOutput[0], ctx.histogramSize);

	check(keyIn.len == expectedLen);
	check(valueIn.len == expectedLen);
	printf("aa %d %d\n", keyIn.len == expectedLen, valueIn.len == expectedLen);
	for (int i = 0; i < expectedLen; i++) {
		check(expectedValue[i] == valueOut[i]);
		printf("bb %d\n", expectedValue[i] == valueOut[i]);
		checkstr(expectedKey[i], str_decode(keyOut[i]));
		printf("dd %s %s\n", expectedKey[i], str_decode(keyOut[i]));
	}
	for (int i = 0; i < ctx.histogramSize; i++) {
		check(expectedHistogram[i] == histogramOutput2[i]);
		printf("cc %d\n", expectedHistogram[i] == histogramOutput2[i]);
	}
})