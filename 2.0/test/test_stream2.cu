#include "test_util.cu"
#include "../src/xtn_inner.cu"

TEST(Stream2, {
	int seqLen = 4, len = 20, distance = 1, memoryConstraint = 100;
	char keys[][5] = {
		"AAA", "AAA", "ADA", "CAA", "CAA", "CAA", "CAA", "CAA", "CAA", "CAA",
		"CAAA", "CAAA", "CAD", "CADA", "CDA", "CDD", "CDK", "CDKD", "CKD", "DKD"
	};
	int values[] =  {0, 2, 1, 0, 0, 0, 1, 2, 2, 2, 0, 2, 1, 1, 1, 3, 3, 3, 3, 3};
	int* histogramOutput, *deviceInt;
	int* histogramOutputHost = (int*)calloc(HISTOGRAM_SIZE, sizeof(int));

	histogramOutput = host_to_device(histogramOutputHost, HISTOGRAM_SIZE);
	cudaMalloc(&deviceInt, sizeof(int));
	Int3* keysInt3 = (Int3*)malloc(sizeof(Int3) * len);
	for (int i = 0; i < len; i++)
		keysInt3[i] = str_encode(keys[i]);

	Chunk<Int3> keyInOut = {.ptr = host_to_device(keysInt3, len), .len = len};
	Chunk<int> valueInOut = {.ptr = host_to_device(values, len), .len = len};

	printf("hello\n");
	stream_handler2(keyInOut, valueInOut, histogramOutput,
	                distance, seqLen, memoryConstraint, deviceInt);

	print_int3_arr(keyInOut.ptr, keyInOut.len);
	print_int_arr(valueInOut.ptr, valueInOut.len);
	print_int_arr(histogramOutput, HISTOGRAM_SIZE);
	// void stream_handler2(Chunk<Int3> &keyInOut, Chunk<int> &valueInOut, int* &histogramOutput,
	//                  int distance, int seqLen, int memoryConstraint, int* buffer) {
})