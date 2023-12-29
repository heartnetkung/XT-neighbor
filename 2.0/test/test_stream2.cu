#include "test_util.cu"
#include "../src/xtn_inner.cu"

TEST(Stream2, {
	int len = 20;
	char keys[][5] = {
		"AAA", "AAA", "ADA", "CAA", "CAA", "CAA", "CAA", "CAA", "CAA", "CAA",
		"CAAA", "CAAA", "CAD", "CADA", "CDA", "CDD", "CDK", "CDKD", "CKD", "DKD"
	};
	int values[] =  {0, 2, 1, 0, 0, 0, 1, 2, 2, 2, 0, 2, 1, 1, 1, 3, 3, 3, 3, 3};

	Int3* keysInt3 = (Int3*)malloc(sizeof(Int3) * len);
	for (int i = 0; i < len; i++)
		keysInt3[i] = str_encode(keys[i]);


	Chunk<Int3> keyIn = {.ptr = host_to_device(keysInt3, len), .len = len};

	printf("hello\n");
	// stream_handler2(Chunk<Int3> &keyInOut, Chunk<int> &valueInOut,
	//                 int* &histogramOutput, int distance, int seqLen, int* buffer)
})