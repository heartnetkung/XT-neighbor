#include "test_util.cu"
#include "../src/generate_combination.cu"

TEST(test_generate_combination, {
	char input[] = "CDEFG";
	int distance = 2;
	int outputLen = 16;
	Int3 output[outputLen];
	char expected[outputLen][6] = {
		"EFG", "DFG", "DEG", "DEF",
		"DEFG", "CFG", "CEG", "CEF",
		"CEFG", "CDG", "CDF", "CDFG",
		"CDE", "CDEG", "CDEF", "CDEFG"
	};

	expand_keys(str_encode(input), distance, output, 0, outputLen);
	for (int i = 0; i < outputLen; i++)
		checkstr(str_decode(output[i]), expected[i]);
})

TEST(test_generate_combination2, {
	char input[] = "CD";
	int distance = 3;
	int outputLen = 4;
	Int3 output[outputLen];
	char expected[outputLen][3] = {
		"", "D", "C", "CD"
	};

	expand_keys(str_encode(input), distance, output, 0, outputLen);
	for (int i = 0; i < outputLen; i++)
		checkstr(str_decode(output[i]), expected[i]);
})