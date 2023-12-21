#include "test_util.cu"
#include "../src/brute_force.cu"

TEST(levenshtein, {
	char input1[] = "ABC";
	char input2[] = "AB";
	int output = 1;
	check(levenshtein(input1, input2) == output);
	check(levenshtein(input2, input1) == output);

	char input3[] = "ABCDE";
	char input4[] = "ABCDE";
	int output2 = 0;
	check(levenshtein(input3, input4) == output2);
	check(levenshtein(input4, input3) == output2);

	char input5[] = "ABC";
	char input6[] = "ACB";
	int output3 = 2;
	check(levenshtein(input5, input6) == output3);
	check(levenshtein(input6, input5) == output3);
})

TEST(check_intput, {
	int inputsLen = 5;
	char inputs[inputsLen][6] = {"AQCDE", "AQC", "AQ", "ACQ", "AQCDE"};
	int outputLen = 6;
	SymspellOutput output;

	Int3 inputs1Temp[inputsLen];
	for (int i = 0; i < inputsLen; i++)
		inputs1Temp[i] = str_encode(inputs[i]);

	auto answer = pairwise_distance(inputs1Temp, inputsLen, 2);
	check(answer.size() == outputLen); // number of pairs lte 2

	// not equal uninitialized inputs
	check(!check_intput(answer, output));

	Int2 indexPairs[outputLen];
	indexPairs[0].x = 0; indexPairs[0].y = 1;
	indexPairs[1].x = 0; indexPairs[1].y = 4;
	indexPairs[2].x = 1; indexPairs[2].y = 2;
	indexPairs[3].x = 1; indexPairs[3].y = 3;
	indexPairs[4].x = 1; indexPairs[4].y = 4;
	indexPairs[5].x = 2; indexPairs[5].y = 3;
	char pairwiseDistances[] = {2, 0, 1, 2, 2, 1};
	output.len = outputLen;
	output.indexPairs = indexPairs;
	output.pairwiseDistances = pairwiseDistances;

	// check pass
	check(check_intput(answer, output));

	//incorrect length
	output.len = 2;
	check(!check_intput(answer, output));
	output.len = outputLen;

	//incorrect distance
	indexPairs[0].x = 99;
	check(!check_intput(answer, output));
})
