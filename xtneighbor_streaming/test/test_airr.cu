#include "test_util.cu"
#include "../src/airr.cu"

TEST(find_header, {
	int output1, output2, output3;

	char input1[10] = "";
	output3 = find_header(input1, output1, output2, true);
	check(output1 == -1);
	check(output2 == -1);
	check(output3 == 1);

	char input2[50] = "aa\tbb\tcdr3\tcdr3_aa\tcc\t\tduplicate_count\tcdr3_aa";
	output3 = find_header(input2, output1, output2, true);
	check(output1 == 3);
	check(output2 == 6);
	check(output3 == 8);
	output3 = find_header(input2, output1, output2, false);
	check(output1 == 3);
	check(output2 == -1);
	check(output3 == 8);
})

TEST(extract_data, {
	int output, retval, len;
	SeqArray* seqArr = new SeqArray(2);
	char* result;
	char expectStr[] = "CAAK";

	char input1[10] = "";
	retval = extract_data(input1, 3, 6, 8, 0, seqArr, output, true);
	check(retval == ERROR);
	check(seqArr->getSize() == 0);

	char input2[50] = "0\t1\t2\tCAAK\t4\t\t6\t7";
	retval = extract_data(input2, 3, 6, 8, 0, seqArr, output, true);
	len = seqArr->getItemCPU(0, result);
	for (int i = 0; i < len; i++)
		check(expectStr[i] == result[i]);
	check(output == 6);
	check(retval == SUCCESS);
	check(seqArr->getSize() == 1);
	retval = extract_data(input2, 3, 6, 8, 0, seqArr, output, false);
	len = seqArr->getItemCPU(0, result);
	for (int i = 0; i < len; i++)
		check(expectStr[i] == result[i]);
	check(output == -1);
	check(retval == SUCCESS);
	check(seqArr->getSize() == 2);
})
