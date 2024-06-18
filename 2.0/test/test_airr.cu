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
	Int3 output1;
	int output2, retval;

	char input1[10] = "";
	retval = extract_data(input1, 3, 6, 8, 0, output1, output2, true);
	check(retval == ERROR);

	char input2[50] = "0\t1\t2\tCAAK\t4\t\t6\t7";
	retval = extract_data(input2, 3, 6, 8, 0, output1, output2, true);
	checkstr(str_decode(output1), "CAAK");
	check(output2 == 6);
	check(retval == SUCCESS);
	retval = extract_data(input2, 3, 6, 8, 0, output1, output2, false);
	checkstr(str_decode(output1), "CAAK");
	check(output2 == -1);
	check(retval == SUCCESS);
})
