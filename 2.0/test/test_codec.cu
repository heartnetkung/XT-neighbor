#include <stdio.h>
#include "test_util.cu"
#include "../src/codec.cu"

TEST(char_encode, {
	check(char_encode('A') == 1 );
	check(char_encode('X') == -1 );
	check(char_encode('0') == -1 );
	check(char_encode('Y') == 25 );
})

TEST(str_encode, {
	char input[] = "A";
	Int3 output = str_encode(input);
	check(output.entry[0] == 0x08000000);
	check(output.entry[1] == 0);
	check(output.entry[2] == 0);

	char input2[] = "AC";
	checkstr(str_decode(str_encode(input2)), input2);
})

TEST(len_decode, {
	char input[] = "ACC";
	Int3 output = str_encode(input);
	check(len_decode(output) == 3);
})

TEST(remove_char, {
	char input[] = "ACDEFGHIKLMNPQRSTV";
	Int3 binForm = str_encode(input);

	checkstr(str_decode(remove_char(binForm, 0)), "CDEFGHIKLMNPQRSTV");
	checkstr(str_decode(remove_char(binForm, 1)), "ADEFGHIKLMNPQRSTV");
	checkstr(str_decode(remove_char(binForm, 2)), "ACEFGHIKLMNPQRSTV");
	checkstr(str_decode(remove_char(binForm, 3)), "ACDFGHIKLMNPQRSTV");
	checkstr(str_decode(remove_char(binForm, 4)), "ACDEGHIKLMNPQRSTV");

	checkstr(str_decode(remove_char(binForm, 5)), "ACDEFHIKLMNPQRSTV");
	checkstr(str_decode(remove_char(binForm, 6)), "ACDEFGIKLMNPQRSTV");
	checkstr(str_decode(remove_char(binForm, 7)), "ACDEFGHKLMNPQRSTV");
	checkstr(str_decode(remove_char(binForm, 8)), "ACDEFGHILMNPQRSTV");
	checkstr(str_decode(remove_char(binForm, 9)), "ACDEFGHIKMNPQRSTV");

	checkstr(str_decode(remove_char(binForm, 10)), "ACDEFGHIKLNPQRSTV");
	checkstr(str_decode(remove_char(binForm, 11)), "ACDEFGHIKLMPQRSTV");
	checkstr(str_decode(remove_char(binForm, 12)), "ACDEFGHIKLMNQRSTV");
	checkstr(str_decode(remove_char(binForm, 13)), "ACDEFGHIKLMNPRSTV");
	checkstr(str_decode(remove_char(binForm, 14)), "ACDEFGHIKLMNPQSTV");

	checkstr(str_decode(remove_char(binForm, 15)), "ACDEFGHIKLMNPQRTV");
	checkstr(str_decode(remove_char(binForm, 16)), "ACDEFGHIKLMNPQRSV");
	checkstr(str_decode(remove_char(binForm, 17)), "ACDEFGHIKLMNPQRST");

	char input2[] = "ACD";
	Int3 binForm2 = str_encode(input2);
	checkstr(str_decode(remove_char(binForm2, 0)), "CD");
	checkstr(str_decode(remove_char(binForm2, 1)), "AD");
	checkstr(str_decode(remove_char(binForm2, 2)), "AC");
})

TEST(seq_array, {
	char input1[] = "CAD\tDEF\2\n", input2[] = "CAAK\n", input3[] = "";
	char* result;
	int len;

	SeqArray* seqArr = new SeqArray(5);
	check(seqArr->append(input1) == SUCCESS);
	check(seqArr->append(input2) == SUCCESS);
	check(seqArr->append(input3) == SUCCESS);
	check(seqArr->getSize() == 2);
	seqArr->destroy();
})