#include "test_util.cu"
#include "../src/xtn_overlap_inner.cu"

TEST(SeqInfoEquality, {
	//CAAA CAAD CAAA
	char allStr[] = "CAAACAADCAAA";
	unsigned int offsets[] = {0, 4, 8, 12};
	SeqInfo info1 = {.frequency = 1, .repertoire = 0, .originalIndex = 0};
	SeqInfo info2 = {.frequency = 2, .repertoire = 1, .originalIndex = 1};
	SeqInfo info3 = {.frequency = 3, .repertoire = 2, .originalIndex = 2};

	_allStr = allStr;
	_allStrOffset = offsets;

	check(info1 == info1);
	check(info2 == info2);
	check(info3 == info3);
	check(info1 != info2);
	check(info2 != info3);
	check(info1 == info3);
})