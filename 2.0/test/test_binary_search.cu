#include "test_util.cu"
#include "../src/kernel.cu"
// #include "../src/xtn_inner.cu"

TEST(BinarySearch, {
	int data[] = {2, 4, 6};
	int len = 3, input, expect;

	input = 0, expect = 0;
	check(binarySearch(input, data, len) == expect);
	input = 1, expect = 0;
	check(binarySearch(input, data, len) == expect);
	input = 2, expect = 0;
	check(binarySearch(input, data, len) == expect);
	input = 3, expect = 1;
	check(binarySearch(input, data, len) == expect);
	input = 4, expect = 1;
	check(binarySearch(input, data, len) == expect);
	input = 5, expect = 2;
	check(binarySearch(input, data, len) == expect);
	input = 6, expect = 2;
	check(binarySearch(input, data, len) == expect);
	input = 7, expect = 3;
	check(binarySearch(input, data, len) == expect);
})