#include <locale.h>
#include "test_util.cu"
#include "../src/util.cu"

TEST(memory, {
	setlocale(LC_ALL, "");
	print_gpu_memory();
	print_main_memory();
})