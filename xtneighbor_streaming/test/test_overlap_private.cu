#include "test_util.cu"
#include "../src/xtn_overlap_inner.cu"

__global__
void do_eq(SeqInfo* left, SeqInfo* right, bool* out, int n) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid >= n)
		return;
	out[tid] = (left[tid] == right[tid]);
}

TEST(SeqInfoEquality, {
	//CAAA CAAD CAAA
	char allStr[] = "CAAACAADCAAA";
	unsigned int offsets[] = {0, 4, 8, 12};
	char* allStr_d = host_to_device(allStr, 12);
	unsigned int* offsets_d = host_to_device(offsets, 4);

	int infoLen = 6;
	SeqInfo info1 = {.frequency = 1, .repertoire = 0, .originalIndex = 0};
	SeqInfo info2 = {.frequency = 2, .repertoire = 1, .originalIndex = 1};
	SeqInfo info3 = {.frequency = 3, .repertoire = 2, .originalIndex = 2};
	SeqInfo lefts_h[] = {info1, info2, info3, info1, info2, info1};
	SeqInfo rights_h[] = {info1, info2, info3, info2, info3, info3};
	SeqInfo* lefts_d = host_to_device(lefts_h, infoLen);
	SeqInfo* rights_d = host_to_device(rights_h, infoLen);

	bool* output;
	cudaMallocHost(&output, sizeof(bool)*infoLen);
	_setGlobalVar <<< 1, 1>>>(allStr_d, offsets_d);
	do_eq <<< infoLen, 1>>>(lefts_d, rights_d, output, infoLen);

	bool expected[] = {true, true, true, false, false, true};
	check_device_arr(output, expected, infoLen);
})

TEST(DeduplicateFullLength, {
	int totalLen = 12;
	int seqLen = 3;
	char allStr[] = "CAAACAADCAAA";
	unsigned int offsets[] = {0, 4, 8, 12};
	SeqInfo info[] = {{.frequency = 1, .repertoire = 0, .originalIndex = 0},
		{.frequency = 2, .repertoire = 1, .originalIndex = 1},
		{.frequency = 3, .repertoire = 2, .originalIndex = 2}
	};
	int* buffer;
	int* infoLenOut;
	Int3* seqOut;
	cudaMalloc(&buffer, sizeof(int));

	//run
	char* allStr_d = host_to_device(allStr, totalLen);
	unsigned int* offsets_d = host_to_device(offsets, seqLen + 1);
	SeqInfo* info_d = host_to_device(info, seqLen);
	int uniqueLen = deduplicate_full_length(allStr_d, offsets_d, info_d, seqOut, infoLenOut , seqLen, buffer);

	//expected
	int expectedLen = 2;
	int expectedInfoLenOut[] = {2, 1};
	SeqInfo expectedInfo[] = {{.frequency = 1, .repertoire = 0, .originalIndex = 0},
		{.frequency = 3, .repertoire = 2, .originalIndex = 2},
		{.frequency = 2, .repertoire = 1, .originalIndex = 1}
	};
	Int3* seqOut_h = device_to_host(seqOut, uniqueLen);
	SeqInfo* info_h = device_to_host(info_d, seqLen);

	//check
	check(uniqueLen == expectedLen);
	check_device_arr(infoLenOut, expectedInfoLenOut, uniqueLen);
	checkstr(str_decode(seqOut_h[0]), "CAAA" );
	checkstr(str_decode(seqOut_h[1]), "CAAD" );

	//manual checking expectedInfo since equality testing doesn't work
	for (int i = 0; i < seqLen; i++) {
		check(info_h[i].frequency == expectedInfo[i].frequency);
		check(info_h[i].repertoire == expectedInfo[i].repertoire);
		check(info_h[i].originalIndex == expectedInfo[i].originalIndex);
	}
})