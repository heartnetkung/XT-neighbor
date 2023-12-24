#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include "xtn_inner.cu"

const size_t MAX_ARRAY_SIZE = INT_MAX >> 3;

size_t cal_chunksize1(int distance) {
	size_t ans = MAX_ARRAY_SIZE;
	for (int i = 0; i < distance; i++)
		ans /= MAX_INPUT_LENGTH;
	return ans;
}

void xtn_perform(XTNArgs args, Int3* seq1, void callback(XTNOutput)) {
	int distance = args.distance, verbose = args.verbose, seq1Len = args.seq1Len;
	int* deviceInt;
	cudaMalloc((void**)&deviceInt, sizeof(int));

	//=====================================
	// step 1: transfer input to GPU
	//=====================================
	Int3* seq1Device = host_to_device(seq1, seq1Len);
	size_t seq1ChunkSize = cal_chunksize1(distance);
	size_t seq1ChunkCount = divideCeil(seq1Len, seq1ChunkSize);
	GPUInputStream<Int3> seq1Stream(seq1Device, seq1Len, seq1ChunkSize);

	print_tp(verbose, "1", seq1Stream.get_throughput());

	//=====================================
	// step 2: generate deletion combinations
	//=====================================
	D2Stream<Int3> combKeyStream(chunkCount);
	D2Stream<int> combValueStream(chunkCount);

	Chunk<Int3> seq1Chunk, combKeyChunk;
	Chunk<int> combValueChunk;
	while ((seq1Chunk = seq1Stream.read()).not_null()) {
		//perform
		stream_handler1(seq1Chunk, combKeyChunk, combValueChunk, distance);

		// gen histogram

		//simple print
		print_int3(combKeyChunk.ptr, combKeyChunk.len, ' ');
		print_int_arr(combValueChunk.ptr, combValueChunk.len);

		//flush
		combKeyStream.write(combKeyChunk.ptr, combKeyChunk.len);
		combValueStream.write(combValueChunk.ptr, combValueChunk.len);
		_cudaFree(combKeyChunk.ptr, combValueChunk.ptr);
	}
	// sum histogram
	// size_t** combKeyOffset;
	// size_t combKeyOffsetLen;
	// combKeyStream.set_offsets(combKeyOffset, combKeyOffsetLen);
	// combKeyStream.set_offsets(combKeyOffset, combKeyOffsetLen);
	print_tp(verbose, "2", combKeyStream.get_throughput());

	//=====================================
	// step 3: cal offsets and histograms
	//=====================================


	//=====================================
	// step 4: generate pairs and postprocess
	//=====================================

	//=====================================
	// step 5: deallocate
	//=====================================


	//test file writing
	XTNOutput out1;
	out1.len = 2;
	out1.indexPairs = (Int2*)malloc(sizeof(Int2) * 2);
	out1.indexPairs[0] = {0, 0}; out1.indexPairs[1] = {0, 1};
	out1.pairwiseDistances = (char*)malloc(sizeof(char) * 2);
	out1.pairwiseDistances[0] = 0; out1.pairwiseDistances[1] = 1;
	callback(out1);

	out1.len = 1;
	out1.indexPairs = (Int2*)malloc(sizeof(Int2) * 1);
	out1.indexPairs[0] = {1, 0};
	out1.pairwiseDistances = (char*)malloc(sizeof(char) * 1);
	out1.pairwiseDistances[0] = 2;
	callback(out1);
}
