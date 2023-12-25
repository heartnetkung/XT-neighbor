#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include "xtn_inner.cu"

const size_t MAX_ARRAY_SIZE = INT_MAX >> 3;

size_t cal_stream1_chunk_size(int distance) {
	size_t ans = MAX_ARRAY_SIZE;
	for (int i = 0; i < distance; i++)
		ans /= MAX_INPUT_LENGTH;
	return ans;
}

size_t cal_stream3_chunk_size() {

}

void xtn_perform(XTNArgs args, Int3* seq1, void callback(XTNOutput)) {
	int distance = args.distance, verbose = args.verbose, seq1Len = args.seq1Len;
	int* deviceInt;
	cudaMalloc((void**)&deviceInt, sizeof(int));
	printf("1\n");

	//=====================================
	// step 1: transfer input to GPU
	//=====================================
	Int3* seq1Device = host_to_device(seq1, seq1Len);
	size_t seq1ChunkSize = cal_stream1_chunk_size(distance);
	size_t seq1ChunkCount = divideCeil(seq1Len, seq1ChunkSize);
	GPUInputStream<Int3> seq1Stream(seq1Device, seq1Len, seq1ChunkSize);

	print_tp(verbose, "1", seq1Stream.get_throughput());
	printf("2\n");

	//=====================================
	// step 2: generate deletion combinations
	//=====================================
	D2Stream<Int3> combKeyStream(seq1ChunkCount);
	D2Stream<int> combValueStream(seq1ChunkCount);

	Chunk<Int3> seq1Chunk, combKeyChunk;
	Chunk<int> combValueChunk;
	printf("3\n");

	while ((seq1Chunk = seq1Stream.read()).not_null()) {
		//perform
		stream_handler1(seq1Chunk, combKeyChunk, combValueChunk, distance);

		// gen histogram

		//simple print
		//TODO device_to_host
		print_int3_arr(combKeyChunk.ptr, combKeyChunk.len);
		print_int_arr(combValueChunk.ptr, combValueChunk.len);

		//flush
		combKeyStream.write(combKeyChunk.ptr, combKeyChunk.len);
		combValueStream.write(combValueChunk.ptr, combValueChunk.len);
		_cudaFree(combKeyChunk.ptr, combValueChunk.ptr);
	}
	printf("4\n");

	// sum histogram
	size_t combOffset[][1] = {{combKeyChunk.len}}; //TODO
	size_t combOffsetLen = 1; //TODO
	combKeyStream.set_offsets(combOffset, combOffsetLen);
	combValueStream.set_offsets(combOffset, combOffsetLen);
	print_tp(verbose, "2", combKeyStream.get_throughput());
	printf("5\n");

	// //=====================================
	// // step 3: cal histograms and sort
	// //=====================================
	// int len = 1;
	// int len2[] = {9999};
	// Int3** keyOutArray = (Int3**)malloc(len * sizeof(Int3*));
	// int** valueOutArray = (int**)malloc(len * sizeof(int*));
	// keyOutArray[0] = (Int3*)malloc(len2[0] * sizeof(Int3));
	// valueOutArray[0] = (int*)malloc(len2[0] * sizeof(int));


	// RAMOutputStream<Int3> *keyOutStream = new RAMOutputStream<Int3>(keyOutArray, len, len2);
	// RAMOutputStream<int> *valueOutStream = new RAMOutputStream<int>(valueOutArray, len, len2);
	// int lowerbounds[] = {INT_MAX};
	// int lowerboundLen = 1;

	// while ( (combKeyChunk = combKeyStream.read()).not_null() ) {
	// 	combValueChunk = combValueStream.read();
	// 	sort_key_values(combKeyChunk.ptr, combValueChunk.ptr, combKeyChunk.len);
	// 	keyOutStream.write(combKeyChunk.ptr,combKeyChunk.len);
	// 	valueOutStream.write(combValueChunk.ptr,combValueChunk.len);
	// }


	// combKeyStream.deconstruct();
	// combValueStream.deconstruct();
	// print_tp(verbose, "3", keyOutStream.get_throughput());

	// //=====================================
	// // step 4: generate pairs and postprocess
	// //=====================================
	// // declare input
	// RAMInputStream<Int3> *keyInStream;
	// RAMInputStream<int> *valueInStream;


	// size_t maxReadableSize = INT_MAX >> 4;
	// Int3* keyInBuffer;
	// int* valueInBuffer;
	// cudaMalloc((void**)&keyInBuffer, sizeof(Int3)*maxReadableSize);
	// cudaMalloc((void**)&valueInBuffer, sizeof(int)*maxReadableSize);

	// for (int i = 0; i < lowerboundLen; i++) {
	// 	int lowerbound = lowerbounds[i];
	// 	size_t new_len = keyOutStream->get_new_len1();
	// 	size_t* new_len2 = keyOutStream->get_new_len2();
	// 	keyInStream = new RAMInputStream<int>(input , new_len, new_len2, maxReadableSize, keyInBuffer);
	// 	valueInStream = new RAMInputStream<int>(input , new_len, new_len2, maxReadableSize, valueInBuffer);
	// 	keyOutStream = new RAMOutputStream<int>(input, new_len, new_len2);
	// 	valueOutStream = new RAMOutputStream<int>(input, new_len, new_len2);

	// 	Chunk<Int3> keyInChunk, keyOutChunk;
	// 	Chunk<int> valueInChunk, valueOutChunk;
	// 	XTNOutput finalOutput;
	// 	while ((keyInChunk = keyInStream->read()).not_null()) {
	// 		valueInChunk = valueInStream->read();

	// 		stream_handler3(keyInChunk, valueInChunk, keyOutChunk, valueOutChunk,
	// 		                finalOutput, seq1Device, seq1Len, distance, lowerbound, deviceInt);

	// 		keyOutStream->write(keyOutChunk.ptr, keyOutChunk.len);
	// 		valueOutStream->write(valueOutChunk.ptr, valueOutChunk.len);
	// 		callback(finalOutput);
	// 		_free(finalOutput.indexPairs, finalOutput.pairwiseDistances);
	// 	}
	// }

	//=====================================
	// step 5: deallocate
	//=====================================

}
