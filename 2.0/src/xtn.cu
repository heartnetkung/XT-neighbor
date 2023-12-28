#include <stdio.h>
#include <stdlib.h>
#include "xtn_inner.cu"

const size_t MAX_ARRAY_SIZE = INT_MAX >> 3;

size_t cal_stream1_chunk_size(int distance) {
	size_t ans = MAX_ARRAY_SIZE;
	for (int i = 0; i < distance; i++)
		ans /= MAX_INPUT_LENGTH;
	return ans;
}

// size_t cal_stream3_chunk_size() {

// }


void xtn_perform(XTNArgs args, Int3* seq1, void callback(XTNOutput)) {
	int distance = args.distance, verbose = args.verbose, seq1Len = args.seq1Len;
	int* deviceInt;
	cudaMalloc(&deviceInt, sizeof(int));

	//=====================================
	// step 1: transfer input to GPU
	//=====================================
	Int3* seq1Device = host_to_device(seq1, seq1Len);
	size_t seq1ChunkSize = cal_stream1_chunk_size(distance);
	size_t seq1ChunkCount = divide_ceil(seq1Len, seq1ChunkSize);
	GPUInputStream<Int3> seq1Stream(seq1Device, seq1Len, seq1ChunkSize);

	print_tp(verbose, "1", seq1Stream.get_throughput());

	//=====================================
	// step 2: generate deletion combinations
	//=====================================
	D2Stream<Int3> keyD2Stream(seq1ChunkCount);
	D2Stream<int> valueD2Stream(seq1ChunkCount);
	int* histogramOutput;

	Chunk<Int3> seq1Chunk, combKeyChunk;
	Chunk<int> combValueChunk;
	while ((seq1Chunk = seq1Stream.read()).not_null()) {
		//perform
		stream_handler1(seq1Chunk, combKeyChunk, combValueChunk, histogramOutput, distance);

		// gen histogram

		//simple print
		print_int3_arr(combKeyChunk.ptr, combKeyChunk.len);
		print_int_arr(combValueChunk.ptr, combValueChunk.len);

		//flush
		keyD2Stream.write(combKeyChunk.ptr, combKeyChunk.len);
		valueD2Stream.write(combValueChunk.ptr, combValueChunk.len);
		_cudaFree(combKeyChunk.ptr, combValueChunk.ptr);
	}
	printf("4\n");

	// sum histogram
	size_t** combOffset = (size_t**)malloc( sizeof(size_t*)); //TODO
	combOffset[0] = (size_t*)malloc(sizeof(size_t));
	combOffset[0][0] =  combKeyChunk.len;
	size_t combOffsetLen = 1; //TODO
	keyD2Stream.set_offsets(combOffset, combOffsetLen);
	valueD2Stream.set_offsets(combOffset, combOffsetLen);
	print_tp(verbose, "2", keyD2Stream.get_throughput());
	printf("5\n");

	//=====================================
	// step 3: cal histograms and sort
	//=====================================
	size_t len = 1;
	size_t* len2 = (size_t*)malloc(len * sizeof(size_t));
	len2[0] = 9999;
	Int3** keyOutArray = (Int3**)malloc(len * sizeof(Int3*));
	int** valueOutArray = (int**)malloc(len * sizeof(int*));
	keyOutArray[0] = (Int3*)malloc(len2[0] * sizeof(Int3));
	valueOutArray[0] = (int*)malloc(len2[0] * sizeof(int));
	printf("6\n");

	RAMOutputStream<Int3> *keyOutStream = new RAMOutputStream<Int3>(keyOutArray, len, len2);
	RAMOutputStream<int> *valueOutStream = new RAMOutputStream<int>(valueOutArray, len, len2);
	int lowerbounds[] = {INT_MAX};
	int lowerboundLen = 1;
	printf("7\n");

	while ( (combKeyChunk = keyD2Stream.read()).not_null() ) {
		combValueChunk = valueD2Stream.read();
		sort_key_values(combKeyChunk.ptr, combValueChunk.ptr, combKeyChunk.len);

		print_int3_arr(combKeyChunk.ptr, combKeyChunk.len);
		print_int_arr(combValueChunk.ptr, combValueChunk.len);

		keyOutStream->write(combKeyChunk.ptr, combKeyChunk.len);
		valueOutStream->write(combValueChunk.ptr, combValueChunk.len);
	}
	printf("8\n");

	keyD2Stream.deconstruct();
	valueD2Stream.deconstruct();
	print_tp(verbose, "3", keyOutStream->get_throughput());

	//=====================================
	// step 4: generate pairs and postprocess
	//=====================================
	// declare input
	RAMInputStream<Int3> *keyInStream;
	RAMInputStream<int> *valueInStream;
	printf("9\n");

	size_t maxReadableSize = INT_MAX >> 4;
	Int3* keyInBuffer;
	int* valueInBuffer;
	cudaMalloc(&keyInBuffer, sizeof(Int3)*maxReadableSize);
	cudaMalloc(&valueInBuffer, sizeof(int)*maxReadableSize);
	printf("10\n");

	for (int i = 0; i < lowerboundLen; i++) {
		int lowerbound = lowerbounds[i];
		size_t new_len = keyOutStream->get_new_len1();
		size_t* new_len2 = keyOutStream->get_new_len2();
		keyInStream = new RAMInputStream<Int3>(keyOutArray , new_len, new_len2, maxReadableSize, keyInBuffer);
		valueInStream = new RAMInputStream<int>(valueOutArray , new_len, new_len2, maxReadableSize, valueInBuffer);
		keyOutStream = new RAMOutputStream<Int3>(keyOutArray, new_len, new_len2);
		valueOutStream = new RAMOutputStream<int>(valueOutArray, new_len, new_len2);
		printf("11\n");

		Chunk<Int3> keyInChunk, keyOutChunk;
		Chunk<int> valueInChunk, valueOutChunk;
		XTNOutput finalOutput;
		while ((keyInChunk = keyInStream->read()).not_null()) {
			valueInChunk = valueInStream->read();
			printf("12\n");

			stream_handler3(keyInChunk, valueInChunk, keyOutChunk, valueOutChunk,
			                finalOutput, seq1Device, seq1Len, distance, lowerbound, deviceInt);
			printf("13\n");

			print_int3_arr(keyOutChunk.ptr, keyOutChunk.len);
			print_int_arr(valueOutChunk.ptr, valueOutChunk.len);

			keyOutStream->write(keyOutChunk.ptr, keyOutChunk.len);
			valueOutStream->write(valueOutChunk.ptr, valueOutChunk.len);

			callback(finalOutput);
			_cudaFreeHost(finalOutput.indexPairs, finalOutput.pairwiseDistances);
			printf("14\n");
		}
	}

	//=====================================
	// step 5: deallocate
	//=====================================

}
