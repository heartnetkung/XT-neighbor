#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "xtn_inner.cu"

D2Stream<Int2> *b3 = NULL; /*global variable for callback*/

//=====================================
// Private Functions
//=====================================

MemoryContext cal_memory_stream1(int distance) {
	MemoryContext ans;
	ans.gpuSize = get_gpu_memory();

	int deletionMultiplier = (distance == 1) ? (18 + 1) : (153 + 18 + 1);
	int multiplier = sizeof(int) + //input
	                 sizeof(int) + //int *combinationOffsets
	                 deletionMultiplier * (sizeof(Int3) + sizeof(int)); //Int3* &deletionsOutput int* &indexOutput
	ans.chunkSize = (ans.gpuSize) / (2 * multiplier);
	ans.chunkCount = (ans.gpuSize + 2 * multiplier - 1) / (2 * multiplier);
	return ans;
}

MemoryContext cal_memory_stream2() {
	MemoryContext ans;
	ans.gpuSize = get_gpu_memory();
	// ans.maxThroughput;//TODO
	// ans.maxThroughputExponent;//TODO
	return ans;
}

MemoryContext cal_memory_stream3() {
	MemoryContext ans;
	ans.gpuSize = get_gpu_memory();
	// ans.maxThroughput;//TODO
	return ans;
}

MemoryContext cal_memory_stream4() {
	MemoryContext ans;
	ans.gpuSize = get_gpu_memory();

	// memory usage is from
	// Int2* uniquePairs; char* uniqueDistances, *flags;
	int multiplier = sizeof(Int2) + //input
	                 sizeof(Int2) + //Int2* uniquePairs
	                 2 * sizeof(char) + //char* uniqueDistances, *flags
	                 sizeof(Int2) + //Int2* &pairOutput
	                 sizeof(char);// char* &distanceOutput
	ans.chunkSize = (ans.gpuSize) / (2 * multiplier);
	ans.chunkCount = (ans.gpuSize + 2 * multiplier - 1) / (2 * multiplier);
	// ans.maxThroughputExponent;//TODO
	return ans;
}

void cal_lowerbounds(int* &lowerbounds, int &lbLen) {
	lbLen = 1;
	cudaMallocHost(&lowerbounds, sizeof(int)*lbLen);
	lowerbounds[0] = 999999;
}

void write_b3(Int2* pairOutput, int pairLen) {
	b3->write(pairOutput, pairLen);
}

int* concat_clear_histograms(std::vector<int*> histograms, MemoryContext ctx) {
	int* ans, *ansPtr;
	int len = histograms.size();
	int memsize = sizeof(int*) * ctx.histogramSize;
	cudaMalloc(&ans, sizeof(int)*len * ctx.histogramSize);
	ansPtr = ans;

	for (int* histogram : histograms) {
		cudaMemcpy(ansPtr, histogram, memsize, cudaMemcpyDeviceToDevice);
		cudaFree(histogram);
		ans += ctx.histogramSize;
	}
	histograms.clear();
	return ans;
}

template <typename T1, typename T2>
int** set_d2_offsets(std::vector<int*> histograms, D2Stream<T1> s1, D2Stream<T2> s2,
                     int* buffer, MemoryContext ctx) {
	int** offsets;
	int* fullHistograms;
	int len, offsetLen;

	len = histograms.size();
	fullHistograms = concat_clear_histograms(histograms, ctx);
	offsetLen = solve_bin_packing(fullHistograms, offsets, len, ctx.histogramSize, buffer, ctx);
	s1->set_offsets(offsets, offsetLen);
	if (s2 != NULL)
		s2->set_offsets(offsets, offsetLen);

	cudaFree(fullHistograms);
	return offsets;
}

//=====================================
// Public Functions
//=====================================

void xtn_perform(XTNArgs args, Int3* seq1, void callback(XTNOutput)) {

	int* deviceInt, *lowerbounds;
	Int3* seq1Device;
	MemoryContext ctx0;
	std::vector<int*> histograms;
	int** offsets;
	int lowerboundsLen;
	int distance = args.distance, verbose = args.verbose, seq1Len = args.seq1Len;
	printf("1\n");

	GPUInputStream<Int3> *b0;
	D2Stream<Int3> *b1key;
	D2Stream<int> *b1value;
	RAMInputStream<Int3> *b2keyInput;
	RAMInputStream<int> *b2valueInput;
	RAMOutputStream<Int3> *b2keyOutput;
	RAMOutputStream<int> *b2valueOutput;
	Chunk<Int3> b0Chunk, b1keyChunk, b2keyChunk;
	Chunk<Int2> b3Chunk;
	Chunk<int> b1valueChunk, b2valueChunk;
	Int3* b1keyOut;
	int* b1valueOut;
	printf("2\n");

	cudaMalloc(&deviceInt, sizeof(int)); gpuerr();
	seq1Device = host_to_device(seq1, seq1Len); gpuerr();
	printf("3\n");

	//=====================================
	// stream 1: generate deletions
	//=====================================

	MemoryContext ctx1 = cal_memory_stream1(distance);
	int outputLen;
	printf("%lu %d %d\n", ctx1.gpuSize, ctx1.chunkCount, ctx1.chunkSize);

	b0 = new GPUInputStream<Int3>(seq1Device, seq1Len, ctx1.chunkSize);
	b1key = new D2Stream<Int3>(ctx1.chunkCount);
	b1value = new D2Stream<int>(ctx1.chunkCount);
	printf("4\n");

	while ((b0Chunk = b0->read()).not_null()) {
		stream_handler1(b0Chunk, b1keyOut, b1valueOut, histograms,
		                outputLen, distance, ctx1);
		b1key->write(b1keyOut, outputLen);
		b1value->write(b1valueOut, outputLen);
		_cudaFree(b1keyOut, b1valueOut); gpuerr();
	}

	printf("5\n");


	// //=====================================
	// // stream 2: group key values
	// //=====================================

	// MemoryContext ctx2 = cal_memory_stream2();

	// offsets = set_d2_offsets(histograms, b1key, b1value, deviceInt, ctx2);
	// b2keyOutput = new RAMOutputStream<Int3>();//(input, len, len2);
	// b2valueOutput = new RAMOutputStream<int>();//(input, len, len2);

	// while ((b1keyChunk = b1key->read()).not_null()) {
	// 	b1valueChunk = b1value->read();
	// 	stream_handler2(b1keyChunk, b1valueChunk, histograms,
	// 	                distance, seq1Len, deviceInt, ctx2);
	// 	b2keyOutput->write(b1keyChunk, b1keyChunk.len);
	// 	b2valueOutput->write(b1valueChunk, b1valueChunk.len);
	// 	_cudaFree(b1keyOut, b1valueOut); gpuerr();
	// }

	// b1key->deconstruct();
	// b1value->deconstruct();
	// _cudaFreeHost2D(offsets);

	// //=====================================
	// // loop: lower bound
	// //=====================================

	// cal_lowerbounds(lowerbounds, lowerboundsLen);
	// for (int i = 0; i < lowerboundsLen; i++) {
	// 	int lowerbound = lowerbounds[i];

	// 	//=====================================
	// 	// stream 3: generate pairs
	// 	//=====================================

	// 	MemoryContext ctx3 = cal_memory_stream3();

	// 	b3 = new D2Stream<Int2>(int len1); //TODO
	// 	b2keyInput = new RAMInputStream<Int3>();//input , len, len2, maxReadableSize, deviceBuffer
	// 	b2valueInput = new RAMInputStream<int>();//input , len, len2, maxReadableSize, deviceBuffer
	// 	b2keyOutput = new RAMOutputStream<Int3>();//(input, len, len2);
	// 	b2valueOutput = new RAMOutputStream<int>();//(input, len, len2);

	// 	while ((b2keyChunk = b2keyInput->read()).not_null()) {
	// 		b2valueChunk = b2valueInput->read();
	// 		stream_handler3(b2keyChunk, b2valueChunk, write_b3, histograms,
	// 		                lowerbound, seq1Len, deviceInt, ctx3);
	// 		b2keyOutput->write(b2keyChunk, b2keyChunk.len);
	// 		b2valueOutput->write(b2valueChunk, b2valueChunk.len);
	// 	}

	// 	fullHistograms = concat_clear_histograms(histograms, ctx0);
	// 	cudaFree(fullHistograms);

	// 	//=====================================
	// 	// stream 4: postprocessing
	// 	//=====================================

	// 	MemoryContext ctx4 = cal_memory_stream4();

	// 	offsets = set_d2_offsets(histograms, b3, NULL, deviceInt, ctx4);
	// 	XTNOutput finalOutput;

	// 	while ((b3Chunk = b3->read()).not_null()) {
	// 		stream_handler4(b3Chunk, finalOutput, seq1Device, seq1Len, distance, deviceInt);
	// 		callback(finalOutput);
	// 	}

	// 	b3->deconstruct();
	// 	_cudaFreeHost2D(offsets);
	// }

	// //=====================================
	// // boilerplate: deallocalte
	// //=====================================
	// cudaFreeHost(lowerbounds); gpuerr();
	// _cudaFree(deviceInt, seq1Device); gpuerr();

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