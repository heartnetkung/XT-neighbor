#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "xtn_inner.cu"

D2Stream<Int2> *b3 = NULL; /*global variable for callback*/

//=====================================
// Private Memory Functions
//=====================================

void cal_bandwidth_stream1(MemoryContext &ctx, int distance) {
	ctx.gpuSize = get_gpu_memory();
	int deletionMultiplier = (distance == 1) ? (18 + 1) : (153 + 18 + 1);
	int multiplier =
	    sizeof(int) + //input
	    sizeof(int) + //int *combinationOffsets
	    //Int3* &deletionsOutput int* &indexOutput unsigned int *histogramValue;
	    deletionMultiplier * (sizeof(Int3) + 2 * sizeof(int));
	ctx.bandwidth1 = (7 * ctx.gpuSize) / (10 * multiplier);
}

void cal_bandwidth_stream2(MemoryContext &ctx) {
	ctx.gpuSize = get_gpu_memory();
	int multiplier =
	    sizeof(Int3) + sizeof(int) + //input
	    2 * sizeof(int); // int* &inputOffsets, int* &outputLengths
	ctx.bandwidth1 = ctx.gpuSize / (20 * multiplier);
	ctx.bandwidth2 = ctx.bandwidth1 * 13;
}

void cal_bandwidth_stream3(MemoryContext &ctx) {
	ctx.gpuSize = get_gpu_memory();
	int multiplier =
	    sizeof(Int3) + sizeof(int) + //input
	    2 * sizeof(int) + // int* &inputOffsets, int* &outputLengths
	    sizeof(char) + sizeof(Int3) + sizeof(int); //char* flags Int3* keyOut int* valueOut;
	ctx.bandwidth1 = ctx.gpuSize / (10 * multiplier);
	ctx.bandwidth2 = ctx.bandwidth1 * 6;
}

void cal_bandwidth_stream4(MemoryContext &ctx) {
	ctx.gpuSize = get_gpu_memory();
	int multiplier =
	    sizeof(Int2) + //input
	    sizeof(Int2) + //Int2* uniquePairs
	    2 * sizeof(char) + //char* uniqueDistances, *flags
	    sizeof(Int2) + //Int2* &pairOutput
	    sizeof(char);// char* &distanceOutput
	ctx.bandwidth1 = (7 * ctx.gpuSize) / (10 * multiplier);
}

MemoryContext cal_memory_stream1(int seq1Len, int distance) {
	MemoryContext ans;
	cal_bandwidth_stream1(ans, distance);
	ans.chunkSize = (seq1Len < ans.bandwidth1) ? seq1Len : ans.bandwidth1;
	return ans;
}

MemoryContext cal_memory_stream2() {
	MemoryContext ans;
	cal_bandwidth_stream2(ans);
	return ans;
}

MemoryContext cal_memory_stream3() {
	MemoryContext ans;
	cal_bandwidth_stream3(ans);
	return ans;
}

MemoryContext cal_memory_stream4() {
	MemoryContext ans;
	cal_bandwidth_stream4(ans);
	return ans;
}

//=====================================
// Other Private Functions
//=====================================

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
	int memsize = sizeof(int) * ctx.histogramSize;
	cudaMalloc(&ans, sizeof(int)*len * ctx.histogramSize); gpuerr();
	ansPtr = ans;

	for (int* histogram : histograms) {
		cudaMemcpy(ansPtr, histogram, memsize, cudaMemcpyDeviceToDevice); gpuerr();
		cudaFree(histogram); gpuerr();
		ansPtr += ctx.histogramSize;
	}
	histograms.clear();
	return ans;
}

// black magic way to calculate floor(log2(n))
int cal_max_exponent(unsigned int intput) {
	int ans = 0;
	while (input >>= 1)
		ans++;
	return ans;
}

template <typename T1, typename T2>
int** set_d2_offsets(std::vector<int*> histograms, D2Stream<T1> *s1, D2Stream<T2> *s2,
                     int* buffer, int &offsetLen, MemoryContext ctx) {
	int** offsets;
	int* fullHistograms;
	int len;

	len = histograms.size();
	fullHistograms = concat_clear_histograms(histograms, ctx);
	ctx.maxThroughputExponent = cal_max_exponent(bandwidth1);
	offsetLen = solve_bin_packing(fullHistograms, offsets, len, ctx.histogramSize, buffer, ctx);
	s1->set_offsets(offsets, len, offsetLen);
	if (s2 != NULL)
		s2->set_offsets(offsets, len, offsetLen);

	cudaFree(fullHistograms); gpuerr();
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
	Int3** keyStorage;
	int** valueStorage;
	int* keyStorageLen, *valueStorageLen;
	printf("2\n");

	cudaMalloc(&deviceInt, sizeof(int)); gpuerr();
	seq1Device = host_to_device(seq1, seq1Len); gpuerr();
	printf("3\n");

	//=====================================
	// stream 1: generate deletions
	//=====================================

	MemoryContext ctx1 = cal_memory_stream1(seq1Len, distance);
	int outputLen;

	b0 = new GPUInputStream<Int3>(seq1Device, seq1Len, ctx1.chunkSize);
	b1key = new D2Stream<Int3>();
	b1value = new D2Stream<int>();
	printf("4\n");

	while ((b0Chunk = b0->read()).not_null()) {
		stream_handler1(b0Chunk, b1keyOut, b1valueOut, histograms,
		                outputLen, distance, ctx1);
		b1key->write(b1keyOut, outputLen);
		b1value->write(b1valueOut, outputLen);
		_cudaFree(b1keyOut, b1valueOut); gpuerr();
	}

	printf("5\n");


	//=====================================
	// stream 2: group key values
	//=====================================

	MemoryContext ctx2 = cal_memory_stream2();
	int chunkCount, offsetLen;

	printf("6\n");

	offsetLen = histograms.size();
	offsets = set_d2_offsets(histograms, b1key, b1value, deviceInt, chunkCount, ctx2);
	printf("7\n");
	cudaMallocHost(&keyStorage, sizeof(Int3*)*chunkCount);
	cudaMallocHost(&valueStorage, sizeof(int*)*chunkCount);
	cudaMallocHost(&keyStorageLen, sizeof(int)*chunkCount);
	cudaMallocHost(&valueStorageLen, sizeof(int)*chunkCount);
	b2keyOutput = new RAMOutputStream<Int3>(keyStorage, chunkCount, keyStorageLen);
	b2valueOutput = new RAMOutputStream<int>(valueStorage, chunkCount, valueStorageLen);
	printf("8\n");

	while ((b1keyChunk = b1key->read()).not_null()) {
		b1valueChunk = b1value->read();
		printf("9\n");
		stream_handler2(b1keyChunk, b1valueChunk, histograms,
		                distance, seq1Len, deviceInt, ctx2);
		printf("10\n");
		b2keyOutput->write(b1keyChunk.ptr, b1keyChunk.len);
		b2valueOutput->write(b1valueChunk.ptr, b1valueChunk.len);
		// _cudaFree(b1keyChunk.ptr, b1valueChunk.ptr); gpuerr();
	}

	printf("11\n");
	b1key->deconstruct();
	b1value->deconstruct();
	_cudaFreeHost2D(offsets, offsetLen);
	printf("12\n");

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

	// 	b3 = new D2Stream<Int2>(); //TODO
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