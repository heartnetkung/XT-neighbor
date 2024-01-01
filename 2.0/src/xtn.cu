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
int cal_max_exponent(unsigned int input) {
	unsigned input2 = input;
	int ans = 0;
	while (input2 >>= 1)
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
	ctx.maxThroughputExponent = cal_max_exponent(ctx.bandwidth1);
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
	cudaMallocHost(&keyStorage, sizeof(Int3*)*chunkCount); gpuerr();
	cudaMallocHost(&valueStorage, sizeof(int*)*chunkCount); gpuerr();
	cudaMallocHost(&keyStorageLen, sizeof(int)*chunkCount); gpuerr();
	cudaMallocHost(&valueStorageLen, sizeof(int)*chunkCount); gpuerr();
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
	}

	printf("11\n");
	b1key->deconstruct();
	b1value->deconstruct();
	_cudaFreeHost2D(offsets, offsetLen); gpuerr();
	printf("12\n");

	//=====================================
	// loop: lower bound
	//=====================================

	histograms.clear(); //TODO
	cal_lowerbounds(lowerbounds, lowerboundsLen);
	for (int i = 0; i < lowerboundsLen; i++) {
		int lowerbound = lowerbounds[i];
		printf("13\n");

		//=====================================
		// stream 3: generate pairs
		//=====================================

		MemoryContext ctx3 = cal_memory_stream3();
		int len = b2keyOutput->get_new_len1();
		int* len2 = b2keyOutput->get_new_len2();
		int bandwidth1 = ctx3.bandwidth1;
		Int3* keyReadBuffer;
		int* valueReadBuffer;
		printf("13.5\n");

		b3 = new D2Stream<Int2>();
		cudaMalloc(&keyReadBuffer, sizeof(Int3) * bandwidth1); gpuerr();
		cudaMalloc(&valueReadBuffer, sizeof(int) * bandwidth1); gpuerr();
		b2keyInput = new RAMInputStream<Int3>(keyStorage, len, len2, bandwidth1, keyReadBuffer);
		b2valueInput = new RAMInputStream<int>(valueStorage, len, len2, bandwidth1, valueReadBuffer);
		b2keyOutput = new RAMOutputStream<Int3>(keyStorage, len, len2);
		b2valueOutput = new RAMOutputStream<int>(valueStorage, len, len2);
		printf("14\n");

		while ((b2keyChunk = b2keyInput->read()).not_null()) {
			b2valueChunk = b2valueInput->read();
			printf("15\n");
			stream_handler3(b2keyChunk, b2valueChunk, write_b3, histograms,
			                lowerbound, seq1Len, deviceInt, ctx3);
			printf("16\n");
			b2keyOutput->write(b2keyChunk.ptr, b2keyChunk.len);
			b2valueOutput->write(b2valueChunk.ptr, b2valueChunk.len);
			printf("17\n");
		}

		printf("18\n");
		_cudaFree(keyReadBuffer, valueReadBuffer); gpuerr();

		//=====================================
		// stream 4: postprocessing
		//=====================================

		MemoryContext ctx4 = cal_memory_stream4();
		D2Stream<int> *dummy = NULL;

		offsetLen = histograms.size();
		offsets = set_d2_offsets(histograms, b3, dummy, deviceInt, chunkCount, ctx4);
		XTNOutput finalOutput;
		printf("19\n");

		while ((b3Chunk = b3->read()).not_null()) {
			stream_handler4(b3Chunk, finalOutput, seq1Device, seq1Len, distance, deviceInt);
			callback(finalOutput);
			printf("20\n");
		}

		b3->deconstruct();
		_cudaFreeHost2D(offsets, offsetLen); gpuerr();
		printf("21\n");
	}

	//=====================================
	// boilerplate: deallocalte
	//=====================================
	cudaFreeHost(lowerbounds); gpuerr();
	_cudaFree(deviceInt, seq1Device); gpuerr();
	printf("22\n");
}