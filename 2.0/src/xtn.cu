#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "xtn_inner.cu"

/**
 * @file
 * @brief The core algorithm on a low-level abstraction concerning how buffers,
 * streamings, and memory management techniques fit together.
 */

D2Stream<Int2> *b3 = NULL; /*global variable for callback*/
const int MAX_PROCESSING = 1 << 30;

//=====================================
// Private Memory Functions
//=====================================

MemoryContext initMemory(int seq1Len, bool isGPU) {
	MemoryContext ans;
	if (isGPU)
		ans.gpuSize = get_gpu_memory();
	else
		ans.ramSize = get_main_memory();
	if (ans.histogramSize > seq1Len)
		ans.histogramSize = seq1Len;
	return ans;
}

// black magic way to calculate floor(log2(n))
int cal_max_exponent(size_t input) {
	size_t input2 = input;
	int ans = 0;
	while (input2 >>= 1)
		ans++;
	return ans;
}

MemoryContext cal_memory_stream1(int seq1Len, int distance) {
	MemoryContext ans = initMemory(seq1Len, true);
	int deletionMultiplier = (distance == 1) ? (18 + 1) : (153 + 18 + 1);
	int multiplier =
	    //bottleneck: Int3* &deletionsOutput int* &indexOutput sort_key_values
	    deletionMultiplier * (2 * sizeof(Int3) + 2 * sizeof(int));

	size_t temp = ans.gpuSize / multiplier; /*safety factor is included in deletionMultiplier*/
	ans.bandwidth1 = (temp > MAX_PROCESSING) ? MAX_PROCESSING : temp;
	ans.chunkSize = (seq1Len < ans.bandwidth1) ? seq1Len : ans.bandwidth1;
	return ans;
}

MemoryContext cal_memory_stream2(int seq1Len) {
	MemoryContext ans = initMemory(seq1Len, true);
	int multiplier =
	    //bottleneck: input sort_key_values
	    2 * sizeof(Int3) + 2 * sizeof(int);

	size_t temp = ans.gpuSize / (10 * multiplier);
	ans.bandwidth1 = (temp > MAX_PROCESSING) ? MAX_PROCESSING : temp;
	temp = temp * 9;
	ans.bandwidth2 = (temp > MAX_PROCESSING) ? MAX_PROCESSING : temp;
	ans.maxThroughputExponent = cal_max_exponent(ans.bandwidth1);
	return ans;
}

MemoryContext cal_memory_stream3(int seq1Len) {
	MemoryContext ans = initMemory(seq1Len, true);
	int multiplier =
	    2 * sizeof(int) + // int* &inputOffsets, int* &outputLengths
	    sizeof(char) + sizeof(Int3) + sizeof(int); //char* flags Int3* keyOut int* valueOut;

	size_t temp = ans.gpuSize / (10 * multiplier);
	ans.bandwidth1 = (temp > MAX_PROCESSING) ? MAX_PROCESSING : temp;
	temp = temp * 9;
	ans.bandwidth2 = (temp > MAX_PROCESSING) ? MAX_PROCESSING : temp;
	return ans;
}

MemoryContext cal_memory_stream4(int seq1Len) {
	MemoryContext ans = initMemory(seq1Len, true);
	int multiplier =
	    sizeof(Int2) + //Int2* uniquePairs
	    2 * sizeof(char) + //char* uniqueDistances, *flags
	    sizeof(Int2) + //Int2* &pairOutput
	    sizeof(char);// char* &distanceOutput

	size_t temp = ans.gpuSize / multiplier;
	ans.bandwidth1 = (temp > MAX_PROCESSING) ? MAX_PROCESSING : temp;
	ans.maxThroughputExponent = cal_max_exponent(ans.bandwidth1);
	return ans;
}

MemoryContext cal_memory_lowerbound(int seq1Len) {
	MemoryContext ans = initMemory(seq1Len, false);
	size_t bandwidth = 7 * ans.ramSize / (sizeof(Int2) * 10);
	ans.maxThroughputExponent = cal_max_exponent(bandwidth);
	return ans;
}

//=====================================
// Other Private Functions
//=====================================

void write_b3(Int2* pairOutput, int pairLen) {
	b3->write(pairOutput, pairLen);
}

int* concat_histograms(std::vector<int*> histograms, MemoryContext ctx) {
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
	return ans;
}

int cal_lowerbounds(std::vector<int*> histograms, int* &lowerbounds, int seqLen, int* buffer) {
	int* fullHistograms;
	int outputLen;
	MemoryContext ctx;

	ctx = cal_memory_lowerbound(seqLen);
	fullHistograms = concat_histograms(histograms, ctx);
	outputLen = solve_bin_packing_lowerbounds(
	                fullHistograms, lowerbounds, histograms.size(), seqLen, buffer, ctx);

	cudaFree(fullHistograms); gpuerr();
	return outputLen;
}

template <typename T1, typename T2>
int** set_d2_offsets(std::vector<int*> histograms, D2Stream<T1> *s1, D2Stream<T2> *s2,
                     int* buffer, MemoryContext ctx) {
	int** offsets;
	int* fullHistograms;
	int len, offsetLen;

	len = histograms.size();
	fullHistograms = concat_histograms(histograms, ctx);
	offsetLen = solve_bin_packing_offsets(
	                fullHistograms, offsets, len, buffer, ctx);

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
	clock_start();

	int* deviceInt, *lowerbounds;
	Int3* seq1Device;
	std::vector<int*> histograms;
	int** offsets;
	int lowerboundsLen;
	int distance = args.distance, seq1Len = args.seq1Len;

	GPUInputStream<Int3> *b0;
	D2Stream<Int3> *b1key;
	D2Stream<int> *b1value;
	RAMSwapStream<Int3> *b2key;
	RAMSwapStream<int> *b2value;
	Chunk<Int3> b0Chunk, b1keyChunk, b2keyChunk;
	Chunk<Int2> b3Chunk;
	Chunk<int> b1valueChunk, b2valueChunk;
	Int3* b1keyOut;
	int* b1valueOut;

	cudaMalloc(&deviceInt, sizeof(int)); gpuerr();
	seq1Device = host_to_device(seq1, seq1Len);
	print_v("0A");

	//=====================================
	// stream 1: generate deletions
	//=====================================

	MemoryContext ctx1 = cal_memory_stream1(seq1Len, distance);
	int outputLen;

	b0 = new GPUInputStream<Int3>(seq1Device, seq1Len, ctx1.chunkSize);
	b1key = new D2Stream<Int3>();
	b1value = new D2Stream<int>();
	print_v("1A");

	while ((b0Chunk = b0->read()).not_null()) {
		print_bandwidth(b0Chunk.len, ctx1.bandwidth1, "1");

		stream_handler1(b0Chunk, b1keyOut, b1valueOut, histograms,
		                outputLen, distance, ctx1);

		b1key->write(b1keyOut, outputLen);
		b1value->write(b1valueOut, outputLen);

		_cudaFree(b1keyOut, b1valueOut);
		print_v("1B");
	}

	print_tl("1", b1key->get_total_len());

	//=====================================
	// stream 2: group key values
	//=====================================

	MemoryContext ctx2 = cal_memory_stream2(seq1Len);
	int offsetLen;
	size_t totalLen2B = 0;

	offsetLen = histograms.size();
	offsets = set_d2_offsets(histograms, b1key, b1value, deviceInt, ctx2);
	histograms.clear();

	b2key = new RAMSwapStream<Int3>();
	b2value = new RAMSwapStream<int>();
	print_v("2A");

	while ((b1keyChunk = b1key->read()).not_null()) {
		b1valueChunk = b1value->read();
		print_bandwidth(b1keyChunk.len, ctx2.bandwidth1, "2");
		stream_handler2(b1keyChunk, b1valueChunk, histograms, totalLen2B,
		                distance, seq1Len, deviceInt, ctx2);
		b2key->write(b1keyChunk.ptr, b1keyChunk.len);
		b2value->write(b1valueChunk.ptr, b1valueChunk.len);
		print_v("2B");
	}

	b1key->deconstruct();
	b1value->deconstruct();
	_cudaFreeHost2D(offsets, offsetLen);
	print_tl("2", b2key->get_total_len());
	print_tl("2B", totalLen2B);

	//=====================================
	// loop: lower bound
	//=====================================

	size_t totalLen3B = 0;
	lowerboundsLen = cal_lowerbounds(histograms, lowerbounds, seq1Len, deviceInt);
	histograms.clear();
	if (verboseGlobal)
		print_int_arr(lowerbounds, lowerboundsLen);

	for (int i = 0; i < lowerboundsLen; i++) {
		int lowerbound = lowerbounds[i];
		if (verboseGlobal)
			printf("lower bound loop: %d / %d\n", i + 1, lowerboundsLen);

		//=====================================
		// stream 3: generate pairs
		//=====================================

		MemoryContext ctx3 = cal_memory_stream3(seq1Len);
		b2key->swap();
		b2value->swap();
		b2key->set_max_readable_size(ctx3.bandwidth1);
		b2value->set_max_readable_size(ctx3.bandwidth1);
		b3 = new D2Stream<Int2>();
		print_v("3A");

		while ((b2keyChunk = b2key->read()).not_null()) {
			b2valueChunk = b2value->read();
			print_bandwidth(b2keyChunk.len, ctx3.bandwidth1, "3");
			stream_handler3(b2keyChunk, b2valueChunk, write_b3, histograms,
			                lowerbound, seq1Len, deviceInt, ctx3);
			b2key->write(b2keyChunk.ptr, b2keyChunk.len);
			b2value->write(b2valueChunk.ptr, b2valueChunk.len);

			_cudaFree(b2keyChunk.ptr, b2valueChunk.ptr);
			print_v("3B");
		}

		print_tl("3.1", b2key->get_total_len());
		print_tl("3.2", b3->get_total_len());
		totalLen3B += b3->get_total_len();

		//=====================================
		// stream 4: postprocessing
		//=====================================

		MemoryContext ctx4 = cal_memory_stream4(seq1Len);
		D2Stream<int> *dummy = NULL;
		size_t totalLen4 = 0;

		offsetLen = histograms.size();
		offsets = set_d2_offsets(histograms, b3, dummy, deviceInt, ctx4);
		histograms.clear();
		XTNOutput finalOutput;
		print_v("4A");

		while ((b3Chunk = b3->read()).not_null()) {
			print_bandwidth(b3Chunk.len, ctx4.bandwidth1, "4");
			stream_handler4(b3Chunk, finalOutput, seq1Device, seq1Len, distance, deviceInt);
			totalLen4 += finalOutput.len;
			callback(finalOutput);
			_cudaFreeHost(finalOutput.indexPairs, finalOutput.pairwiseDistances);
			print_v("4B");
		}

		b3->deconstruct();
		_cudaFreeHost2D(offsets, offsetLen);
		print_tl("4", totalLen4);
	}

	//=====================================
	// boilerplate: deallocalte
	//=====================================
	cudaFreeHost(lowerbounds); gpuerr();
	_cudaFree(deviceInt, seq1Device);
	b2key->deconstruct();
	b2value->deconstruct();
	if (verboseGlobal)
		printf("totalLen 3B: %'lu\n", totalLen3B);
	print_v("5");
}