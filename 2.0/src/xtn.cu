#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "xtn_inner.cu"

/**
 * @file
 * The core algorithm on a low-level abstraction concerning how buffers,
 * streamings, and memory management techniques fit together.
 */

D2Stream<Int2> *b3 = NULL; /*global variable for callback*/
const int MAX_PROCESSING = 1 << 30;

//=====================================
// Private Memory Functions
//=====================================

/**
 * private function.
*/
MemoryContext initMemory(int seq1Len, bool isGPU) {
	MemoryContext ans;
	if (isGPU)
		ans.gpuSize = get_gpu_memory();
	else
		ans.ramSize = get_main_memory();
	if (ans.histogramSize > seq1Len)
		ans.histogramSize = seq1Len;
	else if (seq1Len > 10000000)
		// ans.histogramSize = 262144;
		ans.histogramSize = 1048576;
	return ans;
}

// black magic way to calculate floor(log2(n))
/**
 * private function.
*/
int cal_max_exponent(size_t input) {
	size_t input2 = input;
	int ans = 0;
	while (input2 >>= 1)
		ans++;
	return ans;
}

/**
 * calculate memory constraint for stream 1 using the upper bound of the memory allocation during the operation.
*/
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

/**
 * calculate memory constraint for stream 2 using the upper bound of the memory allocation during the operation.
*/
MemoryContext cal_memory_stream2(int seq1Len) {
	MemoryContext ans = initMemory(seq1Len, true);
	int multiplier =
	    //bottleneck: input sort_key_values
	    2 * sizeof(Int3) + 2 * sizeof(int);

	size_t temp = ans.gpuSize / (2 * multiplier);
	ans.bandwidth1 = (temp > MAX_PROCESSING) ? MAX_PROCESSING : temp;
	ans.bandwidth2 = (temp > MAX_PROCESSING) ? MAX_PROCESSING : temp;
	ans.maxThroughputExponent = cal_max_exponent(ans.bandwidth1);
	return ans;
}

/**
 * calculate memory constraint for stream 3 using the upper bound of the memory allocation during the operation.
*/
MemoryContext cal_memory_stream3(int seq1Len) {
	MemoryContext ans = initMemory(seq1Len, true);
	int multiplier =
	    2 * sizeof(int) + // int* &inputOffsets, int* &outputLengths
	    sizeof(char) + sizeof(Int3) + sizeof(int); //char* flags Int3* keyOut int* valueOut;

	size_t temp = ans.gpuSize / (2 * multiplier);
	ans.bandwidth1 = (temp > MAX_PROCESSING) ? MAX_PROCESSING : temp;
	ans.bandwidth2 = (temp > MAX_PROCESSING) ? MAX_PROCESSING : temp;
	return ans;
}

/**
 * calculate memory constraint for stream 4 using the upper bound of the memory allocattion during the operation.
*/
MemoryContext cal_memory_stream4(int seq1Len, bool overlapMode) {
	MemoryContext ans = initMemory(seq1Len, true);
	int multiplier = 2 * sizeof(Int2) + //Int2* uniquePairs, sorting
	                 2 * sizeof(char) + //char* uniqueDistances, *flags
	                 sizeof(Int2);//Int2* &pairOutput
	if (!overlapMode)
		multiplier += sizeof(char); // char* &distanceOutput;

	size_t temp = (8 * ans.gpuSize) / (10 * multiplier);
	ans.bandwidth1 = (temp > MAX_PROCESSING) ? MAX_PROCESSING : temp;
	ans.maxThroughputExponent = cal_max_exponent(ans.bandwidth1);
	return ans;
}

/**
 * calculate RAM constraint for lower bound calculation.
*/
MemoryContext cal_memory_lowerbound(int seq1Len) {
	MemoryContext ans = initMemory(seq1Len, false);
	size_t bandwidth = 7 * ans.ramSize / (sizeof(Int2) * 10);
	ans.maxThroughputExponent = cal_max_exponent(bandwidth);
	return ans;
}

//=====================================
// Other Private Functions
//=====================================

/**
 * callback function to write generated pairs.
*/
void write_b3(Int2* pairOutput, int pairLen) {
	b3->write(pairOutput, pairLen);
}

/**
 * flatten histograms stored as vector<int*> to a 1D array.
*/
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

/**
 * calculate the lowerbounds from the collected histogram and memory constraints.
*/
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

/**
 * apply precalculated bin packing offset to the given 2D buffers.
*/
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

/**
 * the main function for XT-neighbor algorithm.
 *
 * @param args all flags parsed from command line
 * @param seq1 sequence input
 * @param seqFreqHost frequency of each CDR3 sequence, only used in overlap mode
 * @param repSizesHost size of each repertiore, only used in overlap mode
 * @param callback function to be invoked once a chunk of output is ready
*/
void xtn_perform(XTNArgs args, Int3* seq1, int* seqFreqHost,
                 int* repSizesHost, void callback(XTNOutput)) {
	clock_start();

	int* deviceInt, *lowerbounds, *seqFreq = NULL, *repSizes = NULL;
	Int3* seq1Device;
	std::vector<int*> histograms;
	int** offsets;
	int lowerboundsLen;
	int distance = args.distance, seq1Len = args.seq1Len;
	bool overlapMode = (args.infoPath != NULL);

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
	int outputLen, carry = 0;

	b0 = new GPUInputStream<Int3>(seq1Device, seq1Len, ctx1.chunkSize);
	b1key = new D2Stream<Int3>();
	b1value = new D2Stream<int>();
	print_v("1A");

	while ((b0Chunk = b0->read()).not_null()) {
		print_bandwidth(b0Chunk.len, ctx1.bandwidth1, "1");

		stream_handler1(b0Chunk, b1keyOut, b1valueOut, histograms,
		                outputLen, distance, carry, ctx1);

		b1key->write(b1keyOut, outputLen);
		b1value->write(b1valueOut, outputLen);

		_cudaFree(b1keyOut, b1valueOut);
		print_v("1B");
	}

	printf("AX0\n");
	cudaDeviceSynchronize(); gpuerr();
	printf("AX1\n");

	print_tl("1", b1key->get_total_len());

	//=====================================
	// stream 2: group key values
	//=====================================

	printf("BX0\n");
	cudaDeviceSynchronize(); gpuerr();
	printf("BX1\n");

	MemoryContext ctx2 = cal_memory_stream2(seq1Len);
	int offsetLen;
	size_t totalLen2B = 0;

	printf("CX0\n");
	cudaDeviceSynchronize(); gpuerr();
	printf("CX1\n");

	offsetLen = histograms.size();
	offsets = set_d2_offsets(histograms, b1key, b1value, deviceInt, ctx2);
	histograms.clear();

	printf("DX0\n");
	cudaDeviceSynchronize(); gpuerr();
	printf("DX1\n");

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


	printf("EX0\n");
	cudaDeviceSynchronize(); gpuerr();
	printf("EX1\n");

	b1key->deconstruct();
	b1value->deconstruct();
	_cudaFreeHost2D(offsets, offsetLen);
	print_tl("2", b2key->get_total_len());
	print_tl("2B", totalLen2B);

	//=====================================
	// loop: lower bound
	//=====================================

	printf("FX0\n");
	cudaDeviceSynchronize(); gpuerr();
	printf("FX1\n");

	size_t totalLen3B = 0;
	XTNOutput finalOutput;
	lowerboundsLen = cal_lowerbounds(histograms, lowerbounds, seq1Len, deviceInt);
	histograms.clear();
	if (verboseGlobal)
		print_int_arr(lowerbounds, lowerboundsLen);


	printf("GX0\n");
	cudaDeviceSynchronize(); gpuerr();
	printf("GX1\n");

	if (overlapMode) {
		Int2* indexPairs;
		size_t* pairwiseFrequencies;
		seqFreq = host_to_device(seqFreqHost, seq1Len);
		repSizes = host_to_device(repSizesHost, args.infoLen);
		printf("GX2\n");
		cudaDeviceSynchronize(); gpuerr();
		printf("GX3\n");
		init_overlap(indexPairs, pairwiseFrequencies, seqFreq, repSizes, seq1Len, args.infoLen);
		finalOutput.len = seq1Len;
		finalOutput.indexPairs = indexPairs;
		finalOutput.pairwiseFrequencies = finalOutput;
	}


	printf("HX0\n");
	cudaDeviceSynchronize(); gpuerr();
	printf("HX1\n");

	for (int i = 0; i < lowerboundsLen; i++) {
		int lowerbound = lowerbounds[i];
		if (verboseGlobal)
			printf("lower bound loop: %d / %d\n", i + 1, lowerboundsLen);


		printf("IX0");
		cudaDeviceSynchronize(); gpuerr();
		printf("IX1");

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


		printf("JX0");
		cudaDeviceSynchronize(); gpuerr();
		printf("JX1");

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

		MemoryContext ctx4 = cal_memory_stream4(seq1Len, overlapMode);
		D2Stream<int> *dummy = NULL;
		size_t totalLen4 = 0;

		offsetLen = histograms.size();
		offsets = set_d2_offsets(histograms, b3, dummy, deviceInt, ctx4);
		histograms.clear();
		print_v("4A");

		while ((b3Chunk = b3->read()).not_null()) {
			print_bandwidth(b3Chunk.len, ctx4.bandwidth1, "4");

			if (overlapMode)
				stream_handler4_overlap(b3Chunk, finalOutput, seq1Device,
				                        seqFreq, repSizes, args.infoLen,
				                        seq1Len, distance, args.measure, deviceInt);
			else {
				stream_handler4_nn(b3Chunk, finalOutput, seq1Device, seq1Len,
				                   distance, args.measure, deviceInt);
				callback(finalOutput);
				_cudaFreeHost(finalOutput.indexPairs, finalOutput.pairwiseDistances);
			}

			totalLen4 += finalOutput.len;
			print_v("4B");
		}

		b3->deconstruct();
		_cudaFreeHost2D(offsets, offsetLen);
		print_tl("4", totalLen4);
	}

	if (overlapMode) {
		Int2* indexPairs = device_to_host(finalOutput.indexPairs, finalOutput.len);
		size_t* pairwiseFreq = device_to_host(
		                           finalOutput.pairwiseFrequencies, finalOutput.len);
		_cudaFree(seqFreq, repSizes, finalOutput.indexPairs,
		          finalOutput.pairwiseFrequencies);
		finalOutput.indexPairs = indexPairs;
		finalOutput.pairwiseFrequencies = pairwiseFreq;
		callback(finalOutput);
		_cudaFreeHost(indexPairs, pairwiseFreq);
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