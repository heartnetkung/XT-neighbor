#include "test_util.cu"
#include "../src/stream.cu"

TEST(RAMSwapStream, {
	int input[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
	int len = 9;
	int histogram[] = {2, 2, 1, 1, 3};
	int histogramSize = 5;

	int* input_d = host_to_device(input, len);
	RAMSwapStream<int> *stream = new RAMSwapStream<int>();
	stream->set_max_writable_size(3);
	stream->set_max_readable_size(4);

	stream->write(input_d, len, histogram, histogramSize);
	stream->swap();

	int expectedData[][3] = {{1, 2}, {3, 4, 5}, {6}, {7, 8, 9}};
	int expectedLen2[] = {2, 3, 1, 3};
	int expectedLen = 4;

	Chunk<int> data;
	int count = 0;
	while ((data = stream->read()).not_null()) {
		check(data.len == expectedLen2[count]);
		int* hostPtr = device_to_host(data.ptr, data.len);
		for (int i = 0; i < data.len; i++)
			check(expectedData[count][i] == hostPtr[i]);
		count++;
	}
	check(count == expectedLen);
})


TEST(D2Stream, {
	int len = 4;
	int len2[] = {5, 7, 6, 5};
	int** input = (int**)malloc(len * sizeof(int*));
	int count = 0;
	for (int i = 0; i < len; i++) {
		input[i] = (int*) malloc(len2[i] * sizeof(int));
		for (int j = 0; j < len2[i]; j++)
			input[i][j] = ++count;
	}
	int offset_len = 3;
	int** offsets = (int**)malloc(len * sizeof(int*));
	for (int i = 0; i < len; i++)
		offsets[i] = (int*)malloc(offset_len * sizeof(int));
	offsets[0][0] = 0; offsets[0][1] = 1; offsets[0][2] = 5;
	offsets[1][0] = 3; offsets[1][1] = 3; offsets[1][2] = 7;
	offsets[2][0] = 1; offsets[2][1] = 3; offsets[2][2] = 6;
	offsets[3][0] = 1; offsets[3][1] = 2; offsets[3][2] = 5;
	D2Stream<int> *stream = new D2Stream<int>();

	//write
	for (int i = 0; i < len; i++) {
		stream->write(host_to_device(input[i], len2[i]), len2[i]);
	}
	stream->set_offsets(offsets, len, offset_len);

	//expectation
	int expectedLen[] = {5, 4, 14};
	int expectedData[][14] = {{6, 7, 8, 13, 19}, {1, 14, 15, 20}, {2, 3, 4, 5, 9, 10, 11, 12, 16, 17, 18, 21, 22, 23}};

	//read
	Chunk<int> buffer;
	int chunkCount = 0;
	while ( (buffer = stream->read()).not_null() ) {
		int* data = device_to_host(buffer.ptr, buffer.len);
		check(buffer.len == expectedLen[chunkCount]);
		for (int i = 0; i < buffer.len; i++)
			check(data[i] == expectedData[chunkCount][i]);
		chunkCount++;
	}
	stream->deconstruct();
})