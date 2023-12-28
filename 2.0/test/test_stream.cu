#include "test_util.cu"
#include "../src/stream.cu"

TEST(GPUInputStream, {
	int input[] = {1, 2, 3, 4, 5, 6, 7, 8};
	int len = 8;
	int chunkSize = 3;
	int outputLen[] = {3, 3, 2};
	int outputData[][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8}};

	int* input_d = host_to_device(input, len);
	GPUInputStream<int> stream(input_d, len, chunkSize);
	Chunk<int> buffer;
	int count = 0;

	while ( (buffer = stream.read()).not_null() ) {
		check(buffer.len == outputLen[count]);
		int* temp = device_to_host(buffer.ptr, buffer.len);
		for (int i = 0; i < buffer.len; i++)
			check(outputData[count][i] == temp[i]);
		count++;
	}
})

TEST(RAMStream, {
	int len = 4;
	int len2[] = {7, 3, 4, 5};
	int** input = (int**)malloc(len * sizeof(int*));
	int count = 0;
	for (int i = 0; i < len; i++) {
		input[i] = (int*)malloc(len2[i] * sizeof(int));
		for (int j = 0; j < len2[i]; j++)
			input[i][j] = ++count;
	}

	int expectedLen2[] = {7, 7, 5};
	int expectedData[][7] = {{1, 2, 3, 4, 5, 6, 7}, {8, 9, 10, 11, 12, 13, 14}, {15, 16, 17, 18, 19}};

	int maxReadableSize = 7;
	int* deviceBuffer;
	cudaMalloc((void**)&deviceBuffer, sizeof(int)*maxReadableSize);
	RAMInputStream<int> *istream = new RAMInputStream<int>(input , len, len2, maxReadableSize, deviceBuffer);
	RAMOutputStream<int> *ostream = new RAMOutputStream<int>(input, len, len2);

	// first write loop
	Chunk<int> buffer;
	int chunkCount = 0;
	while ( (buffer = istream->read()).not_null() ) {
		int* data = device_to_host(buffer.ptr, buffer.len);
		check(buffer.len == expectedLen2[chunkCount]);
		for (int i = 0; i < buffer.len; i++)
			check(data[i] == expectedData[chunkCount][i]);

		//simple filtering
		int n_data = 0;
		for (int i = 0; i < buffer.len; i++)
			if (data[i] % 2 == 0)
				data[n_data++] = data[i];

		//continue
		data = host_to_device(data, n_data);
		ostream->write(data, n_data);
		chunkCount++;
	}

	// check what is written
	int expected2Len = 3;
	int expected2Len2[] = {3, 4, 2};
	int expected2Data[][4] = {{2, 4, 6}, {8, 10, 12, 14}, {16, 18}};
	check(ostream->get_new_len1() == expected2Len);
	for (int i = 0; i < chunkCount; i++) {
		check(ostream->get_new_len2()[i] == expected2Len2[i]);
		for (int j = 0; j < len2[i]; j++)
			check(input[i][j] == expected2Data[i][j]);
	}

	int expected3Len = 2;
	int expected3Len2[] = {7, 2};
	int expected3Data[][7] = {{2, 4, 6, 8, 10, 12, 14}, {16, 18}};

	// second write loop
	chunkCount = 0;
	int new_len = ostream->get_new_len1();
	int* new_len2 = ostream->get_new_len2();
	istream = new RAMInputStream<int>(input , new_len, new_len2, maxReadableSize, deviceBuffer);
	ostream = new RAMOutputStream<int>(input, new_len, new_len2);

	while ( (buffer = istream->read()).not_null() ) {
		int* data = device_to_host(buffer.ptr, buffer.len);
		check(buffer.len == expected3Len2[chunkCount]);
		for (int i = 0; i < buffer.len; i++)
			check(data[i] == expected3Data[chunkCount][i]);

		//simple filtering
		int n_data = 0;
		for (int i = 0; i < buffer.len; i++)
			if (data[i] % 3 != 0)
				data[n_data++] = data[i];

		//continue
		data = host_to_device(data, n_data);
		ostream->write(data, n_data);
		chunkCount++;
	}
	check(chunkCount == expected3Len);

	// check what is written
	int expected4Len = 2;
	int expected4Len2[] = {5, 1};
	int expected4Data[][5] = {{2, 4, 8, 10, 14}, {16}};
	check(ostream->get_new_len1() == expected4Len);
	for (int i = 0; i < chunkCount; i++) {
		check(ostream->get_new_len2()[i] == expected4Len2[i]);
		for (int j = 0; j < len2[i]; j++)
			check(input[i][j] == expected4Data[i][j]);
	}
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
	D2Stream<int> stream(len);

	//write
	for (int i = 0; i < len; i++)
		stream.write( host_to_device(input[i], len2[i]), len2[i]);
	stream.set_offsets(offsets, offset_len);

	//expectation
	int expectedLen[] = {5, 4, 14};
	int expectedData[][14] = {{6, 7, 8, 13, 19}, {1, 14, 15, 20}, {2, 3, 4, 5, 9, 10, 11, 12, 16, 17, 18, 21, 22, 23}};

	//read
	Chunk<int> buffer;
	int chunkCount = 0;
	while ( (buffer = stream.read()).not_null() ) {
		int* data = device_to_host(buffer.ptr, buffer.len);
		check(buffer.len == expectedLen[chunkCount]);
		for (int i = 0; i < buffer.len; i++)
			check(data[i] == expectedData[chunkCount][i]);
		chunkCount++;
	}
	stream.deconstruct();
})