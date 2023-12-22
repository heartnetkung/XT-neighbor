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
	size_t len = 4;
	size_t len2[] = {7, 3, 4, 5};
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

	chunkCount = 0;
	size_t new_len = ostream->get_new_len1();
	size_t* new_len2 = ostream->get_new_len2();
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