#include "test_util.cu"
#include "../src/stream.cu"

TEST(RAMSwapStream, {
	int input[] = {1, 2, 3, 4, 5, 6};
	int len = 6;

	int* input_d = host_to_device(input, len);
	RAMSwapStream<int> *stream = new RAMSwapStream<int>();

	stream->write(input_d, len);
	stream->swap();
	stream->set_max_readable_size(4);

	// a row is never split, so a row larger than the budget(4) still comes back whole. the
	// buffer is sized up to the longest queued row to make that possible.
	check(stream->get_max_readable_size() == 6);

	int expectedData[][6] = {{1, 2, 3, 4, 5, 6}};
	int expectedLen2[] = {6};
	int expectedLen = 1;

	Chunk<int> data;
	int count = 0;
	while ((data = stream->read(4)).not_null()) {
		check(data.len == expectedLen2[count]);
		check_device_arr(data.ptr, expectedData[count], data.len);
		count++;
	}
	check(count == expectedLen);

	// fresh stream/budget so this case (coalescing several small writes into one read) is
	// self-contained instead of depending on whatever budget the previous case left behind
	RAMSwapStream<int> *stream2 = new RAMSwapStream<int>();

	int input2[][4] = {{10, 11}, {12}, {13, 14}, {15, 16, 17, 18}};
	int len22[] = {2, 1, 2, 4};
	int len2 = 4;

	for (int i = 0; i < len2; i++) {
		input_d = host_to_device(input2[i], len22[i]);
		stream2->write(input_d, len22[i]);
	}
	stream2->swap();
	stream2->set_max_readable_size(6);

	// regression guard: the first chunk stops at 5 of the 6 budgeted elements rather than
	// taking one element of the 4-row to pack tighter. splitting a row there would cut a
	// deletion key group in half and silently lose the pairs that cross the cut.
	int expectedData2[][5] = {{10, 11, 12, 13, 14}, {15, 16, 17, 18}};
	int expectedLen22[] = {5, 4};
	int expectedLen21 = 2;

	count = 0;
	while ((data = stream2->read(6)).not_null()) {
		check(data.len == expectedLen22[count]);
		check_device_arr(data.ptr, expectedData2[count], data.len);
		count++;
	}
	check(count == expectedLen21);

	// the buffer shrinks back once a later round needs less, instead of staying at the high
	// water mark(6) for the rest of the run
	int input3[][1] = {{20}, {21}};
	for (int i = 0; i < 2; i++) {
		input_d = host_to_device(input3[i], 1);
		stream2->write(input_d, 1);
	}
	stream2->swap();
	stream2->set_max_readable_size(2);
	check(stream2->get_max_readable_size() == 2);

	int expectedData3[] = {20, 21};
	count = 0;
	while ((data = stream2->read(2)).not_null()) {
		check(data.len == 2);
		check_device_arr(data.ptr, expectedData3, data.len);
		count++;
	}
	check(count == 1);

	// the buffer is capped at what is actually queued, not at the budget. the budget grows as
	// the pipeline frees memory while the queue drains, and the excess could never be filled
	int input4[][1] = {{30}, {31}, {32}};
	for (int i = 0; i < 3; i++) {
		input_d = host_to_device(input4[i], 1);
		stream2->write(input_d, 1);
	}
	stream2->swap();
	stream2->set_max_readable_size(1000);
	check(stream2->get_max_readable_size() == 3);

	int expectedData4[] = {30, 31, 32};
	count = 0;
	while ((data = stream2->read(1000)).not_null()) {
		check(data.len == 3);
		check_device_arr(data.ptr, expectedData4, data.len);
		count++;
	}
	check(count == 1);

	// an empty reading queue leaves the buffer as it is rather than sizing it to zero
	stream2->swap();
	stream2->set_max_readable_size(1000);
	check(stream2->get_max_readable_size() == 3);

	// release_buffer drops the device buffer without touching the queued rows, so that the
	// caller can measure free GPU memory without last round's buffer counted against it
	int input5[][2] = {{40, 41}, {42, 43}};
	for (int i = 0; i < 2; i++) {
		input_d = host_to_device(input5[i], 2);
		stream2->write(input_d, 2);
	}
	stream2->release_buffer();
	check(stream2->get_max_readable_size() == 0);

	stream2->swap();
	stream2->set_max_readable_size(4);
	check(stream2->get_max_readable_size() == 4);

	int expectedData5[] = {40, 41, 42, 43};
	count = 0;
	while ((data = stream2->read(4)).not_null()) {
		check(data.len == 4);
		check_device_arr(data.ptr, expectedData5, data.len);
		count++;
	}
	check(count == 1);
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
		check(buffer.len == expectedLen[chunkCount]);
		check_device_arr(buffer.ptr, expectedData[chunkCount], buffer.len);
		chunkCount++;
	}
	stream->deconstruct();
})