#include "../src/util.cu"

/**
 * Chunk represents a unit of data read by the stream
 */
template <typename T> class Chunk {
public:
	T* ptr = NULL;
	size_t len = 0;
	bool not_null() {return ptr != NULL;}
};

/**
 * A stream where the data is already fit inside the GPU memory, so
 * streaming is basically just shift the pointer around.
 */
template <typename T> class GPUInputStream {
private:
	T* _data = NULL;
	size_t _len = 0, _chunkSize = 0;

	void check_input(T* data, size_t len, size_t chunkSize) {
		if (data == NULL)
			print_err("GPUInputStream: data == NULL");
		if (len <= 0)
			print_err("GPUInputStream: len <= 0");
		if (chunkSize <= 0)
			print_err("GPUInputStream: chunkSize <= 0");
	}

public:
	GPUInputStream(T* data, size_t len, size_t chunkSize) {
		check_input(data, len, chunkSize);
		_data = data;
		_len = len;
		_chunkSize = chunkSize;
	}

	Chunk<T> read() {
		Chunk<T> ans;
		if (_len == 0)
			return ans;

		ans.ptr = _data;
		ans.len = _chunkSize > _len ? _len : _chunkSize;
		_len -= ans.len;
		_data += ans.len;
		return ans;
	}
};

/**
 * A 1 dimensional stream where the original data resides in the RAM
 * and a chunk is transfered to GPU memory one at a time.
 *
 * Note that the RAM backend is designed to be shared with RAMOutputStream.
 */
template <typename T> class RAMInputStream {
private:
	T** _data = NULL;
	size_t* _len2 = NULL;
	size_t _len1 = 0, _maxReadableSize = 0, _index = 0;
	T* _deviceBuffer = NULL;

	void check_input(T** data, size_t len1, size_t* len2, size_t maxReadableSize, T* deviceBuffer) {
		if (len1 <= 0)
			print_err("RAMInputStream: len1 <= 0");
		if (maxReadableSize <= 0)
			print_err("RAMInputStream: maxReadableSize <= 0");
		if (data == NULL)
			print_err("RAMInputStream: data == NULL");
		if (deviceBuffer == NULL)
			print_err("RAMInputStream: deviceBuffer == NULL");
		for (size_t i = 0; i < len1; i++) {
			if (len2[i] > maxReadableSize)
				print_err("RAMInputStream: len2[i] > maxReadableSize will lead to infinite loop");
			if ((data[i] == NULL) && (len2[i] > 0))
				print_err("RAMInputStream: (data[i] == NULL) && (len2[i] > 0)");
		}
	}

public:
	RAMInputStream(T** data, size_t len1, size_t* len2, size_t maxReadableSize, T* deviceBuffer) {
		check_input(data, len1, len2, maxReadableSize, deviceBuffer);
		_data = data;
		_len1 = len1;
		_len2 = len2;
		_maxReadableSize = maxReadableSize;
		_deviceBuffer = deviceBuffer;
	}

	Chunk<T> read() {
		Chunk<T> ans;
		if (_index == _len1)
			return ans;

		ans.ptr = _deviceBuffer;
		T* currentPtr = _deviceBuffer;
		for (; _index < _len1; _index++) {
			size_t newLen = _len2[_index];
			if (ans.len + newLen > _maxReadableSize)
				break;
			if (newLen == 0)
				continue;

			cudaMemcpy(currentPtr, _data[_index], sizeof(T)*newLen, cudaMemcpyHostToDevice);
			currentPtr += newLen;
			ans.len += newLen;
		}
		return ans;
	}
};

/**
 * A 1 dimensional stream where the original data resides in the RAM
 * and a chunk is written from GPU memory one at a time.
 *
 * Note that the RAM backend is designed to be shared with RAMInputStream.
 */
template <typename T> class RAMOutputStream {
private:
	T** _data = NULL;
	size_t* _len2 = NULL;
	size_t _len1 = 0, _index = 0;

	void check_input(T** data, size_t len1, size_t* len2) {
		if (len1 <= 0)
			print_err("RAMOutputStream: len1 <= 0");
		if (data == NULL)
			print_err("RAMOutputStream: data == NULL");
	}

	void check_input_write(T* newData, size_t n) {
		if ((newData == NULL) && (n > 0))
			print_err("RAMOutputStream: (newData == NULL) && (n > 0)");
	}

public:
	RAMOutputStream(T** data, size_t len1, size_t* len2) {
		check_input(data, len1, len2);
		_data = data;
		_len1 = len1;
		_len2 = len2;
	}

	void write(T* newData, size_t n) {
		check_input_write(newData, n);
		if (_index >= _len1)
			print_err("RAMOutputStream: writing more than allocated");
		if (n > 0)
			cudaMemcpy(_data[_index], newData, sizeof(T)*n, cudaMemcpyDeviceToHost);
		_len2[_index] = n;
		_index++;
	}

	size_t get_new_len1() {
		return _index;
	}

	size_t* get_new_len2() {
		return _len2;
	}
};

// template <typename T> class D2Stream {

// }
