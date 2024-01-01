#include "codec.cu"

/**
 * Chunk represents a unit of data read by the stream
 */
template <typename T> class Chunk {
public:
	T* ptr = NULL;
	int len = 0;
	bool not_null() {return ptr != NULL;}
};

/**
 * A stream where the data is already fit inside the GPU memory, so
 * streaming is basically just shift the pointer around.
 */
template <typename T> class GPUInputStream {
private:
	T* _data = NULL;
	int _len = 0, _chunkSize = 0, _tp;

	void check_input(T* data, int len, int chunkSize) {
		if (data == NULL)
			print_err("GPUInputStream: data == NULL");
		if (len <= 0)
			print_err("GPUInputStream: len <= 0");
		if (chunkSize <= 0)
			print_err("GPUInputStream: chunkSize <= 0");
	}

public:
	GPUInputStream(T* data, int len, int chunkSize) {
		check_input(data, len, chunkSize);
		_data = data;
		_len = len;
		_chunkSize = chunkSize;
		_tp = len;
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

	size_t get_throughput() {
		return _tp;
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
	T* _deviceBuffer = NULL;
	int* _len2 = NULL;
	int _len1 = 0, _maxReadableSize = 0, _index = 0;

	void check_input(T** data, int len1, int* len2, int maxReadableSize, T* deviceBuffer) {
		if (len1 <= 0)
			print_err("RAMInputStream: len1 <= 0");
		if (maxReadableSize <= 0)
			print_err("RAMInputStream: maxReadableSize <= 0");
		if (data == NULL)
			print_err("RAMInputStream: data == NULL");
		if (deviceBuffer == NULL)
			print_err("RAMInputStream: deviceBuffer == NULL");
		for (int i = 0; i < len1; i++) {
			if (len2[i] > maxReadableSize)
				print_err("RAMInputStream: len2[i] > maxReadableSize will lead to infinite loop");
			if ((data[i] == NULL) && (len2[i] > 0))
				print_err("RAMInputStream: (data[i] == NULL) && (len2[i] > 0)");
		}
	}

public:
	RAMInputStream(T** data, int len1, int* len2, int maxReadableSize, T* deviceBuffer) {
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
			int newLen = _len2[_index];
			if (ans.len + newLen > _maxReadableSize)
				break;
			if (newLen == 0)
				continue;

			cudaMemcpy(currentPtr, _data[_index], sizeof(T)*newLen, cudaMemcpyHostToDevice); gpuerr();
			currentPtr += newLen;
			ans.len += newLen;
		}
		return ans;
	}

	size_t get_throughput() {
		size_t ans = 0;
		for (int i = 0; i < _len1; i++)
			ans += _len2[i];
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
	int* _len2 = NULL;
	int _len1 = 0, _index = 0;

	void check_input(T** data, int len1, int* len2) {
		if (len1 <= 0)
			print_err("RAMOutputStream: len1 <= 0");
		if (data == NULL)
			print_err("RAMOutputStream: data == NULL");
	}

	void check_input_write(T* newData, int n) {
		if ((newData == NULL) && (n > 0))
			print_err("RAMOutputStream: (newData == NULL) && (n > 0)");
	}

public:
	RAMOutputStream(T** data, int len1, int* len2) {
		check_input(data, len1, len2);
		_data = data;
		_len1 = len1;
		_len2 = len2;
	}

	void write(T* newData, int n) {
		check_input_write(newData, n);
		if (_index >= _len1)
			print_err("RAMOutputStream: writing more than allocated");
		if (n > 0) {
			if (_data[_index] != NULL)
				cudaFreeHost(_data[_index]); gpuerr();
			cudaMallocHost(&_data[_index], sizeof(T)*n); gpuerr();
			cudaMemcpy(_data[_index], newData, sizeof(T)*n, cudaMemcpyDeviceToHost); gpuerr();
		}
		_len2[_index] = n;
		_index++;
	}

	int get_new_len1() {
		return _index;
	}

	int* get_new_len2() {
		return _len2;
	}

	size_t get_throughput() {
		size_t ans = 0;
		for (int i = 0; i < _index; i++)
			ans += _len2[i];
		return ans;
	}
};

/**
 * A 2 dimensional stream where the data is written in row direction and read in column direction.
 */
template <typename T> class D2Stream {
private:
	std::vector<int> _len2;
	std::vector<T*> _data;

	int _offset_len;
	int _read_index = 0;
	int** _offsets = NULL;
	T* _deviceBuffer = NULL;

	void check_input_write(T* newData, int n) {
		if ((n != 0) && (newData == NULL))
			print_err("D2Stream: (n != 0) && (newData == NULL)");
	}

	void check_input_offsets(int** offsets, int nRow, int nColumn) {
		if (offsets == NULL)
			print_err("D2Stream: offsets == NULL");
		if (nRow != _data.size())
			print_err("D2Stream: nRow !=_ data.size()");
		for (int i = 0; i < nRow; i++) {
			if (offsets[i] == NULL)
				print_err("D2Stream: offsets[i]==NULL");
			if (offsets[i][nColumn - 1] != _len2[i])
				print_err("D2Stream: the last offset should cover the whole stream");
		}
	}

public:
	D2Stream() {/*do nothing*/}

	void write(T* newData, int n) {
		check_input_write(newData, n);
		_data.push_back((n > 0) ? device_to_host(newData, n) : NULL);
		_len2.push_back(n);
	}

	void set_offsets(int** offsets, int n, int offset_len) {
		check_input_offsets(offsets, n, offset_len);
		if (_deviceBuffer != NULL)
			print_err("D2Stream: set_offsets is called more than once");

		_offsets = offsets;
		_offset_len = offset_len;

		// find the largest column size to allocate deviceBuffer
		int maxLength = 0;
		int _len1 = _data.size();
		for (int i = 0; i < offset_len; i++) {
			int newLength = 0;
			for (int j = 0; j < _len1; j++) {
				int start = i == 0 ? 0 : offsets[j][i - 1];
				newLength += offsets[j][i] - start;
			}
			if (newLength > maxLength)
				maxLength = newLength;
		}
		cudaMalloc(&_deviceBuffer, sizeof(T)*maxLength); gpuerr();
	}

	Chunk<T> read() {
		Chunk<T> ans;
		if (_offsets == NULL)
			print_err("D2Stream: _offsets == NULL");
		if (_read_index == _offset_len)
			return ans;

		ans.ptr = _deviceBuffer;
		T * currentPtr = _deviceBuffer;
		int _len1 = _data.size();
		for (int i = 0; i < _len1; i++) {
			int start = _read_index == 0 ? 0 : _offsets[i][_read_index - 1];
			int chunkLen = _offsets[i][_read_index] - start;
			if (chunkLen <= 0)
				continue;

			cudaMemcpy(currentPtr, _data[i] + start, sizeof(T)*chunkLen, cudaMemcpyHostToDevice); gpuerr();
			currentPtr += chunkLen;
			ans.len += chunkLen;
		}

		_read_index++;
		return ans;
	}

	void deconstruct() {
		for (T* rowData : _data)
			cudaFreeHost(rowData); gpuerr();
		_data.clear();
		_len2.clear();
		cudaFree(_deviceBuffer); gpuerr();
		_offsets = NULL; /*do not free offset as it can be shared across buffers*/
	}

	size_t get_throughput() {
		size_t ans = 0;
		for (int newLen : _len2)
			ans += newLen;
		return ans;
	}
};
