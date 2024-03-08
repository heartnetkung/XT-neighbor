#include "codec.cu"
#include <algorithm>

/**
 * @file
 * Listing of low-level streaming and buffering module.
 * It concerns how to read/write data to GPU, RAM, and filesystem with throughput constraints.
 */

/**
 * Chunk represents a unit of data read by the stream.
 */
template <typename T> class Chunk {
public:
	T* ptr = NULL;
	int len = 0;
	bool not_null() {return ptr != NULL;}
};


/**
 * A stream where the data is already fit inside the GPU memory, so
 * chunking is basically just shift the pointer around.
 */
template <typename T> class GPUInputStream {
private:
	T* _data = NULL;
	int _len = 0, _chunkSize = 0, _totalLen;

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
		_totalLen = len;
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

	size_t get_total_len() {
		return _totalLen;
	}
};

/**
 * A stream with bin packing capability backed by RAM. It contains 2 queues for duplex transfer: reading and writing queues.
 * The duplex works by taking turns at the end of each stream as follow. First, when the buffer is empty, it is write-only.
 * All writing is done to until depleted. Second, the "swap" is performed and the writing buffer now become reading now the writing buffer is cleared.
 * Third, start duplexing (read from reading buffer and write to writing buffer). Last, once stream is depleted, repeat second step.
 *
 * Note that the memory is immediately deallocated after each read to keep the memory requirement relatively constant.
 */
template <typename T> class RAMSwapStream {
private:
	std::vector<T*> _writing_data, _reading_data;
	std::vector<int> _writing_len2, _reading_len2;
	size_t _totalLen = 0;
	int _maxReadableSize = 0;
	T* _deviceBuffer = NULL;

	void check_readable_input(int maxReadableSize) {
		if (maxReadableSize <= 0)
			print_err("RAMSwapStream: maxReadableSize <= 0");
	}

public:
	size_t get_total_len() {
		return _totalLen;
	}

	void set_max_readable_size(int maxReadableSize) {
		check_readable_input(maxReadableSize);
		if (maxReadableSize <= _maxReadableSize)
			return;

		_maxReadableSize = maxReadableSize;
		if (_deviceBuffer != NULL) {
			cudaFree(_deviceBuffer); gpuerr();
		}
		printf("BB %lu\n",sizeof(T));
		cudaMalloc(&_deviceBuffer, sizeof(T)*maxReadableSize); gpuerr();
		if (verboseGlobal)
			printf("====size grow %'d\n", maxReadableSize);
	}

	Chunk<T> read() {
		printf("A0\n");
		Chunk<T> ans;
		if (_reading_data.empty())
			return ans;
		if (_deviceBuffer == NULL) {
			print_err("RAMSwapStream: _deviceBuffer == NULL");
			return ans;
		}

		int totalLen = 0;
		printf("AA\n");
		print_gpu_memory();
		print_main_memory();
		cudaFree(_deviceBuffer); gpuerr();
		printf("AB\n");
		cudaMalloc(&_deviceBuffer, sizeof(T)*_maxReadableSize); gpuerr();
		T* ptr = _deviceBuffer;
		while (true) {
			printf("AC\n");
			if (_reading_data.empty())
				break;

			int len = _reading_len2.back();
			if (totalLen + len > _maxReadableSize)
				break;

			printf("AD\n");

			T* dataHost = _reading_data.back();
			printf("the line\n");
			cudaMemcpy(ptr, dataHost, sizeof(T)*len , cudaMemcpyHostToDevice); gpuerr();
			_reading_data.pop_back();
			_reading_len2.pop_back();

			cudaFreeHost(dataHost); gpuerr();
			ptr += len;
			totalLen += len;
		}

		// when len exceed _maxReadableSize, enlarge the readable size, and faithfully read with warning
		if ((totalLen == 0) && !_reading_data.empty()) {
			totalLen = _reading_len2.back();
			set_max_readable_size(totalLen);

			T* dataHost = _reading_data.back();
			cudaMemcpy(_deviceBuffer, dataHost, sizeof(T)*totalLen , cudaMemcpyHostToDevice); gpuerr();
			_reading_data.pop_back();
			_reading_len2.pop_back();

			cudaFreeHost(dataHost); gpuerr();
		}

		ans.ptr = _deviceBuffer;
		ans.len = totalLen;
		return ans;
	}

	void write(T* newData, int n) {
		T* dataHost = device_to_host(newData, n);
		_writing_data.push_back(dataHost);
		_writing_len2.push_back(n);
		_totalLen += n;
	}

	void swap() {
		if (_reading_data.size() != 0) {
			print_err("RAMSwapStream: reading buffer should be empty before the swap");
			return;
		}
		_writing_data.swap(_reading_data);
		_writing_len2.swap(_reading_len2);
		std::reverse(_reading_data.begin(), _reading_data.end());
		std::reverse(_reading_len2.begin(), _reading_len2.end());
		_totalLen = 0;
	}

	void deconstruct() {
		cudaFree(_deviceBuffer); gpuerr();
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
		if (n > 0) {
			_data.push_back(device_to_host(newData, n));
		} else {
			T* dummy;
			cudaMallocHost(&dummy, sizeof(T)); gpuerr();
			_data.push_back(dummy);
		}
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
			if (_len2[i] <= 0)
				continue;

			// invalid value
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

	size_t get_total_len() {
		size_t ans = 0;
		for (int newLen : _len2)
			ans += newLen;
		return ans;
	}
};
