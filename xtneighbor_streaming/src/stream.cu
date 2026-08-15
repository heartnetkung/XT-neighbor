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

	/*(re)allocate the device buffer to hold exactly newCapacity elements*/
	void resize_buffer(int newCapacity) {
		if (newCapacity == _maxReadableSize)
			return;
		if (_deviceBuffer != NULL) {
			cudaFree(_deviceBuffer); gpuerr();
		}
		if (verboseGlobal)
			printf("====size %s %'d\n", newCapacity > _maxReadableSize ? "grow" : "shrink", newCapacity);
		_maxReadableSize = newCapacity;
		cudaMalloc(&_deviceBuffer, sizeof(T)*newCapacity); gpuerr();
	}

public:
	size_t get_total_len() {
		return _totalLen;
	}

	int get_max_readable_size() {
		return _maxReadableSize;
	}

	/**
	 * size the device buffer for the round that is about to be read.
	 *
	 * maxReadableSize is the caller's budget for the round (typically a MemoryContext.bandwidth
	 * freshly recomputed from currently-free GPU memory). The buffer is sized to that budget,
	 * except that it is never smaller than the longest row still queued for reading, since
	 * read() never splits a row, and never larger than the whole queue, since anything beyond
	 * it can never be filled. Unlike before, the buffer may also shrink: a single oversized row
	 * used to enlarge it permanently for the remainder of the run, which is what made the
	 * footprint ratchet upwards.
	 *
	 * Must be called after swap(), so that the reading queue it measures is the one to be read.
	*/
	void set_max_readable_size(int maxReadableSize) {
		if (maxReadableSize <= 0) {
			print_err("RAMSwapStream: maxReadableSize <= 0");
			return;
		}
		if (_reading_len2.empty())
			return; /*nothing to read, so leave the buffer alone rather than size it to zero*/

		size_t totalQueued = 0;
		int longestRow = 0;
		for (int len : _reading_len2) {
			totalQueued += len;
			if (len > longestRow)
				longestRow = len;
		}

		/*late lower bound rounds leave only a handful of entries queued. a full budget-sized
		  buffer would then stay resident for the rest of stream 3 and all of stream 4, since
		  deconstruct only runs at the end of the run, shrinking the free memory that
		  cal_memory_stream4 measures for no benefit.*/
		int capacity = maxReadableSize;
		if (totalQueued < (size_t)capacity)
			capacity = (int)totalQueued;
		if (capacity < longestRow)
			capacity = longestRow;
		if (verboseGlobal && (capacity > maxReadableSize))
			printf("====size row of %'d exceeds the budget %'d, buffer sized to the row\n",
			       capacity, maxReadableSize);
		resize_buffer(capacity);
	}

	/**
	 * read the next chunk, packing whole rows only.
	 *
	 * maxReadableSize is the budget for this call (typically a MemoryContext.bandwidth freshly
	 * recomputed from currently-free GPU memory by the caller); it may be smaller than a
	 * previous call's, since memory pressure can grow over a long-running pipeline, so it is
	 * taken fresh each call rather than remembered the way the buffer capacity is.
	 *
	 * A row is never cut. Stream 3 groups equal deletion keys purely by adjacency within the
	 * chunk it is handed (unique_counts in cal_offsets_lowerbound), and the upstream hash-bin
	 * packing guarantees that equal keys share a row; cutting a row mid-group would therefore
	 * split one group into two and silently drop every pair that crosses the cut. A chunk is
	 * consequently returned short of the budget whenever the next row does not fit, and a row
	 * longer than the budget is still returned whole - set_max_readable_size has already sized
	 * the buffer for it.
	*/
	Chunk<T> read(int maxReadableSize) {
		Chunk<T> ans;
		if (_reading_data.empty())
			return ans;
		if (_deviceBuffer == NULL) {
			print_err("RAMSwapStream: _deviceBuffer == NULL");
			return ans;
		}
		if (maxReadableSize > _maxReadableSize)
			maxReadableSize = _maxReadableSize; /*can't exceed the allocated buffer's capacity*/

		int totalLen = 0;
		T* ptr = _deviceBuffer;
		while (!_reading_data.empty()) {
			int len = _reading_len2.back();

			if (totalLen == 0) {
				/*the first row is always taken so that the stream cannot stall. it only fails
				  to fit when set_max_readable_size was not called after the swap, in which case
				  grow to honour it - nothing has been copied yet, and the growth is no longer
				  permanent now that the buffer can shrink again.*/
				if (len > _maxReadableSize) {
					printf("RAMSwapStream: row of %'d exceeds buffer capacity %'d, growing\n",
					       len, _maxReadableSize);
					resize_buffer(len);
					ptr = _deviceBuffer;
				}
			} else if (totalLen + len > maxReadableSize)
				break;

			T* dataHost = _reading_data.back();
			cudaMemcpy(ptr, dataHost, sizeof(T)*len , cudaMemcpyHostToDevice); gpuerr();
			cudaFreeHost(dataHost); gpuerr();
			_reading_data.pop_back();
			_reading_len2.pop_back();
			ptr += len;
			totalLen += len;
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

	/**
	 * free the device buffer while keeping the queued rows.
	 *
	 * meant to be called between rounds, right before the caller measures free GPU memory to
	 * derive the next round's budget: the buffer is about to be replaced anyway, and counting it
	 * as in-use makes every round's budget smaller than the previous one's. Safe at that point
	 * because read() has already copied every consumed row out of the buffer and write() copies
	 * to pinned host memory, so nothing points into it between rounds.
	*/
	void release_buffer() {
		cudaFree(_deviceBuffer); gpuerr();
		_deviceBuffer = NULL;
		_maxReadableSize = 0;
	}

	void deconstruct() {
		release_buffer();
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
