const int MAX_INPUT_LENGTH = 18;
const int MAX_DISTANCE = 2;

struct Int3 {
	unsigned int entry[3] = {0, 0, 0};
	__device__
	bool operator==(const Int3& t) const {
		return (entry[0] == t.entry[0]) && (entry[1] == t.entry[1]) && (entry[2] == t.entry[2]);
	}
};

struct Int2 {
	int x = 0, y = 0;
	__device__
	bool operator==(const Int2& t) const {
		return (x == t.x) && (y == t.y);
	}
};

struct XTNArgs {
	int distance = 1;
	int verbose = 0;
	char* seq1Path = NULL;
	int seq1Len = 0;
	char* outputPath = NULL;
	// Int3* seq2 = NULL;
	// int seq2Len = 0;
};

struct MemoryContext {
	size_t gpuSize = 0;
	size_t ramSize = 0;
	size_t bandwidth1 = 0;
	size_t bandwidth2 = 0;
	int chunkSize = 0;
#ifdef TEST_ENV
	int histogramSize = 16;
	int maxThroughputExponent = 7;
#else
	int histogramSize = 4096;
	int maxThroughputExponent = 20;
#endif
};

struct XTNOutput {
	Int2* indexPairs = NULL;
	char* pairwiseDistances = NULL;
	int len = 0;
};

enum ReturnCode {SUCCESS, ERROR, EXIT};

void xtn_perform(XTNArgs args, Int3* seq1, void callback (XTNOutput));
