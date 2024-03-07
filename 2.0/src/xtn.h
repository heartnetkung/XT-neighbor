/**
 * @file
 * Listing of all shared data structures, constants, and exporting method.
 */

/**CDR3 length limit for compression*/
const int MAX_INPUT_LENGTH = 18;
/**distance limit for immunology use case*/
const int MAX_DISTANCE = 2;
/**control printing of each method*/
int verboseGlobal = 0;
/**enum for distant type*/
const char LEVENSHTEIN = 0;
/**enum for distant type*/
const char HAMMING = 1;

/**
 * 12-byte integer representation for CDR3 string.
*/
struct Int3 {
	unsigned int entry[3] = {0, 0, 0};
	__device__
	bool operator==(const Int3& t) const {
		return (entry[0] == t.entry[0]) && (entry[1] == t.entry[1]) && (entry[2] == t.entry[2]);
	}
};

/**
 * 8-byte integer representation for pair of sequence index.
*/
struct Int2 {
	int x = 0, y = 0;
	__device__
	bool operator==(const Int2& t) const {
		return (x == t.x) && (y == t.y);
	}
};

/**
 * bundled representation for command line arguments.
*/
struct XTNArgs {
	int distance = 1;
	int verbose = 0;
	char* seq1Path = NULL;
	int seq1Len = 0;
	char* outputPath = NULL;
	char measure = LEVENSHTEIN;
	char* infoPath = NULL;
	int infoLen = 0;
};

/**
 * bundled representation for memory constraints and info.
*/
struct MemoryContext {
	size_t gpuSize = 0;
	size_t ramSize = 0;
	int bandwidth1 = 0;
	int bandwidth2 = 0;
	int chunkSize = 0;
#ifdef TEST_ENV
	int histogramSize = 16;
	int maxThroughputExponent = 7;
#else
	int histogramSize = 65536;
	int maxThroughputExponent = 0;
#endif
};

/**
 * bundled representation for algorithm's output.
*/
struct XTNOutput {
	Int2* indexPairs = NULL;
	char* pairwiseDistances = NULL;
	size_t* pairwiseFrequencies = NULL;
	int len = 0;
};

/**
 * control-flow representation.
*/
enum ReturnCode {SUCCESS, ERROR, EXIT};

/**
 * the algorithm's API.
 *
 * @param args algorithm's bundled arguments
 * @param seq1 list of CDR3 sequences
 * @param seqFreq frequency of each CDR3 sequence, only used in overlap mode
 * @param repSizes size of each repertiore, only used in overlap mode
 * @param callback function to be invoked once a chunk of output is ready
*/
void xtn_perform(XTNArgs args, Int3* seq1, int* seqFreq, int* repSizes, void callback (XTNOutput));
