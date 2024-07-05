#include <vector>
/**
 * @file
 * Listing of all shared data structures, constants, and exporting method.
 */

/**CDR3 length limit for compression*/
const int MAX_COMPRESSED_LENGTH = 18;
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
	__device__ __host__
	bool operator==(const Int3& t) const {
		return (entry[0] == t.entry[0]) && (entry[1] == t.entry[1]) && (entry[2] == t.entry[2]);
	}
};

/**
 * 8-byte integer representation for pair of sequence index.
*/
struct Int2 {
	int x = 0, y = 0;
	__device__ __host__
	bool operator==(const Int2& t) const {
		return (x == t.x) && (y == t.y);
	}
};

/**
 * Information related to a sequence
*/
struct SeqInfo {
	int frequency, repertoire;
	bool operator==(const SeqInfo& t) const {
		return (frequency == t.frequency) && (repertoire == t.repertoire);
	}
};

/**
 * bundled representation for command line arguments.
*/
struct XTNArgs {
	int distance = 1;
	int verbose = 0;
	char* seqPath = NULL;
	int seqLen = 0;
	char* outputPath = NULL;
	char measure = LEVENSHTEIN;
	char* infoPath = NULL;
	int infoLen = 0;
	int airr = 0;
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

class SeqArray {
private:
	unsigned int *offsets, *offsets_d = NULL;
	std::vector<char> seqs;
	char* seqs_d = NULL;
	int size = 0;
	int maxSize = 0;

public:
	SeqArray(int seqLen);
	int append(char* inputStr);
	__device__
	int getItem(int index, char* &result);
	int getItemCPU(int index, char* &result);
	void toDevice();
	void destroy();
	int getSize();
	char* getSeqs_d();
	unsigned int getOffsets_d();
};

/**
 * the algorithm's API.
 *
 * @param args algorithm's bundled arguments
 * @param seqArr list of CDR3 sequences
 * @param seqInfo information of each CDR3 sequence, only used in overlap mode
 * @param callback function to be invoked once a chunk of output is ready
*/
void xtn_perform(XTNArgs args, SeqArray seqArr, SeqInfo* seqInfo, void callback (XTNOutput));
