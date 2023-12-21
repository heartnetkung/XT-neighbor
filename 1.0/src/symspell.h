const int MAX_INPUT_LENGTH = 18;
// up to 3060? or 816?
const int MAX_DISTANCE = 4;

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

struct SymspellArgs {
	int distance = 1;
	int verbose = 0;
	char* seq1Path = NULL;
	int seq1Len = 0;
	char* outputPath = NULL;
	int checkOutput = 0;
	// Int3* seq2 = NULL;
	// int seq2Len = 0;
};

struct SymspellOutput {
	Int2* indexPairs = NULL;
	char* pairwiseDistances = NULL;
	size_t len = 0;
};

enum ReturnCode {SUCCESS, ERROR, EXIT};

void symspell_perform(SymspellArgs args, Int3* seq1, SymspellOutput* output);
void symspell_free(SymspellOutput* output);
