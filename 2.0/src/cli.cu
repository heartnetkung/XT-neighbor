#include <stdio.h>
#include <locale.h>
#include "xtn.cu"

FILE* outputFile = NULL; /*global variable for callback*/
size_t totalOutputLen = 0; /*global variable for callback*/

const char VERSION[] = "2.0.0\n";
const char HELP_TEXT[] = "xt_neighbor: perform either nearest neighbor search for CDR3 sequences or immune repertoire overlap using GPU-based xt_neighbor algorithm.\n"
                         "\t====================\n\t Common Options\n\t====================\n"
                         "\t -d or --distance [number]: distance threshold defining the neighbor (default to 1)\n"
                         "\t -o or --output-path [str]: path of the output file (default to no output)\n"
                         "\t -m or --measurement [leven|hamming]: distance measurement (default to leven)\n"
                         "\t -v or --version: print the version of the program then exit\n"
                         "\t -h or --help: print the help text of the program then exit\n"
                         "\t -V or --verbose: print extra detail as the program runs for debugging purpose\n"
                         "\t -x or --extended-buffer: switch to the alternate buffer implementation that can be enlarged using disc memory\n"
                         "\t====================\n\t Nearest Neighbor Options\n\t====================\n"
                         "\t -i or --input-path [str] (required): path of csv input file containing exactly 1 column: CDR3 amino acid sequences\n"
                         "\t -n or --input-length [number] (required): number of rows given in the input file\n"
                         "\t====================\n\t Repertoire Overlap Options\n\t====================\n"
                         "\t -i or --input-path [str] (required): path of csv input file containing exactly 2 columns: CDR3 amino acid sequences and their frequency. Note that the sequences are assumed to be unique\n"
                         "\t -n or --input-length [number] (required): number of sequences given in the input file\n"
                         "\t -r or --info-path [str] (required): path of csv input file containing exactly 1 column: repertoire sizes. Note that the order of input sequence must be sorted according to this repertoire info\n"
                         "\t -N or --info-length [number] (required): number of repertoires given in the info file\n"
                         ;

int parse_args(int argc, char **argv, XTNArgs* ans) {
	char* current;

	for (int i = 1; i < argc; i++) {
		current = argv[i];

		if (strcmp(current, "-v") == 0 || strcmp(current, "--version") == 0) {
			printf("%s", VERSION);
			return EXIT;
		}
		else if (strcmp(current, "-h") == 0 || strcmp(current, "--help") == 0) {
			printf("%s", HELP_TEXT);
			return EXIT;
		}
		else if (strcmp(current, "-V") == 0 || strcmp(current, "--verbose") == 0)
			ans->verbose = 1;
		else if (strcmp(current, "-x") == 0 || strcmp(current, "--extended-buffer") == 0)
			ans->extendedBuffer = 1;
		else if (strcmp(current, "-p") == 0 || strcmp(current, "--input-path") == 0)
			ans->seq1Path = argv[++i];
		else if (strcmp(current, "-r") == 0 || strcmp(current, "--info-path") == 0)
			ans->infoPath = argv[++i];
		else if (strcmp(current, "-o") == 0 || strcmp(current, "--output-path") == 0)
			ans->outputPath = argv[++i];
		else if (strcmp(current, "-d") == 0 || strcmp(current, "--distance") == 0) {
			int distance = ans->distance = atoi(argv[++i]);
			if (distance < 1 || distance > MAX_DISTANCE)
				return print_err("distance must be a valid number ranging from 1-2");
		}
		else if (strcmp(current, "-n") == 0 || strcmp(current, "--input-length") == 0) {
			ans->seq1Len = atoi(argv[++i]);
			if (ans->seq1Len == 0)
				return print_err("invalid input length");
		}
		else if (strcmp(current, "-N") == 0 || strcmp(current, "--info-length") == 0) {
			ans->infoLen = atoi(argv[++i]);
			if (ans->infoLen == 0)
				return print_err("invalid info length");
		}
		else if (strcmp(current, "-m") == 0 || strcmp(current, "--measurement") == 0) {
			char* measure = argv[++i];
			if (strcmp(measure, "leven") == 0)
				ans->measure = LEVENSHTEIN;
			else if (strcmp(measure, "hamming") == 0)
				ans->measure = HAMMING;
			else
				return print_err("invalid measure option");
		}
		else
			return print_err("unknown option");
	}

	if (ans->seq1Path == NULL)
		return print_err("missing path for seq1");
	if (ans->seq1Len == 0)
		return print_err("missing length for seq1");
	if ((ans->infoPath == NULL) != (ans->infoLen == 0) )
		return print_err("repertiore path or repertoire count is missing in overlap mode");

	return SUCCESS;
}

/**
 * read and parse input csv file to Int3* and maybe int*
*/
int parse_input(char* path, Int3* seqOut, int* freqOut, int len, bool doubleCol) {
	FILE* file = fopen(path, "r");
	if (file == NULL)
		return print_err("file reading failed");

	const int BUFFER_SIZE = 50;
	char line[BUFFER_SIZE];
	int lineNumber = 0, inputCount = 0;

	// ignore header
	fgets(line, BUFFER_SIZE, file);

	while (fgets(line, BUFFER_SIZE, file)) {
		lineNumber++;
		if (strcmp(line, "\n") == 0 || strcmp(line, " \n") == 0)
			continue;

		Int3 newInt3 = str_encode(line);
		if (newInt3.entry[0] == 0) {
			fclose(file);
			return print_err_line("parsing error (only upper-cased amino acids with max length of 18 are allowed)", lineNumber);
		}
		seqOut[inputCount++] = newInt3;

		if (doubleCol) {
			char* line2 = strchr(line, ',');
			if (line2 == NULL) {
				fclose(file);
				return print_err_line("parsing error (comma expected)", lineNumber);
			}

			long int temp = strtol(line2);
			if (temp == 0 || temp > INT_MAX || temp < INT_MIN) {
				fclose(file);
				return print_err_line("parsing error (invalid number)", lineNumber);
			}
			freqOut[inputCount] = temp;
		}
	}

	fclose(file);
	if (inputCount != len)
		return print_err("input length doesn't match with the actual");

	return SUCCESS;
}

/**
 * read and parse info csv file to int*
*/
int parse_info(char* path, int* result, int len) {
	FILE* file = fopen(path, "r");
	if (file == NULL)
		return print_err("file reading failed");

	const int BUFFER_SIZE = 20;
	char line[BUFFER_SIZE];
	int lineNumber = 0, inputCount = 0;

	// ignore header
	fgets(line, BUFFER_SIZE, file);

	while (fgets(line, BUFFER_SIZE, file)) {
		lineNumber++;
		if (strcmp(line, "\n") == 0 || strcmp(line, " \n") == 0)
			continue;

		long int temp = strtol(line);
		if (temp == 0 || temp > INT_MAX || temp < INT_MIN) {
			fclose(file);
			return print_err_line("parsing error (invalid number)", lineNumber);
		}
		result[inputCount++] = temp;
	}

	fclose(file);
	if (inputCount != len)
		return print_err("info length doesn't match with the actual");

	return SUCCESS;
}

void null_handler(XTNOutput output) {
	totalOutputLen += output.len;
}

void file_handler_nn(XTNOutput output) {
	Int2 current;
	for (int i = 0; i < output.len; i++) {
		current = output.indexPairs[i];
		fprintf(outputFile, "%d %d %d\n", current.x, current.y , output.pairwiseDistances[i]);
	}
	totalOutputLen += output.len;
}

void file_handler_overlap(XTNOutput output) {
	Int2 current;
	for (int i = 0; i < output.len; i++) {
		current = output.indexPairs[i];
		fprintf(outputFile, "%d %d %'lu\n", current.x, current.y , output.pairwiseFrequencies[i]);
	}
	totalOutputLen += output.len;
}

int sum_check(int* repSizes, int seqLen, int n) {
	int sum = 0;
	for (int i = 0; i < n; i++)
		sum += repSizes[i];
	if (sum != seqLen)
		return print_err("total repertoires' size does not match sequence count");
	return SUCCESS;
}

int free_all(Int3* seq1, int* seqFreq, int* repSizes, int returnCode) {
	cudaFreeHost(seq1); gpuerr();
	if (seqFreq != NULL) {
		cudaFreeHost(seqFreq); gpuerr();
	}
	if (repSizes != NULL) {
		cudaFreeHost(repSizes); gpuerr();
	}
	return returnCode;
}

int main(int argc, char **argv) {
	XTNArgs args;
	int returnCode;
	Int3* seq1;
	int* seqFreq = NULL, *repSizes = NULL;

	// 1. parse command line arguments
	setlocale(LC_ALL, "");
	returnCode = parse_args(argc, argv, &args);
	verboseGlobal = args.verbose;
	if (returnCode != SUCCESS)
		return returnCode;

	// 2. read input
	bool overlapMode = args.infoPath != NULL;
	if (overlapMode) {
		cudaMallocHost(&repSizes, sizeof(int) * args.infoLen); gpuerr();
		returnCode = parse_info(args.infoPath, repSizes, args.infoLen);
		if (returnCode != SUCCESS)
			return free_all(seq1, seqFreq, repSizes, returnCode);
		cudaMallocHost(&seqFreq, sizeof(int) * args.seq1Len); gpuerr();
	}
	cudaMallocHost(&seq1, sizeof(Int3) * args.seq1Len); gpuerr();
	returnCode = parse_input(args.seq1Path, seq1, seqFreq, args.seq1Len, overlapMode);
	if (returnCode != SUCCESS)
		return free_all(seq1, seqFreq, repSizes, returnCode);
	returnCode = sum_check(repSizes, args.seq1Len, args.infoLen);
	if (returnCode != SUCCESS)
		return free_all(seq1, seqFreq, repSizes, returnCode);

	// 3. perform algorithm
	if (verboseGlobal)
		print_args(args);
	if (args.outputPath != NULL) {
		if (outputFile != NULL)
			return print_err("output file has already been allocated, possibly due to concurrency");
		outputFile = fopen(args.outputPath, "w");
		if (outputFile == NULL)
			return print_err("file reading failed");
		xtn_perform(args, seq1,
		            overlapMode ? file_handler_overlap : file_handler_nn);
		fclose(outputFile);
	} else {
		xtn_perform(args, seq1, null_handler);
	}

	printf("total output length: %'lu\n", totalOutputLen);
	return free_all(seq1, seqFreq, repSizes, SUCCESS);
}