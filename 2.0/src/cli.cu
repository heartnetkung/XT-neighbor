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
                         "\t====================\n\t Nearest Neighbor Options\n\t====================\n"
                         "\t -i or --input-path [str] (required): path of csv input file containing a single column of CDR3 amino acid sequences\n"
                         "\t -n or --input-length [number] (required): number of rows given in the input file\n"
                         "\t====================\n\t Repertoire Overlap Options\n\t====================\n"
                         "\t -i or --input-path [str] (required): path of csv input file containing 2 columns: CDR3 amino acid sequences and their frequency. Note that the sequences are assumed to be unique\n"
                         "\t -n or --input-length [number] (required): number of sequences given in the input file\n"
                         "\t -r or --repertoire-info [str] (required): path of csv input file containing 2 columns: repertoire names, and their sizes. Note that the order of input sequence must be sorted according to this repertoire info\n"
                         "\t -N or --info-length [number] (required): number of repertoires given in the input file\n"
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
		else if (strcmp(current, "-p") == 0 || strcmp(current, "--input-path") == 0)
			ans->seq1Path = argv[++i];
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

	return SUCCESS;
}

/**
 * read and parse text file to Int3*
*/
int parse_file(char* path, Int3* result, int len) {
	FILE* file = fopen(path, "r");
	if (file == NULL)
		return print_err("file reading failed");

	const int BUFFER_SIZE = 50;
	char line[BUFFER_SIZE];
	int lineNumber = 0, inputCount = 0;
	Int3 newInt3;

	while (fgets(line, BUFFER_SIZE, file)) {
		lineNumber++;
		if (strcmp(line, "\n") == 0 || strcmp(line, " \n") == 0)
			continue;

		newInt3 = str_encode(line);
		if (newInt3.entry[0] == 0) {
			fclose(file);
			char msg[100];
			sprintf(msg, "parsing error at line: %d (only upper-cased amino acids with max length of %d are allowed)", lineNumber, MAX_INPUT_LENGTH);
			return print_err(msg);
		}

		result[inputCount++] = newInt3;
	}

	if (inputCount != len)
		return print_err("input length doesn't match with the actual");

	fclose(file);
	return SUCCESS;
}

void null_handler(XTNOutput output) {
	totalOutputLen += output.len;
}

void file_handler(XTNOutput output) {
	Int2 current;
	for (int i = 0; i < output.len; i++) {
		current = output.indexPairs[i];
		fprintf(outputFile, "%d %d %d\n", current.x, current.y , output.pairwiseDistances[i]);
	}
	totalOutputLen += output.len;
}

int main(int argc, char **argv) {
	int returnCode;
	XTNArgs args;
	Int3* seq1;

	// 1. parse command line arguments
	setlocale(LC_ALL, "");
	returnCode = parse_args(argc, argv, &args);
	verboseGlobal = args.verbose;
	if (returnCode != SUCCESS)
		return returnCode;

	// 2. read input
	cudaMallocHost(&seq1, sizeof(Int3) * args.seq1Len); gpuerr();
	returnCode = parse_file(args.seq1Path, seq1, args.seq1Len);
	if (returnCode != SUCCESS) {
		cudaFree(seq1); gpuerr();
		return returnCode;
	}
	if (verboseGlobal)
		print_args(args);

	// 3. perform algorithm
	if (args.outputPath != NULL) {
		if (outputFile != NULL)
			return print_err("output file has already been allocated, possibly due to concurrency");
		outputFile = fopen(args.outputPath, "w");
		if (outputFile == NULL)
			return print_err("file reading failed");
		xtn_perform(args, seq1, file_handler);
		fclose(outputFile);
	} else {
		xtn_perform(args, seq1, null_handler);
	}
	printf("total output length: %'lu\n", totalOutputLen);

	// 4. clean up
	cudaFreeHost(seq1); gpuerr();
	return 0;
}