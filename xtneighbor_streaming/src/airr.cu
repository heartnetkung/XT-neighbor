#include <stdio.h>
#include <string.h>
#include "codec.cu"

/**
 * @file
 * A collection of patching functions that add airr-format compatibility to cli.cu.
 */

/**maximum number of tsv columns*/
const int DEFAULT_MAX_COLUMN = 300;
/**maximum number of characters in a line*/
const int DEFAULT_LINE_SIZE = 5000;
/**tsv delimitter*/
const char DELIMITER = '\t';
/**field header for sequence data*/
const char CDR_FIELD[8] = "cdr3_aa";
/**field header for sequence data*/
const char FREQ_FIELD[16] = "duplicate_count";

/**
 * private function
 */
int split_offset(char* str, int* &offsetOutput, int nColumn) {
	char c;
	int i = 0, count = 1;
	offsetOutput[0] = 0;

	while ((c = str[i++]) != '\0') {
		if (c == DELIMITER) {
			offsetOutput[count++] = i;
			if (count == nColumn)
				return count;
		}
	}
	return count;
}

/**
 * private function
 */
int find_idx(char* haystack, const char* needle, int* offsets, int offsetLen) {
	char* startPos = strstr(haystack, needle);
	int startPosInt = startPos - haystack;
	for (int i = 0; i < offsetLen; i++)
		if (offsets[i] == startPosInt)
			return i;
	return -1;
}

/**
 * private function
 */
int find_header(char* line, int &cdrIndexOut, int &freqIndexOut, bool doubleCol) {
	int* offsets;
	cudaMallocHost(&offsets, DEFAULT_MAX_COLUMN * sizeof(int)); gpuerr();
	int offsetLen = split_offset(line, offsets, DEFAULT_MAX_COLUMN);

	cdrIndexOut = find_idx(line, CDR_FIELD, offsets, offsetLen);
	if (doubleCol)
		freqIndexOut = find_idx(line, FREQ_FIELD, offsets, offsetLen);
	else
		freqIndexOut = -1;
	cudaFreeHost(offsets);
	return offsetLen;
}

/**
 * private function
 */
int extract_data(char* line, int cdrIndex, int freqIndex, int nColumn, int lineNumber,
                 SeqArray* seqOut, int &freqOut, bool doubleCol) {
	int* offsets;
	cudaMallocHost(&offsets, nColumn * sizeof(int)); gpuerr();
	int offsetLen = split_offset(line, offsets, nColumn);
	if (offsetLen <= cdrIndex) {
		cudaFreeHost(offsets);
		return ERROR; // empty line
	}

	int appendResult = seqOut->append(line + offsets[cdrIndex]);
	if (appendResult == ERROR) {
		cudaFreeHost(offsets);
		return print_err_line("invalid seq", lineNumber);
	}

	if (doubleCol) {
		if (offsetLen <= freqIndex) {
			cudaFreeHost(offsets);
			return ERROR; // empty line
		}

		long int temp = strtol(line + offsets[freqIndex], NULL, 10);
		if (temp == 0 || temp > INT_MAX || temp < INT_MIN) {
			cudaFreeHost(offsets);
			return print_err_line("invalid number", lineNumber);
		}

		freqOut = temp;
	} else
		freqOut = -1;
	cudaFreeHost(offsets);
	return SUCCESS;
}

/**
 * read and parse input airr file to SeqArray and maybe SeqInfo
 *
 * @param path file path to read
 * @param seqOut output sequences
 * @param freqOut output frequency, if doubleCol is set to true
 * @param len expected length
 * @param doubleCol if true, read freqOut
 * @return execution result
*/
int parse_airr_input(char* path, SeqArray* seqOut, SeqInfo* freqOut, int len, bool doubleCol) {
	FILE* file = fopen(path, "r");
	if (file == NULL)
		return print_err("input file reading failed");

	char line[DEFAULT_LINE_SIZE];
	int lineNumber = -1, inputCount = 0, seqIndex = -1, freqIndex = -1;

	// read header line
	fgets(line, DEFAULT_LINE_SIZE, file);
	int nColumn = find_header(line, seqIndex, freqIndex, doubleCol);

	while (fgets(line, DEFAULT_LINE_SIZE, file)) {
		lineNumber++;
		if (strcmp(line, "\n") == 0 || strcmp(line, " \n") == 0)
			continue;

		int freq;
		int success = extract_data(line, seqIndex, freqIndex, nColumn, lineNumber, seqOut, freq, doubleCol);
		if (success != SUCCESS) {
			fclose(file);
			return print_err_line("line parsing error", lineNumber);
		}

		if (doubleCol)
			freqOut[inputCount].frequency = freq;

		inputCount++;
	}

	fclose(file);
	if (inputCount != len)
		return print_err("input length doesn't match with the actual");

	return SUCCESS;
}