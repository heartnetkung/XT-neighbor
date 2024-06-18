#include <stdio.h>
#include <string.h>
#include "codec.cu"

const int DEFAULT_MAX_COLUMN = 300, DEFAULT_LINE_SIZE = 5000;
const char DELIMITER = '\t';
const char CDR_FIELD[8] = "cdr3_aa", FREQ_FIELD[16] = "duplicate_count";

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

int find_idx(char* haystack, const char* needle, int* offsets, int offsetLen) {
	char* startPos = strstr(haystack, needle);
	int startPosInt = startPos - haystack;
	for (int i = 0; i < offsetLen; i++)
		if (offsets[i] == startPosInt)
			return i;
	return -1;
}

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

int extract_data(char* line, int cdrIndex, int freqIndex, int nColumn, int lineNumber,
                 Int3 &seqOut, int &freqOut, bool doubleCol) {
	int* offsets;
	cudaMallocHost(&offsets, nColumn * sizeof(int)); gpuerr();
	int offsetLen = split_offset(line, offsets, nColumn);
	if (offsetLen <= cdrIndex) {
		cudaFreeHost(offsets);
		return ERROR; // empty line
	}

	seqOut = str_encode(line + offsets[cdrIndex]);
	if (seqOut.entry[0] == 0) {
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

int parse_airr_input(char* path, Int3* seqOut, SeqInfo* freqOut, int len, bool doubleCol) {
	FILE* file = fopen(path, "r");
	if (file == NULL)
		return print_err("input file reading failed");

	char line[DEFAULT_LINE_SIZE];
	int lineNumber = -1, inputCount = -1, seqIndex = -1, freqIndex = -1;

	// read header line
	fgets(line, DEFAULT_LINE_SIZE, file);
	int nColumn = find_header(line, seqIndex, freqIndex, doubleCol);

	while (fgets(line, DEFAULT_LINE_SIZE, file)) {
		lineNumber++;
		if (strcmp(line, "\n") == 0 || strcmp(line, " \n") == 0)
			continue;

		Int3 seq;
		int freq;
		int success = extract_data(line, seqIndex, freqIndex, nColumn, lineNumber, seq, freq, doubleCol);
		if (!success){
			fclose(file);
			return print_err_line("",lineNumber);
		}

		seqOut[inputCount] = seq;
		if (doubleCol)
			freqOut[inputCount].frequency = freq;

		inputCount++;
	}

	fclose(file);
	if (inputCount != len) {
		printf("ab %d %d", inputCount, len);
		return print_err("input length doesn't match with the actual");
	}

	return SUCCESS;
}