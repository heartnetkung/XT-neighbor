#ifndef CODEC_DEF
#define CODEC_DEF

#include "util.cu"
#include <string.h>

// A CDEFGHI KLMN PQRST VW Y
const int A_CHAR = (int)'A';
const int BEFORE_A_CHAR = A_CHAR - 1;
const int Y_CHAR = (int) 'Y';

/**
 * encode character into 5 bit value (0-31).
 * -1 for non amino acid character
*/
int char_encode(char amino_acid) {
	if (amino_acid < A_CHAR || amino_acid > Y_CHAR)
		return -1;
	switch (amino_acid) {
	case 'B':
	case 'J':
	case 'O':
	case 'U':
	case 'X':
		return -1;
	default:
		return amino_acid - BEFORE_A_CHAR;
	}
}

/**
 * encode peptide string into int3 struct with 6 characters encoded into an integer.
*/
Int3 str_encode(char *str) {
	Int3 ans;
	for (int i = 0; i < MAX_INPUT_LENGTH; i++) {
		char c = str[i];
		if (c == '\0' || c == '\n')
			break; // end

		int value = char_encode(c);
		if (value == -1) {
			ans.entry[0] = 0;
			break; // invalid character
		}

		ans.entry[i / 6] |= value << (27 - 5 * (i % 6));
	}

	return ans;
}


/**
 * decode binary form into peptide string
*/
char* str_decode(Int3 binary) {
	char* ans = (char*) malloc((MAX_INPUT_LENGTH + 1) * sizeof(char));

	for (int i = 0; i < MAX_INPUT_LENGTH; i++) {
		char c = (binary.entry[i / 6] >> (27 - 5 * (i % 6))) & 0x1F;
		if (c == 0) {
			ans[i] = '\0';
			return ans;
		}

		ans[i] = BEFORE_A_CHAR + c;
	}

	ans[MAX_INPUT_LENGTH] = '\0';
	return ans;
}

/**
 * find the string length in original form
*/
__device__ __host__
int len_decode(Int3 binary) {
	int ans = 18;
	int lastIndex = 2;
	unsigned int* entry = binary.entry;
	if (entry[2] == 0) {
		ans -= 6;
		lastIndex = 1;
	}
	if (entry[1] == 0) {
		ans -= 6;
		lastIndex = 0;
	}
	for (int i = 5; i >= 0; i--) {
		char c = (binary.entry[lastIndex] >> (27 - 5 * i)) & 0x1F;
		if (c != 0)
			break;
		ans--;
	}
	return ans;
}

/**
 * remove one character from the string in binary form (without turning it into the original string form)
*/
__device__ __host__
Int3 remove_char(Int3 binary, int position) {
	Int3 ans;
	unsigned int* ansEntry = ans.entry;
	unsigned int* binEntry = binary.entry;
	unsigned int b2 = binEntry[2], b1 = binEntry[1], b0 = binEntry[0];

	// approximate shift
	ansEntry[2] = b2 << 5;
	if (position < 12)
		ansEntry[1] = (b1 << 5) | ((b2 & 0xF8000000) >> 25);
	else
		ansEntry[1] = b1;
	if (position < 6)
		ansEntry[0] = (b0 << 5) | ((b1 & 0xF8000000) >> 25);
	else
		ansEntry[0] = b0;

	// handle boundary
	int lastIndex = position / 6;
	switch (position % 6) {
	case 1:
		ansEntry[lastIndex] = (binEntry[lastIndex] & 0xF8000000) | (ansEntry[lastIndex] & 0x7FFFFFF);
		break;
	case 2:
		ansEntry[lastIndex] = (binEntry[lastIndex] & 0xFFC00000) | (ansEntry[lastIndex] & 0x03FFFFF);
		break;
	case 3:
		ansEntry[lastIndex] = (binEntry[lastIndex] & 0xFFFE0000) | (ansEntry[lastIndex] & 0x001FFFF);
		break;
	case 4:
		ansEntry[lastIndex] = (binEntry[lastIndex] & 0xFFFFF000) | (ansEntry[lastIndex] & 0x0000FFF);
		break;
	case 5:
		ansEntry[lastIndex] = (binEntry[lastIndex] & 0xFFFFFF80) | (ansEntry[lastIndex] & 0x000007F);
		break;
	}//default do nothing
	return ans;
}

#endif