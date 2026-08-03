#include <vector>
#include <stdio.h>
#include <string>
#define TEST_ENV

std::vector<std::string> descriptions {};
std::vector<int> results {};

void check(int result) {
	if (results.back() && !result) {
		results.pop_back();
		results.push_back(0);
	}
}

void checkstr(const char* a, const char* b) {
	check(!strcmp(a, b));
}

int _append(std::string name, void (*func)()) {
	descriptions.push_back(name);
	results.push_back(1);
	func();
	return 0;
}

#define TEST(name, ...) void test_##name() { __VA_ARGS__ } int temp_##name = _append(#name,test_##name)?0:1;

int main() {
	int allSuccess = 1, countSuccess = 0, current;
	for (int i = 0; i < results.size(); i++) {
		current = results.at(i);
		if (current)
			countSuccess++;
		else
			allSuccess = 0;
	}

	if (allSuccess) {
		printf("\033[0;32mAll %lu tests passed \033[0m\n", results.size());
		return 0;
	}

	for (int i = 0; i < results.size(); i++) {
		if (!results.at(i))
			printf("\033[1;31m%d. %s \033[0m\n", i + 1, descriptions.at(i).c_str());
	}

	printf("\033[1;31mOnly %d/%lu tests passed \033[0m\n", countSuccess, results.size());
	return 1;
}
