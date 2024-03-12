#include <vector>
#include <stdio.h>
#include <string>
#define TEST_ENV

std::vector<std::string> descriptions {};
std::vector<int> results {};
int successCount = 0;

void check(int result) {
	successCount++;
	if (results.back() && !result) {
		results.pop_back();
		results.push_back(0);
	}
}

void checkstr(const char* a, const char* b) {
	check(!strcmp(a, b));
}

template <typename T>
void check_device_arr(T* deviceArr, T* expectedArr, int n) {
	T* hostArr;
	size_t tempBytes = sizeof(T) * n;
	cudaMallocHost(&hostArr, tempBytes);
	cudaMemcpy(hostArr, arr, tempBytes, cudaMemcpyDeviceToHost);
	for (int i = 0; i < n; i++)
		check(hostArr[i] == expectedArr[i]);
	cudaFreeHost(hostArr);
}

int _append(std::string name, void (*func)()) {
	descriptions.push_back(name);
	results.push_back(1);
	func();
	return 0;
}

#define TEST(name, ...) void test_##name() { __VA_ARGS__ } int temp_##name = _append(#name,test_##name)?0:1;

int main() {
	int allSuccess = 1, current;
	for (int i = 0; i < results.size(); i++) {
		current = results.at(i);
		if (!current)
			allSuccess = 0;
	}

	if (allSuccess) {
		printf("\033[0;32mAll %d tests passed \033[0m\n", successCount);
		return 0;
	}

	for (int i = 0; i < results.size(); i++)
		if (!results.at(i))
			printf("\033[1;31m%d. %s failed!\033[0m\n", i + 1, descriptions.at(i).c_str());
	return 1;
}
