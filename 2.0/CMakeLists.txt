cmake_minimum_required(VERSION 3.23.1 FATAL_ERROR)

project(xt_neighbor VERSION 2.0.0 LANGUAGES CXX CUDA)

option(BUILD_CUB "Build with tests related to cub library" OFF)
option(BUILD_NON_CUB "Build tests not related to cub library " OFF)
option(BUILD_BINARY "Build binary" ON)

if(BUILD_BINARY)
	add_executable(xt_neighbor src/cli.cu)
endif(BUILD_BINARY)

if(BUILD_CUB)
	add_executable(test_bin_packing test/test_bin_packing.cu)
	add_executable(test_stream1 test/test_stream1.cu)
	add_executable(test_stream2 test/test_stream2.cu)
	add_executable(test_stream3 test/test_stream3.cu)
	add_executable(test_stream4 test/test_stream4.cu)
endif(BUILD_CUB)

if(BUILD_NON_CUB)
	add_executable(test_codec test/test_codec.cu)
	add_executable(test_generate_combination test/test_generate_combination.cu)
	add_executable(test_stream test/test_stream.cu)
	add_executable(test_memory test/test_memory.cu)
endif(BUILD_NON_CUB)
