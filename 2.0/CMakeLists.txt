cmake_minimum_required(VERSION 3.23.1 FATAL_ERROR)

project(xt_neighbor VERSION 1.0.0 LANGUAGES CXX CUDA)

option(BUILD_CUB "Build with cub library" OFF)
option(BUILD_BINARY "Build binary" ON)

if(NOT BUILD_CUB)
	add_executable(test_codec test/test_codec.cu)
	add_executable(test_generate_combination test/test_generate_combination.cu)
	add_executable(test_stream test/test_stream.cu)
elseif(BUILD_BINARY)
	add_executable(xt_neighbor src/cli.cu)
endif(NOT BUILD_CUB)
