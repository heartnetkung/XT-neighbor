cmake_minimum_required(VERSION 3.23.1 FATAL_ERROR)

project(xt_neighbor VERSION 1.0.0 LANGUAGES CXX CUDA)

option(BUILD_TEST "Build tests" OFF)
option(BUILD_BINARY "Build binary" ON)

if(BUILD_BINARY)
	add_executable(xt_neighbor src/cli.cu)
endif(BUILD_BINARY)

if(BUILD_TEST)
	add_executable(test_codec test/test_codec.cu)
	add_executable(test_brute_force test/test_brute_force.cu)
	add_executable(test_generate_combination test/test_generate_combination.cu)
endif(BUILD_TEST)
