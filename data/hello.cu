#include <iostream>
#include <stdlib.h>
#include <stdio.h>

__global__ void cuda_hello(){
    printf("Hello World from GPU! %d\n", threadIdx.x*gridDim.x);
}

int main() {
    printf("Hello World from CPU!\n");
    cuda_hello<<<50,1024>>>();
    cudaDeviceSynchronize();
    return 0;
}
