## Cmake
```bash
# test
mkdir build;cd build;cmake ..;make

#build with cub
mkdir build;cd build;cmake .. -DBUILD_CUB=ON;make

#test cub
mkdir build;cd build;cmake .. -DBUILD_CUB=ON -DBUILD_BINARY=OFF;make

# run given test
./test_codec

# run cli
./xt_neighbor -p ../test/sample/input1.txt -n 4 -V -c -o output.txt
```