## Description

## Requirement
- CUDA SDK version 11.0+
- Linux OS or Google Colab runtime

## Usage
- [link to Google Colab demo with examples and all the benchmarking code](https://colab.research.google.com/drive/13zHkThcsIpe_dYMLb6IlbcTn2wAzfox7)

## Documentation
- [link to auto generated documentation](https://heartnetkung.github.io/XT-neighbor/files.html)

## Dev Commands
```sh
# compile just the binary
cd 2.0; mkdir build; cd build; cmake ..; make

# compile the binary and all the test cases
cd 2.0; mkdir build; cd build; cmake .. -DBUILD_CUB=ON -DBUILD_NON_CUB=ON -DBUILD_BINARY=ON; make

# run test
cd 2.0/build; ./test_codec

# update doc
# follow instruction in 2.0/make_doc.sh
```

## Citation
```bibtex
@misc{vaswani2023attention,
      title={Attention Is All You Need}, 
      author={Ashish Vaswani and Noam Shazeer and Niki Parmar and Jakob Uszkoreit and Llion Jones and Aidan N. Gomez and Lukasz Kaiser and Illia Polosukhin},
      year={2023},
      eprint={1706.03762},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```