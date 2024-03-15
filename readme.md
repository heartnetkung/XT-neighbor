## Description
XT-neighbor is a fast scalable computational method for nearest neighbor search of T-cell receptors (TCR) using GPU. In simple terms, our inputs are CDR3 regions of TCR represented as a string of amino acids and the algorithm find all pairs of TCR such that their similarity is within the specified Levenshtein distance threshold. The key feature of our algorithm is orders of magnitude faster and more scalable than the current fastest method thanks to algorithmic advancemence, GPU adaptation, and memory optimization techniques. To read more about XT-neighbor, the research paper is available on [arxiv website](https://arxiv.org/abs/2403.09010).

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
# follow instructions in 2.0/make_doc.sh
```

## Citation
```bibtex
@misc{chotisorayuth2024lightningfast,
      title={Lightning-fast adaptive immune receptor similarity search by symmetric deletion lookup}, 
      author={Touchchai Chotisorayuth and Andreas Tiffeau-Mayer},
      year={2024},
      eprint={2403.09010},
      archivePrefix={arXiv},
      primaryClass={q-bio.QM}
}
```
