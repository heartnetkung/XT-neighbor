## Description
XT-neighbor is a fast scalable computational method for nearest neighbor search of T-cell receptors (TCR) using GPU. In simple terms, our inputs are CDR3 regions of TCR represented as a string of amino acids and the algorithm find all pairs of TCR such that their similarity is within the specified Levenshtein distance threshold. The key feature of our algorithm is orders of magnitude faster and more scalable than the current fastest method thanks to algorithmic advancemence, GPU adaptation, and memory optimization techniques. To read more about XT-neighbor, the research paper is available on [arxiv website](https://arxiv.org/abs/2403.09010).

## Requirement
- CUDA SDK version 11.0+
- Linux OS or Google Colab runtime

## Usage
- [link to Google Colab demo with examples and all the benchmarking code](https://colab.research.google.com/drive/13zHkThcsIpe_dYMLb6IlbcTn2wAzfox7)

```txt
xt_neighbor: perform either nearest neighbor search for CDR3 sequences or immune repertoire overlap using GPU-based xt_neighbor algorithm.
	====================
	 Common Options
	====================
	 -d or --distance [number]: distance threshold defining the neighbor (default to 1)
	 -o or --output-path [str]: path of the output file (default to no output)
	 -m or --measurement [leven|hamming]: distance measurement (default to leven)
	 -v or --version: print the version of the program then exit
	 -h or --help: print the help text of the program then exit
	 -V or --verbose: print extra detail as the program runs for debugging purpose
	====================
	 Nearest Neighbor Options
	====================
	 -i or --input-path [str] (required): path of csv input file containing exactly 1 column: CDR3 amino acid sequences
	 -n or --input-length [number] (required): number of rows given in the input file
	====================
	 Repertoire Overlap Options
	====================
	 -i or --input-path [str] (required): path of csv input file containing exactly 2 columns: CDR3 amino acid sequences and their frequency. Note that the sequences are assumed to be unique
	 -n or --input-length [number] (required): number of sequences given in the input file
	 -I or --info-path [str] (required): path of csv input file containing exactly 1 column: repertoire sizes. Note that the order of input sequence must be sorted according to this repertoire info
	 -N or --info-length [number] (required): number of repertoires given in the info file
```

## Deduplication Warning
- A major factor in runtime of the program is duplication in the input. Please drop all duplicates before using it as input. If duplication matters, you should decuplicate, give it to XT-neighbor, then recombine it with your original input.
- Without deduplication, the number of output triplets usually grow at least quadratically to the input size, thus the runtime also grows quadratically.
- The reason is that, assume your data contains a cluster of size `N`, any additional member to these clusters add `N` more redundant results, thus introduce quadratic scaling. In real datasets, TCR are distributed in clusters.

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
