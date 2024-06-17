## Description

XTNeighbor is a fast scalable method for nearest neighbor search of adaptive immune receptors (AIRs) using GPU. In simple terms, our inputs are CDR3 regions of AIRs represented as a string of amino acids and the algorithm finds all pairs of AIRs such that their similarity is within a specified Levenshtein distance threshold. XTNeighbor is orders of magnitude faster than current methods thanks to a symmetric deletion algorithmic approach, GPU acceleration, and memory optimization. A detailed description of the method is provide in our [arXiv preprint](https://doi.org/10.48550/arXiv.2403.09010).

## Quick Usage

This is [the Google Colab Notebook](https://colab.research.google.com/drive/1JbRLtRrmUv9zZollSfT9xp6WqOy7LB7q) that allows user to quickly use this tool from web browser without setting up the GPU environment.

## Installation

XTNeighbor has been tested with the following environment:
- CUDA SDK version 11.0+
- Nvidia T4 GPU, Nvidia V100 GPU (see Google Colab demo below)
- Linux OS or Google Colab runtime

Detailed installation instructions, examples, and testing code are provided via a [Google Colab demo](https://colab.research.google.com/drive/1UrTLHNcW0XAp_6jL2ys1FVNutaoJOX9K).

For advanced tutorial in compiling XT-neighbor on bare-bone Linux, read this [tutorial.](https://github.com/heartnetkung/XT-neighbor/wiki/Bare%E2%80%90Bone-Installation-on-Linux)

## Usage

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

## Benchmarking and Reproducibility
- Benchmarking code on nearest neighbor search is provided via [Google Colab Notebook](https://colab.research.google.com/drive/1j-DO11k2NQPlNJF966BNjRKhHPY94sJJ).
- Benchmarking code on immune repertoire comparison is provided via [Google Colab Notebook](https://colab.research.google.com/drive/19Qh1cgw-Zgs2aWQRIV-WkbzdGJBi0zGg).
- Source code for producing figures in the preprint is provided in the `/pub` folder.

## Deduplication Warning
- A major factor in runtime of the program is duplication in the input. Please drop all duplicates before using it as input. If duplication matters, you should decuplicate, give it to XT-neighbor, then recombine it with your original input.
- Without deduplication, the number of output triplets usually grow at least quadratically to the input size, thus the runtime also grows quadratically.
- The reason is that, assume your data contains a cluster of size `N`, any additional member to these clusters add `N` more redundant results, thus introduce quadratic scaling. In real datasets, TCR are distributed in clusters.

## Documentation
- [link to auto generated documentation](https://heartnetkung.github.io/XT-neighbor/files.html)

## Note on 1.0 and 2.0 Version of the Algorithm
- The code in this repo contains both 1.0 and 2.0 versions which is named XTNeighbor and XTNeighbor-streaming in the paper.
- 1.0 has the limitation on CDR3-length not exceeding 18 which might not be practical for most users. All users are adviced to use the 2.0 version which does not have this limitation.

## FAQ
- Is multiple GPU supported?
  - No, but contribution is welcomed.
- Is there a CPU version?
  - Yes. The CPU version (also implemented by the same author) is included in a Python library toolkit for immune repertoire analysis called [Pyrepseq](https://github.com/andim/pyrepseq) (the function is pyrepseq.nn.symdel). In fact for average-load task, it is more convenient to use that package since it's pip-installable, whereas this package requires GPU, CUDA driver/SDK installation. The sample code can be seen in this [Google Colab.](https://colab.research.google.com/drive/1Tsv5Yiinj6PPJdp58-fch_gCm_AbQ5vs#scrollTo=roHKguZq4F6N)
- How can I be confident about the correctness of this approach coming from biology background?
  - We provide this [Colab Notebook](https://colab.research.google.com/drive/1Tsv5Yiinj6PPJdp58-fch_gCm_AbQ5vs#scrollTo=FKQHIC8J4L1c) to present the correctness of our approach by showing that our approach produce the same correct result as the one produce by simple for-loop approach. In addition, if you find a bug or mistake, you can use that code as a template for bug report as well.

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
