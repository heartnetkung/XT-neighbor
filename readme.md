# Lightning-fast adaptive immune receptor similarity search by symmetric deletion lookup

This repository allows reproduction of the results reported in our [arXiv preprint](https://doi.org/10.48550/arXiv.2403.09010).

## Benchmarking and Reproducibility
- Source code for producing figures in the preprint is provided in the `/pub` folder.
- Source code for the benchmarking scripts is provided in the `/benchmarks` folder.
- A `snakemake` workflow can be used to download benchmarking data and to control the overall code execution.

## Quick Usage

Please see [SymScan](https://github.com/yutanagano/symscan) for quickstart examples.

## XTNeighbor

This repository also provides XTNeighbor, our GPU-accelerated neighbor search tool.

In practice, we recommend [SymScan](https://github.com/yutanagano/symscan) as the default: it runs on CPUs, can be used from Python, and was faster in our benchmarks on datasets of up to about a million sequences. XTNeighbor is preferable for the largest workloads when a GPU is available and where its streaming design keeps memory use constant.

XTNeighbor has been tested with the following environment:
- CUDA SDK version 11.0+
- Nvidia RTX4090, T4 GPU, V100 GPU
- Linux OS or Google Colab runtime

Additional installation instructions, examples, and testing code are provided via a [Google Colab demo](https://colab.research.google.com/drive/1UrTLHNcW0XAp_6jL2ys1FVNutaoJOX9K).

For advanced tutorial in compiling XT-neighbor on bare-bone Linux, read this [tutorial.](https://github.com/heartnetkung/XT-neighbor/wiki/Bare%E2%80%90Bone-Installation-on-Linux)

### Usage

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
	 -a or --airr: use AIRR format for input-path instead. Relevant fields are cdr3_aa and duplicate_count
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

### Supported inputs and limits
- **Distance threshold:** any positive integer (`-d`), for both Levenshtein and Hamming distance.
- **Sequence length:** input sequences of up to 500 characters are supported.
- **Prefix truncation:** to save memory, XTNeighbor internally left-aligns and truncates sequences to an 18-character prefix when generating candidate pairs. This never causes true neighbor pairs to be missed (see the appendix of the accompanying paper for a proof), and every candidate is verified against the full, untruncated sequences before being returned.

### Documentation
- [link to auto generated documentation](https://heartnetkung.github.io/XT-neighbor/files.html)

### Note on versions of XTNeighbor
- The code in this repo contains XTNeighbor-streaming (the default) and a non-streaming variant simply called XTNeighbor.
- All users are adviced to use the XTNeighbor-streaming implementation only. The non-streaming variant is only provided for pedagogical purposes. It only works on sequences of up to 18 characters and does not provide support for AIRR compliant inputs.

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
