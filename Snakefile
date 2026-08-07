# Benchmark and figure pipeline for the SymScan paper.
#
#   sudo benchmarks/bench_mode.sh on
#   tmux new -s bench
#   conda activate symdel
#   snakemake -j1 --keep-going --rerun-incomplete 2>&1 | tee bench.log
#   # Ctrl-b d to detach; ssh back in later and: tmux attach -t bench
#   sudo benchmarks/bench_mode.sh off
#
# Always -j1: the timing rules must never run concurrently with anything else.
# Thread counts come from bench_pinned.sh, not from Snakemake, so a single job
# still uses every pinned core.
#
# Each notebook is executed to a copy under _runs/ rather than in place. That
# copy is the rule's proof of completion, keeps the tracked notebook clean, and
# records the describe_env() output belonging to this particular run.

import os

# Logical CPUs to pin to: one per physical core, one core class. Override with
# --config cpus="0-7".
CPUS = config.get("cpus", "0,2,4,6,8,10,12,14")

# Notebooks that measure something. Executed under bench_pinned.sh.
BENCH = "benchmarks"
# Notebooks that only draw. Cheap, unpinned.
PUB = "pub"
# Executed notebook copies; a log, not an artifact (gitignored).
RUNS = "_runs"


# Refuse to take a measurement unless bench_mode.sh has fixed the clock. Checked
# per benchmark rather than once at startup, so that it also catches the mode
# being turned off part-way through a multi-day campaign -- and so that redrawing
# a figure does not require being in benchmark mode at all.
REQUIRE_BENCH_MODE = (
    'if [ "$(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null)" != 1 ] '
    '|| [ "$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)" '
    '!= performance ]; then '
    'echo "CPU clock not fixed -- run: sudo benchmarks/bench_mode.sh on" >&2; exit 1; fi'
)


def bench_nb(name, cpus=CPUS):
    """Run a benchmark notebook on the pinned cores, from its own directory so
    the notebook's relative paths (tmp/, ../data/) keep working."""
    return (
        REQUIRE_BENCH_MODE + " && "
        "cd {bench} && ./bench_pinned.sh '{cpus}' "
        "jupyter nbconvert --to notebook --execute "
        "--ExecutePreprocessor.timeout=-1 "
        "--output-dir=../{runs} --output {name}.executed {name}.ipynb"
    ).format(bench=BENCH, cpus=cpus, runs=RUNS, name=name)


def fig_nb(name):
    return (
        "cd {pub} && jupyter nbconvert --to notebook --execute "
        "--ExecutePreprocessor.timeout=-1 "
        "--output-dir=../{runs} --output {name}.executed {name}.ipynb"
    ).format(pub=PUB, runs=RUNS, name=name)


rule all:
    input:
        # publication figures
        "pub/figs/scaling_benchmark.pdf",
        "pub/figs/airr_overlap.pdf",
        "pub/figs/symscan_ncpu_scaling.pdf",
        "pub/figs/symscan_memory_scaling.pdf",
        "pub/figs/applications.pdf",
        # correctness checks produce no data, only a clean execution
        f"{RUNS}/01_correctness.executed.ipynb",
        f"{RUNS}/03B_airr_overlap_correctness.executed.ipynb",


# --------------------------------------------------------------------------
# external tools
# --------------------------------------------------------------------------

rule xtneighbor:
    output:
        "xtneighbor/build/xt_neighbor",
    shell:
        "cmake -S xtneighbor -B xtneighbor/build && make -C xtneighbor/build"


rule xtneighbor_streaming:
    output:
        "xtneighbor_streaming/build/xt_neighbor",
    shell:
        "cmake -S xtneighbor_streaming -B xtneighbor_streaming/build "
        "&& make -C xtneighbor_streaming/build"


rule compairr:
    output:
        "benchmarks/compairr/src/compairr",
    shell:
        "[ -d benchmarks/compairr ] || "
        "git clone https://github.com/uio-bmi/compairr.git benchmarks/compairr; "
        "make -C benchmarks/compairr"


# --------------------------------------------------------------------------
# benchmarks
# --------------------------------------------------------------------------

rule correctness:
    """01: all algorithms must return identical neighbour sets. No data output."""
    input:
        f"{BENCH}/01_correctness.ipynb",
        rules.xtneighbor.output,
        rules.xtneighbor_streaming.output,
        "data/emerson1.zip",
    output:
        f"{RUNS}/01_correctness.executed.ipynb",
    shell:
        bench_nb("01_correctness")


# These are all run on a single CPU

rule bench_algorithms:
    """02: algorithm comparison, runtime vs input size and vs edit distance, on
    medium-sized data (up to 100k sequences). CPU only."""
    input:
        f"{BENCH}/02_algorithms.ipynb",
        expand("data/emerson{i}.zip", i=range(1, 7)),
    output:
        "data/cpu_benchmark.csv",
        "data/cpu_dist_benchmark.csv",
        f"{RUNS}/02_algorithms.executed.ipynb",
    shell:
        bench_nb("02_algorithms", cpus="4")


rule bench_symdel_large:
    """02D: symdel implementation comparison, runtime vs input size and vs edit
    distance, at large scale (up to 30M sequences). Needs a GPU."""
    input:
        f"{BENCH}/02D_symdel_large_scale.ipynb",
        rules.xtneighbor.output,
        rules.xtneighbor_streaming.output,
        expand("data/emerson{i}.zip", i=range(1, 7)),
    output:
        "data/gpu_benchmark.csv",
        "data/gpu_dist_benchmark.csv",
        f"{RUNS}/02D_symdel_large_scale.executed.ipynb",
    shell:
        bench_nb("02D_symdel_large_scale")


rule bench_ncpu:
    """02B: SymScan runtime sweeping 1..N threads over the pinned cores."""
    input:
        f"{BENCH}/02B_symscan_ncpu_scaling.ipynb",
        "data/emerson1.zip",
    output:
        "data/symscan_ncpu_benchmark.csv",
        f"{RUNS}/02B_symscan_ncpu_scaling.executed.ipynb",
    shell:
        bench_nb("02B_symscan_ncpu_scaling")


rule bench_memory:
    """02C: peak RSS of SymScan vs SymDel, measured per subprocess."""
    input:
        f"{BENCH}/02C_symscan_memory_scaling.ipynb",
        expand("data/emerson_rep{i}.zip", i=range(1, 15)),
    output:
        "data/symscan_memory_benchmark.csv",
        f"{RUNS}/02C_symscan_memory_scaling.executed.ipynb",
    shell:
        bench_nb("02C_symscan_memory_scaling")


rule bench_airr_overlap:
    """03: repertoire overlap against CompAIRR and XT-neighbor. Needs a GPU."""
    input:
        f"{BENCH}/03_airr_overlap.ipynb",
        rules.xtneighbor_streaming.output,
        rules.compairr.output,
        "data/info.csv",
        expand("data/emerson_rep{i}.zip", i=range(1, 6)),
    output:
        "data/airr_overlap.csv",
        f"{RUNS}/03_airr_overlap.executed.ipynb",
    shell:
        bench_nb("03_airr_overlap")


rule airr_overlap_correctness:
    """03B: overlap counts must agree across implementations. No data output."""
    input:
        f"{BENCH}/03B_airr_overlap_correctness.ipynb",
        rules.xtneighbor_streaming.output,
        rules.compairr.output,
        "data/info.csv",
        expand("data/emerson_rep{i}.zip", i=range(1, 6)),
    output:
        f"{RUNS}/03B_airr_overlap_correctness.executed.ipynb",
    shell:
        bench_nb("03B_airr_overlap_correctness")


rule bench_tcrdist:
    """04: SymScan as a pre-filter for TCRdist neighbour search."""
    input:
        f"{BENCH}/04_tcrdist.ipynb",
        "data/emerson_HIP00110.tsv.gz",
    output:
        "data/tcrdist_benchmark.csv",
        f"{RUNS}/04_tcrdist.executed.ipynb",
    shell:
        bench_nb("04_tcrdist")


rule bench_antibody:
    """05: SymScan on a CDR3 k-mer as a candidate generator for lineages."""
    input:
        f"{BENCH}/05_antibody_lineages.ipynb",
        "data/briney_316188.tsv.gz",
    output:
        "data/antibody_benchmark_results.json",
        f"{RUNS}/05_antibody_lineages.executed.ipynb",
    shell:
        bench_nb("05_antibody_lineages")


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------

rule fig_scaling:
    input:
        f"{PUB}/scaling_plots.ipynb",
        "data/cpu_benchmark.csv",
        "data/cpu_dist_benchmark.csv",
        "data/gpu_benchmark.csv",
        "data/gpu_dist_benchmark.csv",
    output:
        "pub/figs/scaling_benchmark.pdf",
        f"{RUNS}/scaling_plots.executed.ipynb",
    shell:
        fig_nb("scaling_plots")


rule fig_airr_overlap:
    input:
        f"{PUB}/airr_overlap.ipynb",
        "data/airr_overlap.csv",
    output:
        "pub/figs/airr_overlap.pdf",
        f"{RUNS}/airr_overlap.executed.ipynb",
    shell:
        fig_nb("airr_overlap")


rule fig_ncpu:
    input:
        f"{PUB}/symscan_ncpu_scaling.ipynb",
        "data/symscan_ncpu_benchmark.csv",
    output:
        "pub/figs/symscan_ncpu_scaling.pdf",
        "pub/figs/symscan_ncpu_scaling.png",
        "pub/figs/symscan_ncpu_scaling.svg",
        f"{RUNS}/symscan_ncpu_scaling.executed.ipynb",
    shell:
        fig_nb("symscan_ncpu_scaling")


rule fig_memory:
    input:
        f"{PUB}/symscan_memory_scaling.ipynb",
        "data/symscan_memory_benchmark.csv",
    output:
        "pub/figs/symscan_memory_scaling.pdf",
        "pub/figs/symscan_memory_scaling.png",
        "pub/figs/symscan_memory_scaling.svg",
        f"{RUNS}/symscan_memory_scaling.executed.ipynb",
    shell:
        fig_nb("symscan_memory_scaling")


rule fig_applications:
    """Draws all four panels of the applications figure: TCRdist in the top row,
    antibody lineages in the bottom, so it reads both datasets."""
    input:
        f"{PUB}/applications.ipynb",
        "data/tcrdist_benchmark.csv",
        "data/antibody_benchmark_results.json",
    output:
        "pub/figs/applications.pdf",
        "pub/figs/applications.png",
        "pub/figs/applications.svg",
        f"{RUNS}/applications.executed.ipynb",
    shell:
        fig_nb("applications")


# --------------------------------------------------------------------------
# session checks
# --------------------------------------------------------------------------

onstart:
    def _read(path):
        try:
            with open(path) as f:
                return f.read().strip()
        except OSError:
            return None

    # Reported here for the log; enforced per benchmark by REQUIRE_BENCH_MODE, so
    # that figure-only rebuilds do not need the machine in benchmark mode.
    turbo = _read("/sys/devices/system/cpu/intel_pstate/no_turbo")
    gov = _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    print(f"cpu clock: no_turbo={turbo} governor={gov}")

    # The notebooks call sys.executable for their worker subprocesses, so the
    # environment this is launched from is the one being benchmarked.
    print(f"benchmarking environment: {os.environ.get('CONDA_PREFIX', 'no conda env active')}")
    print(f"pinned cpus: {CPUS}")
