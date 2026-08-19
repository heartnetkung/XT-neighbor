import os
import platform
import signal
import subprocess
import sys
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version

timeout_seconds = 100

class BenchmarkTimeout(Exception):
    pass


@contextmanager
def time_limit(seconds=None):
    seconds = timeout_seconds if seconds is None else seconds

    def handler(signum, frame):
        raise BenchmarkTimeout(f'exceeded {seconds}s')

    old_handler = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def run_binary(cmd, timeout=None, echo=True):
    """Run an external benchmark binary in its own process group, so that a
    timeout kills the whole process tree instead of leaving it running on the
    GPU and distorting the following measurements. Raises BenchmarkTimeout on
    timeout (whether it comes from this call's own timer or from an outer
    time_limit() SIGALRM), which the benchmark loop skips over.

    Returns the captured output, which is also printed unless echo=False (set
    that for workers whose stdout is a measurement to be parsed, not read)."""
    timeout = timeout_seconds if timeout is None else timeout
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, start_new_session=True)
    try:
        out, _ = proc.communicate(timeout=timeout)
    except (subprocess.TimeoutExpired, BenchmarkTimeout, KeyboardInterrupt) as e:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.communicate()
        if isinstance(e, subprocess.TimeoutExpired):
            raise BenchmarkTimeout(f'{cmd[0]} exceeded {timeout}s') from None
        raise
    if out and echo:
        print(out, end='')
    if proc.returncode != 0:
        print(f'Warning: {cmd[0]} exited with code {proc.returncode}')
    return out


# packages whose version can move a timing: the search implementations
# themselves, and the numerical/JIT layers their inner loops run on.
tracked_packages = ['symscan', 'pyrepseq', 'pybktree', 'rapidfuzz', 'pwseqdist',
                    'numba', 'numpy', 'scipy', 'pandas']


def _read_text(path):
    """Read a /proc or /sys entry, returning None where it doesn't exist
    (Colab and non-Intel hosts lack most of the cpufreq ones)."""
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return None


def _read_first_line(path):
    text = _read_text(path)
    return text.splitlines()[0].strip() if text else None


def affinity_list():
    """CPUs this process may run on, or None where the platform exposes no
    affinity mask (anything other than Linux)."""
    try:
        return sorted(os.sched_getaffinity(0))
    except AttributeError:
        return None


def available_cpus():
    """Number of CPUs this process may actually run on. Use instead of
    os.cpu_count(), which reports the whole machine and so ignores a taskset
    pin or a cgroup quota -- sizing a thread pool with it under a pin
    oversubscribes the cores and measures contention rather than scaling.

    Counts logical CPUs: pin to one CPU per physical core (SMT siblings share
    execution ports and L1/L2), or this over-counts by the SMT factor."""
    cpus = affinity_list()
    if cpus is not None:
        return len(cpus)
    count = getattr(os, 'process_cpu_count', os.cpu_count)()
    return count or 1


def describe_env(extra=None):
    """Snapshot of everything a timing measurement depends on, to be saved
    next to the results so numbers can be attributed to a machine and a set of
    library versions after the fact.

    The fields worth checking before trusting a run: 'colab' distinguishes a
    shared VM from a pinned host, 'affinity' confirms a taskset/numactl pin
    reached this process, and 'thread_env' confirms the libraries were told to
    honour it (an unset value there means the arm may have gone multithreaded).
    """
    packages = {}
    for name in tracked_packages:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None

    try:
        git_sha = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                                 capture_output=True, text=True,
                                 cwd=os.path.dirname(os.path.abspath(__file__))
                                 ).stdout.strip() or None
    except OSError:
        git_sha = None

    affinity = affinity_list()

    cpu_model = next((line.split(':', 1)[1].strip()
                      for line in (_read_text('/proc/cpuinfo') or '').splitlines()
                      if line.startswith('model name')), None)

    return {
        'colab': 'google.colab' in sys.modules,
        'platform': platform.platform(),
        'python': sys.version.split()[0],
        'git_sha': git_sha,
        'cpu_model': cpu_model,
        'n_cpus_total': os.cpu_count(),
        'affinity': affinity,
        'n_cpus_visible': len(affinity) if affinity is not None else None,
        'governor': _read_first_line(
            '/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor'),
        'no_turbo': _read_first_line(
            '/sys/devices/system/cpu/intel_pstate/no_turbo'),
        'gpu': _describe_gpu(),
        'thread_env': {key: os.environ.get(key) for key in
                       ['RAYON_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                        'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
                        'NUMBA_NUM_THREADS', 'OMP_PROC_BIND', 'OMP_PLACES']},
        'packages': packages,
        'timeout_seconds': timeout_seconds,
        **(extra or {}),
    }


def _describe_gpu():
    """Name and clock caps of the visible GPU, or None on CPU-only hosts.
    Clocks matter because a T4 downclocks under sustained load unless locked
    with `nvidia-smi -lgc`, which shows up as drift across a long sweep."""
    try:
        out = subprocess.run(
            ['nvidia-smi',
             '--query-gpu=name,memory.total,clocks.max.sm,clocks.applications.gr',
             '--format=csv,noheader'],
            capture_output=True, text=True, timeout=15).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return None
    if not out:
        return None
    fields = [f.strip() for f in out.splitlines()[0].split(',')]
    keys = ['name', 'memory_total', 'clocks_max_sm', 'clocks_applications_gr']
    return dict(zip(keys, fields))


def filter_combos(df):
    """Filter a DataFrame of benchmark results to only the algorithm/input_size/distance
    combinations that have the maximum number of runs"""
    nmax = (df.groupby(["algorithm", "input_size", "distance"])
     .size()
     .max()
    )
    good_combos = (
    df
    .groupby(["algorithm", "input_size", "distance"])
    .size()
    .reset_index(name="n")
    .query(f"n == {nmax}")
    )

    good_rows = df.merge(
        good_combos[["algorithm", "input_size", "distance"]],
        on=["algorithm", "input_size", "distance"],
        how="inner"
    ).sort_values(["algorithm", "input_size", "distance"])

    return good_rows