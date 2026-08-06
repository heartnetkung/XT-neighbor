import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pyrepseq
import symscan

from scipy.sparse import csr_matrix

# repo root, assumed to be the parent directory of this file (benchmarks/)
repo_path = Path(__file__).resolve().parent.parent

def _idx_to_repoverlap(i_arr, j_arr, useqs=None, useq_ids=None, dup_counts=None, rep_sizes=None, chunk_size=5_000_000):
    dup_counts = np.asarray(dup_counts, dtype=np.int64)
    rep_sizes = np.asarray(rep_sizes)
    n_rep, n_useq = len(rep_sizes), len(useqs)

    rep_ids = np.repeat(np.arange(n_rep), rep_sizes)
    rep_profile = csr_matrix((dup_counts, (useq_ids, rep_ids)), shape=(n_useq, n_rep))

    # chunked to avoid scipy's int32 indptr overflow in rep_profile[i_arr] when
    # i_arr/j_arr have hundreds of millions of entries (large edit-distance neighbor sets)
    cross = np.zeros((n_rep, n_rep), dtype=np.int64)
    for start in range(0, len(i_arr), chunk_size):
        end = start + chunk_size
        chunk = rep_profile[i_arr[start:end]].T @ rep_profile[j_arr[start:end]]
        cross += np.asarray(chunk.todense())

    # the "sequence overlaps itself" diagonal term
    self_term = rep_profile.T @ rep_profile
    return cross + np.asarray(self_term.todense())

def symdel_overlap(distance, is_hamming, seqs=None, dup_counts=None, rep_sizes=None):
  measure = 'hamming' if is_hamming else None
  useq_ids, useqs = pd.factorize(np.asarray(seqs))
  if is_hamming:
    raw_output = pyrepseq.symdel(useqs, max_edits=distance,
                               custom_distance=measure, max_custom_distance=distance)
  else:
    raw_output = pyrepseq.symdel(useqs, max_edits=distance)

  if len(raw_output):
    arr = np.asarray(raw_output, dtype=np.int64)
    i_arr, j_arr = arr[:, 0], arr[:, 1]
  else:
    i_arr = j_arr = np.empty(0, dtype=np.int64)

  return _idx_to_repoverlap(i_arr, j_arr, useqs=useqs, useq_ids=useq_ids, dup_counts=dup_counts, rep_sizes=rep_sizes)


def symscan_overlap(distance, is_hamming, seqs=None, dup_counts=None, rep_sizes=None):
  useq_ids, useqs = pd.factorize(np.asarray(seqs))
  if is_hamming:
    raw_output = symscan.get_neighbors_within(useqs, max_distance=distance, distance_type='hamming')
  else:
    raw_output = symscan.get_neighbors_within(useqs, max_distance=distance)
  i_arr, j_arr = raw_output[0], raw_output[1]
  # symscan only returns each pair once (row < col); mirror both directions since
  # _idx_to_repoverlap's cross term needs both (i,j) and (j,i) to be symmetric
  i_arr, j_arr = np.concatenate([i_arr, j_arr]), np.concatenate([j_arr, i_arr])

  return _idx_to_repoverlap(i_arr, j_arr, useqs=useqs, useq_ids=useq_ids, dup_counts=dup_counts, rep_sizes=rep_sizes)


def read_pair_file_to_matrix(filename, n_rep, sep, comment=None):
  df = pd.read_csv(filename, sep=sep, header=None, comment=comment, thousands=',')
  mat = np.zeros((n_rep, n_rep), dtype=np.int64)
  x, y, v = df[0].to_numpy(), df[1].to_numpy(), df[2].to_numpy()
  mat[x, y] = v
  mat[y, x] = v # both implementations only write out one direction of each symmetric pair
  return mat

def xt_neighbor_overlap(distance, is_hamming, seqs=None, dup_counts=None, rep_sizes=None, return_matrix=False):
  n, N = len(seqs), len(rep_sizes)
  xt_neighbor_bin = repo_path / 'xtneighbor_streaming' / 'build' / 'xt_neighbor'
  cmd = [str(xt_neighbor_bin), '-i', 'tmp/xt_input1.txt', '-n', str(n),
         '-I', 'tmp/xt_input2.txt', '-N', str(N), '-d', str(distance)]
  if is_hamming:
    cmd += ['-m', 'hamming']
  cmd += ['-o', 'tmp/xt_output.txt']
  subprocess.run(cmd, check=True)
  if return_matrix:
    return read_pair_file_to_matrix('tmp/xt_output.txt', N, sep=' ')
  else:
    return None

def compairr_overlap(distance, is_hamming, seqs=None, dup_counts=None, rep_sizes=None, return_matrix=False, n_cpu=1):
  N = len(rep_sizes)
  if not is_hamming and distance > 1:
    # CompAIRR only supports indels (Levenshtein distance) when d=1
    return None
  cmd = ['./compairr/src/compairr', '-g', '-a', '-m', 'tmp/compairr_input1.txt',
         'tmp/compairr_input1.txt', '-d', str(distance), '--cdr3', '-t', str(n_cpu), '-o', 'tmp/output.tsv']
  if not is_hamming:
    cmd += ['-i']
  subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
  if return_matrix:
    return read_pair_file_to_matrix('tmp/output.tsv', N, sep='\t', comment='#')
  else:
    return None

def sample_repertoire(data,info,n,random_state=0,max_seqs=None):
  corrupted_files = ['HIP14092.tsv.gz','HIP04958.tsv.gz']
  info_subset = info[~info['file'].isin(corrupted_files)][:220-len(corrupted_files)].sample(n, random_state=random_state)
  reps = []
  rep_col = []
  for i in range(len(info_subset)):
    row = info_subset.iloc[i,:]
    chunk = data[row['start']:row['end']]
    if max_seqs is not None and len(chunk) > max_seqs:
      chunk = chunk.sample(max_seqs, random_state=random_state)
    reps.append(chunk)
    rep_col += [i]*len(chunk)
  ans = pd.concat(reps, ignore_index=True)
  ans.rename(columns={'count':'duplicate_count','cdr3':'cdr3_aa'},inplace=True)
  ans['repertoire_id'] = rep_col
  info_subset = info_subset.copy()
  info_subset['count'] = pd.Series(rep_col).value_counts().sort_index().to_numpy()
  return ans, info_subset

def prepare(reps, info_subset):
  reps.to_csv('tmp/compairr_input1.txt',index=False,sep='\t')
  reps.to_csv('tmp/xt_input1.txt',index=False,columns=['cdr3_aa','duplicate_count'])
  info_subset['count'].to_csv('tmp/xt_input2.txt',index=False)
  return reps['cdr3_aa'].tolist(), reps['duplicate_count'].tolist(), info_subset['count'].tolist()