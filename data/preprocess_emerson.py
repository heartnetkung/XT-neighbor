"""Derive the benchmark inputs from the raw Emerson et al. 2017 archive.

    python preprocess_emerson.py global       <archive.zip> <out_dir>
    python preprocess_emerson.py repertoires  <archive.zip> <out_dir>
    python preprocess_emerson.py hip00110     <archive.zip> <out_dir>

global       emerson{1..6}_dl.zip      every subject summed into one
                                       cdr3 -> templates table, in 10M-row files
repertoires  emerson_rep{1..14}_dl.zip one file per batch of 50 subjects, each
                                       subject's CDR3s listed separately, plus
                                       info_dl.csv with the rows per subject.
                                       Subjects without template counts are
                                       skipped (see has_templates)
hip00110     emerson_HIP00110_dl.tsv.gz  one subject's raw TSV, unmodified

Subjects are taken in sorted filename order, so the output is reproducible.
Members are streamed out of the archive one at a time; it is never unpacked.
"""
import gzip
import shutil
import sys
import time
import zipfile
from collections import Counter
from itertools import islice
from pathlib import Path

import pandas as pd

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
MIN_CDR3_LEN = 5
MAX_CDR3_LEN = 30

# A valid CDR3 starts with C, ends with F, W or C, and is made of standard
# amino acids only (the pyrepseq.isvalidcdr3 definition). The first and last
# residues are matched separately, so the middle is 2 residues shorter.
CDR3_RE = rf'C[{AMINO_ACIDS}]{{{MIN_CDR3_LEN - 2},{MAX_CDR3_LEN - 2}}}[FWC]'


def list_members(archive):
    with zipfile.ZipFile(archive) as zf:
        return sorted(m for m in zf.namelist()
                      if m.endswith(('.tsv', '.tsv.gz')) and not m.startswith('__MACOSX'))


def has_templates(zf, member):
    """False for subjects whose `templates` column is null (older Adaptive
    exports without template counts). The column is either null for every row
    or for none, so the first row decides.
    Same filter as in https://github.com/andim/paper-tcellimprint/ in file
    preprocess_emerson_filter.py."""
    with zf.open(member) as handle:
        df = pd.read_csv(handle, sep='\t', usecols=['templates'], nrows=1,
                         compression='gzip' if member.endswith('.gz') else None)
    return not df['templates'].isnull().any()


def iter_subjects(archive, members):
    """Yield (member, templates) for each subject, where templates is the
    summed template count per valid CDR3 (a Series indexed by CDR3)."""
    with zipfile.ZipFile(archive) as zf:
        for member in members:
            with zf.open(member) as handle:
                df = pd.read_csv(handle, sep='\t', usecols=['amino_acid', 'templates'],
                                 compression='gzip' if member.endswith('.gz') else None)
            df = df.dropna()
            df = df[df['amino_acid'].str.fullmatch(CDR3_RE)]
            yield member, df.groupby('amino_acid')['templates'].sum()


def write_cdr3_zip(path, rows):
    """Write (cdr3, count) rows as a CSV inside a zip whose only member is
    named after the zip, as the benchmark notebooks expect."""
    pd.DataFrame(rows, columns=['cdr3', 'count']).to_csv(
        path, index=False,
        compression={'method': 'zip', 'archive_name': path.stem + '.txt'})


def build_global(archive, out_dir, n_files=6, rows_per_file=10_000_000):
    """Sum every subject into one cdr3 -> templates table and split it into
    `n_files` files of `rows_per_file` rows (the last one may be shorter).

    Keeps every distinct CDR3 in memory at once, tens of millions of strings,
    and takes over an hour on the full archive.
    """
    members = list_members(archive)
    print(f'{len(members)} subjects in {archive.name}')

    start = time.time()
    totals = Counter()
    for i, (member, templates) in enumerate(iter_subjects(archive, members)):
        totals.update(templates.to_dict())
        if i % 50 == 0:
            print(f'  {i}/{len(members)}  {len(totals):,} distinct cdr3  '
                  f'{time.time() - start:,.0f}s')

    rows = iter(totals.items())
    for file_i in range(1, n_files + 1):
        chunk = list(islice(rows, rows_per_file))
        if not chunk:
            raise RuntimeError(f'{len(totals):,} distinct cdr3 fill only '
                               f'{file_i - 1} of {n_files} files of {rows_per_file:,} rows')
        write_cdr3_zip(out_dir / f'emerson{file_i}_dl.zip', chunk)
        print(f'wrote emerson{file_i}_dl.zip: {len(chunk):,} rows')
    print(f'done in {time.time() - start:,.0f}s')


def build_repertoires(archive, out_dir, n_batches=14, batch_size=50):
    """Write each batch of `batch_size` subjects to one file, one subject's
    CDR3s after another (a CDR3 in two subjects appears twice), and record
    each subject's row count in info_dl.csv in the same order."""
    members = list_members(archive)
    with zipfile.ZipFile(archive) as zf:
        kept = [m for m in members if has_templates(zf, m)]
    print(f'{len(members)} subjects, skipping {len(members) - len(kept)} '
          f'with missing template counts')
    if len(kept) < n_batches * batch_size:
        raise RuntimeError(f'{len(kept)} subjects with template counts, '
                           f'need {n_batches * batch_size}')
    members = kept[:n_batches * batch_size]

    start = time.time()
    subject_counts = []
    batch_rows = []
    for i, (member, templates) in enumerate(iter_subjects(archive, members), start=1):
        batch_rows.extend(templates.items())
        subject_counts.append((Path(member).name, len(templates)))

        if i % batch_size == 0:
            batch_i = i // batch_size
            write_cdr3_zip(out_dir / f'emerson_rep{batch_i}_dl.zip', batch_rows)
            print(f'batch {batch_i}/{n_batches}: {len(batch_rows):,} rows  '
                  f'{time.time() - start:,.0f}s')
            batch_rows = []

    pd.DataFrame(subject_counts, columns=['file', 'count']).to_csv(
        out_dir / 'info_dl.csv', index=False)
    print(f'done in {time.time() - start:,.0f}s')


def build_hip00110(archive, out_dir, subject='HIP00110'):
    """Copy one subject's raw TSV out of the archive, for the TCRdist benchmark."""
    matches = [m for m in list_members(archive) if Path(m).name.split('.')[0] == subject]
    if not matches:
        raise RuntimeError(f'no subject {subject} in {archive.name}')
    member = matches[0]

    out_path = out_dir / f'emerson_{subject}_dl.tsv.gz'
    with zipfile.ZipFile(archive) as zf, zf.open(member) as src:
        # a .tsv.gz member is already compressed; only a plain .tsv needs gzipping
        with (open(out_path, 'wb') if member.endswith('.gz') else gzip.open(out_path, 'wb')) as dst:
            shutil.copyfileobj(src, dst)
    print(f'wrote {out_path.name} from {member}')


MODES = {'global': build_global, 'repertoires': build_repertoires, 'hip00110': build_hip00110}

if __name__ == '__main__':
    if len(sys.argv) != 4 or sys.argv[1] not in MODES:
        sys.exit(__doc__)
    MODES[sys.argv[1]](Path(sys.argv[2]), Path(sys.argv[3]))
