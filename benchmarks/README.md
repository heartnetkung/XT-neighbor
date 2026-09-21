Run the notebooks as follows (or use the snakemake workflow):

conda activate symdel

sudo ./bench_mode.sh on

nohup ./bench_pinned.sh "4" \
  jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 \
    02_algorithms.ipynb \
  > bench.log 2>&1 &

# Or in the multi-core case
nohup ./bench_pinned.sh "0,2,4,6,8,10,12,14" \
  jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 \
    02D_symdel_large_scale.ipynb \
  > bench.log 2>&1 &

sudo ./bench_mode.sh off

The symscan-airr binary can be installed using:

cargo install symscan-airr --version 0.1.0