Run the notebooks as follows:

sudo ./bench_mode.sh on

nohup ./bench_pinned.sh "0,2,4,6" \
  jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 \
    02_algorithms.ipynb \
  > bench_02.log 2>&1 &

sudo ./bench_mode.sh off