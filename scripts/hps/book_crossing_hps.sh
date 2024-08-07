# hyper-parameter search for bookcrossing
gpu=0
for bs in 128 256 512 1024
do
  for lr in 0.1 0.05 0.001 0.0001
  do
    for optimizer in adam sgd
    do
      gpu=$(( (gpu + 1) % 8 ))
      python main.py --dataset book_crossing \
                     --method lightgcn \
                     --note hps_bookcrossing_0801_0400 \
                     --hps \
                     --optimizer $optimizer \
                     --batch_size $bs \
                     --epochs 50 \
                     --lr $lr \
                     --gpu $gpu &
    done
    wait
  done
done