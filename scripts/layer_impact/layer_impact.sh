for outer in 2 4 6
do
  gpu=-1
  for idx in 0 1
  do
    gpu=$(( (gpu + 1) % 8 ))
    layers=$((outer + idx))
    python main.py --dataset movielense_small \
                   --method lightgcn \
                   --note layers_imapct_movielense_small_0904_1300 \
                   --hps \
                   --num_layers $layers \
                   --dropout 0 \
                   --epochs 100 \
                   --gpu $gpu &
  done
  wait
done
