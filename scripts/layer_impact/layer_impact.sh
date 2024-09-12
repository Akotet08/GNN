for outer in 1 3 5 7
do
  gpu=-1
  for idx in 0 1
  do
    gpu=$(( (gpu + 1) % 8 ))
    layers=$(((outer + idx)*10))
    python main.py --dataset movielense_small \
                   --method lightgcn \
                   --note layers_imapct_movielense_small_0911_0100 \
                   --hps \
                   --num_layers $layers \
                   --dropout 0.5 \
                   --epochs 50 \
                   --gpu $gpu &
  done
  wait
done
