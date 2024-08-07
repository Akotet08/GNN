gpu=0
for outer in 2 5 8 11 14
do
  for idx in 0 1 2
  do
    gpu=$(( (gpu + 1) % 8 ))
    layers=$((outer + idx))
    python main.py --dataset movielense \
                   --method lightgcn \
                   --note layers_imapct_movielense_0806_1830 \
                   --hps \
                   --num_layers $layers \
                   --epochs 100 \
                   --gpu $gpu &
  done
  wait
done
