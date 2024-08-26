for outer in 2 4 6 8 10 12 14 16 18 20
do
  gpu=2
  for idx in 0 1
  do
    gpu=$(( (gpu + 1) % 8 ))
    layers=$((outer + idx))
    for seed in 0 1 2
    do
      python main.py --dataset cora \
                     --method gcn \
                     --note layers_imapct_cora_0826_0940 \
                     --hps \
                     --num_layers $layers \
                     --dropout 0 \
                     --epochs 50 \
                     --gpu $gpu \
                     --random_features \
                     --seed $seed &
    done
  done
  wait
done
