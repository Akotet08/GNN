for outer in 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50 52 54 56 58 60
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
                     --note layers_imapct_cora_random_features_0903_2000 \
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
