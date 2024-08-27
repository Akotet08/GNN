for outer in 2 4 6 8 10 12 14 16 18 20
do
  gpu=2
  for idx in 0 1
  do
    gpu=$(( (gpu + 1) % 8 ))
    layers=$((outer + idx))
    for seed in 0 1 2
    do
      python main.py --dataset pubmed \
                     --method gcn \
                     --note layers_imapct_pubmed_random_features_0826_1900 \
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
