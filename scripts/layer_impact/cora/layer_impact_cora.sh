for outer in 2 6 10 14 18 22 26 30 34 38 42 46 50 54 58 62
do
  gpu=2
  second=$((outer + 2))
  for idx in $outer $second
  do
    gpu=$(( (gpu + 1) % 8 ))
    for inner in 0 1
    do
      layers=$((inner + idx))
      python main.py --dataset cora \
                     --method gcn \
                     --note layers_imapct_cora_0903_2000 \
                     --hps \
                     --num_layers $layers \
                     --dropout 0 \
                     --epochs 50 \
                     --gpu $gpu &
    done
  done
  wait
done
