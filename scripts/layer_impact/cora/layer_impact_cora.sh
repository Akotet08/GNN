for outer in 1 5 9
do
  gpu=-1
  second=$((outer + 2))
  for idx in $outer $second
  do
    gpu=$(( (gpu + 1) % 8 ))
    for inner in 0 1
    do
      layers=$(((inner + idx)*10))
      python main.py --dataset cora \
                     --method gcn \
                     --note layers_imapct_cora_0909_1100 \
                     --hps \
                     --num_layers $layers \
                     --dropout 0 \
                     --epochs 50 \
                     --gpu $gpu &
    done
  done
  wait
done
