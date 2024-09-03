for outer in 2 4 6
do
  gpu=-1
  for idx in 0 1
  do
    gpu=$(( (gpu + 1) % 8 ))
    layers=$((outer + idx))
    python main.py --dataset book_crossing \
                   --method lightgcn \
                   --note layers_imapct_book_crossing_0903_1020 \
                   --hps \
                   --num_layers $layers \
                   --dropout 0 \
                   --epochs 50 \
                   --gpu $gpu &
  done
  wait
done
