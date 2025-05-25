model_name=DeepUHI

root_path_name=./dataset/
data_path_name=./Temperature/SDot_2021_2024_Updated.csv
model_id_name=traffic
data_name=custom
group_file=./dataset/region_group.csv


model_type='gcn'
seq_len=96
for pred_len in 120
do
for random_seed in 2024
do
    python -u run.py \
      --is_training 1 \
      --d_model 64 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --group_file $group_file \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 947 \
      --cycle 24 \
      --model_type $model_type \
      --train_epochs 60 \
      --patience 5 \
      --itr 1 --batch_size 64 --learning_rate 0.01 --random_seed $random_seed #--timeenc 1
done
done
