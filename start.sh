source activate decomp_diff
accelerate launch train.py \
  --data_dir /scratch4/mdredze1/jhuan236/data \
  --output_dir /scratch4/mdredze1/jhuan236/logs/decomp_diffusion/cross_attn \
  --logging_dir /scratch4/mdredze1/jhuan236/logs/decomp_diffusion/cross_attn \
  --enc_channels 64 \
  --image_size 128 \
  --train_batch_size 32 \
  --num_components 4 \
  --emb_dim 256 \
  --time_embed_dim 256 \
  --encoder_channels 128 \
  --p_uncond 0.0 \
  --regu_weight 5e-4 \
  --dataset celebahq \
  --max_train_steps 80000 \
  --extra_desc test

