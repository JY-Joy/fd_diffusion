source activate decomp_diff

python3 scripts/gen_image_script.py \
  --ckpt_path /home/jhuan236/scr4_mdredze1/jhuan236/logs/decomp_diffusion/cross_attn/celebahq_xstart_emb_256_comp_4_CFG_latent_dropout/checkpoint-80000/model.safetensors \
  --save_dir ./sample_images/ \
  --image_size 64 \
  --im_path ./val_imgs/celebahq_0.jpg \
  --im_path_2 ./val_imgs/celebahq_1.jpg \
  --dataset celebahq \
  --num_components 4 \
  --enc_channels 64 \
  --emb_dim 256 \
  --time_embed_dim 256 \
  --encoder_channels 128 \
  --seed 42 \
  --indices "0,1,3"
  --separate
