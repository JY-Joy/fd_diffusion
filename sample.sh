python3 scripts/gen_image_script.py \
  --ckpt_path /apdcephfs/default121254/apdcephfs_cq10/share_916081/jentsehuang/decomp_diffusion/celebahq/unet_model_celebahq_None_xstart_emb_256_implementation_1_256x2latent_batch12x6_retry/checkpoint-40000/model.safetensors \
  --save_dir ./sample_images/ \
  --image_size 128 \
  --im_path ./val_imgs/celebahq_7.jpg \
  --dataset celebahq \
  --num_components 4 \
  --emb_dim 256 \
  --separate
  --combine_method add
