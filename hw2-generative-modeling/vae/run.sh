
python train.py --log_dir ae_latent16 --loss_mode ae --latent_size 16
python train.py --log_dir ae_latent128 --loss_mode ae --latent_size 128
python train.py --log_dir ae_latent1024 --loss_mode ae --latent_size 1024

python train.py --log_dir vae_latent16 --loss_mode vae --latent_size 16
python train.py --log_dir vae_latent128 --loss_mode vae --latent_size 128
python train.py --log_dir vae_latent1024 --loss_mode vae --latent_size 1024

python train.py --log_dir vae_latent16_beta_.8 --loss_mode vae --latent_size 16 --target_beta_val 0.8
python train.py --log_dir vae_latent128_beta_.8 --loss_mode vae --latent_size 128 --target_beta_val 0.8
python train.py --log_dir vae_latent1024_beta_.8 --loss_mode vae --latent_size 1024 --target_beta_val 0.8

python train.py --log_dir vae_latent16_beta_1.2 --loss_mode vae --latent_size 16 --target_beta_val 1.2
python train.py --log_dir vae_latent128_beta_1.2 --loss_mode vae --latent_size 128 --target_beta_val 1.2
python train.py --log_dir vae_latent1024_beta_1.2 --loss_mode vae --latent_size 1024 --target_beta_val 1.2

# Train the VAE for 20 epochs with beta annealing using the value of beta that results in the best samples, 
# out of 0.8, 1.0, and 1.2. Use the latent size from 2.3 that produces the sharpest reconstructions.

