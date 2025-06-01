# Hyperparameters
latent_dim = 512
batch_size = 32
learning_rate = 0.0001
epochs = 50
beta = 1.0  # Weight for KL divergence
max_seq_length = 64  # Maximum sequence length for BERT

# Image transforms
image_size = 256
transform_mean = [0.5, 0.5, 0.5]
transform_std = [0.5, 0.5, 0.5]

# Paths
save_dir = "saved_models"
sample_dir = "generated_samples"