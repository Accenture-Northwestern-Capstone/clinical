import numpy as np
from tensorflow.keras.models import load_model

# Paths to saved decoder models
vae_decoder_path = './models/VAE/decoder_model.h5' 
cvae_decoder_path = './models/CVAE/cvae_decoder_model.h5'

# Load the decoders
vae_decoder = load_model(vae_decoder_path) # Load the VAE decoder
cvae_decoder = load_model(cvae_decoder_path) # Load the CVAE decoder

# Generate synthetic data using the VAE decoder
def generate_vae_synthetic_data(num_samples=10, latent_dim=32):
    """
    Generates synthetic data using the VAE decoder.

    Args:
    num_samples (int): Number of synthetic samples to generate.
    latent_dim (int): Dimensionality of the latent space.

    Returns:
    np.ndarray: Generated synthetic data.
    """
    num_synthetic_samples = num_samples
    latent_samples = np.random.normal(
        loc=0.0,
        scale=2.0,  
        size=(num_synthetic_samples, latent_dim)
    )
    synthetic_eeg = vae_decoder.predict(latent_samples)
    return synthetic_eeg

# Generate synthetic data using the CVAE decoder
def generate_cvae_synthetic_data(class_value, n_samples=1):
    """
    Generates synthetic data using the CVAE decoder for a specific class.
    
    Args:
    class_value (int): Class label for which to generate synthetic data.
    n_samples (int): Number of synthetic samples to generate.

    Returns:
    np.ndarray: Generated synthetic
    """
    # Prepare the class labels array as input, repeating it n_samples times
    class_label_input = np.array([[class_value]] * n_samples)  # Shape (n_samples, 1)

    # Embed the class labels using the same embedding layer as in the encoder
    class_embedding_model = load_model('./models/class_embedding_model.h5')
    embedded_class_labels = class_embedding_model.predict(class_label_input)  # Shape (n_samples, latent_dim)

    # Generate random samples from the latent space with randomness each time
    latent_samples = np.random.normal(loc=0, scale=3, size=(n_samples, latent_dim))  # Shape (n_samples, latent_dim)

    # Concatenate latent samples with embedded class labels
    latent_samples_with_class = np.concatenate([latent_samples, embedded_class_labels], axis=-1)  # Shape (n_samples, latent_dim + latent_dim)

    # Generate synthetic data using the decoder
    synthetic_data = cvae_decoder.predict(latent_samples_with_class)  # Shape (n_samples, 178, 1)
    return synthetic_data

# Example usage
num_samples = 5
latent_dim = 32

# Generate VAE synthetic data
vae_synthetic_data = generate_vae_synthetic_data(num_samples, latent_dim)
# print("VAE Synthetic Data:\n", vae_synthetic_data)

# Generate CVAE synthetic data for a specific class (e.g., class_label = 1)
synthetic_seizure_data = generate_cvae_synthetic_data(1,5)  # Generate data for Seizure class
synthetic_non_seizure_data = generate_cvae_synthetic_data(0,5)  # Generate data for Non-Seizure class