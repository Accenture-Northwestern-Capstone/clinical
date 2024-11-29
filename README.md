# Accenture x Northwestern: Synthetic EEG Signal Augmentation for Enchanced Seizure Detection

## Overview

This project aims to develop and evaluate models for seizure detection using EEG data. It includes data cleaning, model training, and testing model robustness through challenging test cases. Additionally, it explores EEG signal reconstruction and augmentation using autoencoders and variational autoencoders (VAEs).


## File Directory

#### Root Files
- `README.md`

#### Data Directory
- `data/`
  - `data.csv`： original dataset
  - `data_cleaned.csv`: cleaned dataset
  - `hard_test_cases/`
    - `train.csv`
    - `test.csv`

#### Models Directory
- `models/`
  - `CVAE/`
    - `class_embedding_model/`
    - `cvae_decoder_model/`
  - `autoencoder.h5`
  - `autoencoder_2d_dropout.h5`

#### Notebooks
- `eda_and_cleaning.ipynb`
- `baseline_detection_model.ipynb`
- `generate_challenging_test_cases.ipynb`
- `data_augmentation/`: recommend to use GPU
  - `train_autoencoder.ipynb`
  - `train_vae_and_conditional_vae.ipynb` 
  - `data_augmentation_pipeline.ipynb`


## Usage

1. Create environment
```shell
conda create --name seizuredetection python=3.10 -y
conda activate seizuredetection
pip install .
```

2. Download model files from [Google Drive](https://drive.google.com/file/d/1g6DlQMerSRmP_ODjevdYUmoKtIAkfEjp/view?usp=drive_link) and put the `models/` folder in the project directory.


3. Run the notebooks in sequence for a complete workflow:
   - Start with `eda_and_cleaning.ipynb` to clean and preprocess the data.
   - Train baseline models using `baseline_detection_model.ipynb`.
   - Generate challenging test cases with `generate_challenging_test_cases.ipynb`.
   - Perform EEG signal reconstruction and augmentation with `train_autoencoder.ipynb`, `train_vae_and_conditional_vae.ipynb`.
   - Augment signals and evaluate again with seizure dataset with `data_augmentation_pipeline.ipynb`.

## Details

### 1. Data Cleaning, Transformation, and Exploratory Data Analysis (EDA)

---

#### Notebook: `eda_and_cleaning.ipynb`
**Description**: This notebook performs initial data cleaning, transformations, and exploratory data analysis to prepare the dataset for modeling.

**Output Files**:
- `data/data.csv`: The original raw dataset.
- `data/cleaned_data.csv`: The cleaned dataset focused on seizure data.

### 2. Baseline Seizure Detection Model with Performance Metrics

---

#### Notebook: `baseline_detection_model.ipynb`
**Description**: Implements baseline models such as SVM, ANN, and Inception Nucleus. Performance is evaluated using standard metrics.

### 3. Generating Challenging Test Cases

---

#### Notebook: `generate_challenging_test_cases.ipynb`
**Description**: This notebook generates challenging test cases to assess the robustness of seizure detection models.

**Output Files**:
- `data/hard_test_cases/train.csv`: Training data for hard test cases.
- `data/hard_test_cases/test.csv`: Hard test cases for model evaluation.

### 4. Data Augmentation

---

#### Notebook 1: `train_autoencoder.ipynb`
**Description**: Trains an autoencoder model for EEG signal reconstruction.

**Output Files**:
- Trained autoencoder model.

#### Notebook 2: `train_vae_and_conditional_vae.ipynb`
**Description**: Trains both a Variational Autoencoder (VAE) and a Conditional VAE for EEG signal augmentation.

**Output Files**:
- Trained VAE and Conditional VAE models.

#### Notebook 3: `data_augmentation_pipeline.ipynb`
**Description**: Implements a synthetic data augmentation pipeline to generate variations of EEG signals using the following methods:
1. **Noise**: Add noise to the data.
2. **Time Warping**: Alters the temporal properties of the signal.
3. **Time Drifting**: Add drift to simulate gradual shifts over time.
4. **Time Croping**: Crop the time-series to 90% of its original length.
5. **Amplitude Scaling**: Scales the amplitude of the signal.
6. **Autoencoder**: Reconstructs or augments EEG signals.
7. **Conditional Variational Autoencoder (CVAE)**: Generates augmented data conditioned on specific labels.

**Output**: A function with original signals and augmentation methods as input, and reconstructed EEG signals as output.

The data augmentation pipeline evaluates seizure detection model performance after applying various augmentation techniques, ensuring robustness and accuracy across diverse datasets.

---

Feel free to reach out to authors for more details:

Naman Garg: namangarg2025@u.northwestern.edu

Meixi Lu: meixilu2025@u.northwestern.edu

Ramirez-Aristizabal, Adolfo: adolfo.ramirez@accenture.com
