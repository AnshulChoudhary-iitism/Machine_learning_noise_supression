"""
Comprehensive usage guide for ML Swell Noise Suppression
"""

# Usage Guide - ML-Assisted Swell Noise Suppression

## Table of Contents
1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Command Line Usage](#command-line-usage)
4. [GUI Application](#gui-application)
5. [Python API](#python-api)
6. [Troubleshooting](#troubleshooting)

## Installation

### System Requirements
- Python 3.9+
- 4GB RAM minimum (8GB recommended)
- GPU with CUDA support (optional, for faster training)

### Step-by-Step Installation

```bash
# 1. Clone repository
git clone https://github.com/AnshulChoudhary-iitism/Machine_learning_noise_supression.git
cd Machine_learning_noise_supression

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. (Optional) Install for development
pip install -e .
```

### GPU Setup (Optional)

For CUDA support:
```bash
# Install CUDA-enabled TensorFlow
pip install tensorflow[and-cuda]

# Or separately:
pip install tensorflow>=2.11.0
pip install tensorflow-gpu>=2.11.0
```

## Quick Start

### Generate Synthetic Data

```bash
python scripts/generate_synthetic_data.py \
    --output data/synthetic_data.npy \
    --num_samples 1000 \
    --trace_length 2048 \
    --sample_rate 1000.0 \
    --seed 42
```

**Output**: `data/synthetic_data.npy` containing 2000 traces (1000 noisy + 1000 clean)

### Train a Model

```bash
python scripts/train.py \
    --model unet \
    --data data/synthetic_data.npy \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --loss mse \
    --output_dir data/models/
```

**Output**: 
- `data/models/best_model.h5` - Best model saved during training
- `data/models/unet_trained.h5` - Final trained model
- `logs/training_history.json` - Training metrics

### Evaluate Model

```bash
python scripts/evaluate.py \
    --model data/models/unet_trained.h5 \
    --test_data data/synthetic_data.npy \
    --output_dir data/results/
```

**Output**:
- `data/results/evaluation_results.json` - Metrics report

### Launch GUI Application

```bash
python scripts/run_gui.py
```

## Command Line Usage

### Synthetic Data Generation

```bash
python scripts/generate_synthetic_data.py \
    --output <output_file> \
    --num_samples <number> \
    --trace_length <length> \
    --sample_rate <rate> \
    --snr <snr_db> \
    --separate \
    --seed <seed>
```

**Parameters**:
- `--output`: Output file path (.npy)
- `--num_samples`: Number of samples to generate (default: 1000)
- `--trace_length`: Length of each trace (default: 2048)
- `--sample_rate`: Sampling rate in Hz (default: 1000.0)
- `--snr`: Target SNR in dB (default: 5.0)
- `--separate`: Save noisy and clean data separately
- `--seed`: Random seed for reproducibility

**Examples**:
```bash
# Generate 500 traces with high SNR
python scripts/generate_synthetic_data.py \
    --output data/high_snr_data.npy \
    --num_samples 500 \
    --snr 10.0

# Generate with separate files
python scripts/generate_synthetic_data.py \
    --output data/training_data.npy \
    --num_samples 2000 \
    --separate
```

### Model Training

```bash
python scripts/train.py \
    --model <model_type> \
    --data <data_path> \
    --epochs <epochs> \
    --batch_size <size> \
    --learning_rate <lr> \
    --loss <loss_function> \
    --validation_split <split> \
    --output_dir <output_dir> \
    --verbose <level>
```

**Parameters**:
- `--model`: Model type: `unet` or `autoencoder`
- `--data`: Path to training data file
- `--synthetic`: Generate synthetic data instead of loading
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size for training (default: 32)
- `--learning_rate`: Learning rate (default: 0.001)
- `--loss`: Loss function: `mse`, `mae`, `l1`, `l2`, `huber`, `perceptual`, `snr`
- `--validation_split`: Validation/test split ratio (default: 0.2)
- `--output_dir`: Output directory for models (default: `data/models/`)
- `--verbose`: Verbosity level 0-2 (default: 1)

**Examples**:
```bash
# Train U-Net with synthetic data
python scripts/train.py \
    --model unet \
    --synthetic \
    --epochs 50 \
    --batch_size 16 \
    --learning_rate 0.0005

# Train Autoencoder with custom loss
python scripts/train.py \
    --model autoencoder \
    --data data/my_data.npy \
    --loss perceptual \
    --epochs 200 \
    --output_dir models/autoencoder/
```

### Model Evaluation

```bash
python scripts/evaluate.py \
    --model <model_path> \
    --test_data <test_data_path> \
    --batch_size <size> \
    --output_dir <output_dir>
```

**Parameters**:
- `--model`: Path to trained model (.h5)
- `--test_data`: Path to test data
- `--batch_size`: Batch size for evaluation (default: 32)
- `--output_dir`: Output directory for results (default: `data/results/`)

**Outputs**:
- Evaluation metrics (MSE, MAE, SNR, SNR improvement)

## GUI Application

### Launching the GUI

```bash
python scripts/run_gui.py
```

### Tab 1: Data Loading & Visualization

1. **Load SEG-Y File**
   - Click "Load SEG-Y File" button
   - Select your marine seismic data file
   - View data statistics in the info panel

2. **Generate Synthetic Data**
   - Click "Generate Synthetic Data"
   - Specify number of samples and trace length
   - View generated data visualization

3. **Data Visualization**
   - Displays seismic traces as 2D image
   - Color represents amplitude
   - Interactive zoom and pan

### Tab 2: Model Training

1. **Select Model**
   - Choose between U-Net and Autoencoder
   - View model information

2. **Configure Parameters**
   - Set number of epochs (typically 50-200)
   - Batch size: 16-64 depending on GPU memory
   - Learning rate: 0.0001 to 0.01
   - Loss function: Select appropriate loss

3. **Start Training**
   - Click "Start Training" button
   - Monitor progress bar and loss curves
   - View real-time training logs

4. **Save Model**
   - Best model automatically saved during training
   - Can load and continue training

### Tab 3: Model Inference

1. **Load Trained Model**
   - Click "Load Trained Model"
   - Select model file (.h5)
   - View model information

2. **Load Data**
   - Load test/validation data
   - Preview data statistics

3. **Run Inference**
   - Click "Run Inference"
   - Monitor progress
   - View before/after comparison

4. **Export Results**
   - Export denoised data to file
   - Save comparison plots
   - Generate evaluation report

### Tab 4: Configuration

1. **Edit Configuration**
   - YAML configuration editor
   - Modify model parameters
   - Adjust training settings

2. **Save Configuration**
   - Save custom configurations
   - Load previously saved configs
   - Reuse optimal settings

## Python API

### Basic Workflow

```python
import numpy as np
from src.data.synthetic_generator import generate_synthetic_seismic_data
from src.models.unet import create_unet
from src.training.trainer import Trainer

# Step 1: Generate or load data
noisy_data, clean_data = generate_synthetic_seismic_data(
    n_samples=1000,
    trace_length=2048
)

# Ensure 3D shape
noisy_data = np.expand_dims(noisy_data, axis=-1)
clean_data = np.expand_dims(clean_data, axis=-1)

# Step 2: Split data
from src.utils import split_data
X_train, X_val, X_test = split_data(noisy_data, train_ratio=0.7, val_ratio=0.15)
y_train, y_val, y_test = split_data(clean_data, train_ratio=0.7, val_ratio=0.15)

# Step 3: Create model
model = create_unet(input_shape=(2048, 1))

# Step 4: Create trainer
trainer = Trainer(model, checkpoint_dir='models/')

# Step 5: Compile and train
trainer.compile(loss='mse', optimizer='adam', learning_rate=0.001)
history = trainer.train(
    train_data=(X_train, y_train),
    val_data=(X_val, y_val),
    epochs=100,
    batch_size=32
)

# Step 6: Evaluate
results = trainer.evaluate((X_test, y_test))

# Step 7: Save model
trainer.save_model('best_model.h5')
```

### Data Loading

```python
from src.data.loaders import load_seismic_data

# Load SEG-Y file
data = load_seismic_data('marine_data.segy')

# Load NumPy file
data = load_seismic_data('data.npy')

# Auto-detect format
data = load_seismic_data('data.sgy')  # Automatically uses SEG-Y loader
```

### Data Preprocessing

```python
from src.data.preprocessing import SeismicPreprocessor

preprocessor = SeismicPreprocessor()

# Normalize data
normalized, stats = preprocessor.normalize(data, method='minmax')

# Remove DC offset
data_dc_removed = preprocessor.remove_dc_offset(data)

# Apply bandpass filter
filtered = preprocessor.apply_bandpass_filter(
    data,
    lowcut=5.0,
    highcut=100.0
)
```

### Model Prediction

```python
from src.inference.predictor import Predictor

# Create predictor
predictor = Predictor('best_model.h5')

# Make predictions
predictions = predictor.predict(test_data)

# Single trace prediction
single_pred = predictor.predict_single(single_trace)
```

### Post-Processing

```python
from src.inference.postprocessing import PostProcessor

processor = PostProcessor()

# Apply post-processing
processed = processor.postprocess(
    predictions,
    clip=True,
    smooth=True,
    bandpass=False
)
```

## Troubleshooting

### Problem: Out of Memory Error

**Solution**:
```bash
# Reduce batch size
python scripts/train.py \
    --batch_size 8 \
    --model unet ...

# Use smaller trace length
python scripts/generate_synthetic_data.py \
    --trace_length 512 ...
```

### Problem: GPU Not Detected

**Solution**:
```python
# Check GPU availability
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))

# Force CPU usage
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

### Problem: Model Not Improving

**Solution**:
- Increase training epochs
- Adjust learning rate (try 0.0005 or 0.005)
- Try different loss function (e.g., perceptual loss)
- Verify data quality and normalization

### Problem: ImportError

**Solution**:
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade

# Check Python version
python --version  # Should be 3.9+
```

### Problem: SEG-Y File Not Loading

**Solution**:
```bash
# Check file format
file your_data.segy

# Try with obspy
python -c "from obspy import read; st = read('your_data.segy'); print(st)"
```

## Performance Tips

1. **Use GPU**: 5-10x faster than CPU
2. **Batch Processing**: Process multiple traces together
3. **Model Checkpointing**: Automatically saves best models
4. **Data Augmentation**: Improves generalization
5. **Learning Rate Scheduling**: Automatic reduction on plateau

## Getting Help

- Check documentation in `/docs/`
- Review example scripts in `/scripts/`
- Check test files in `/tests/` for usage examples
- Open an issue on GitHub for bugs/features
