# ML-Assisted Swell Noise Suppression in Towed Marine Seismic Data

A comprehensive machine learning pipeline for suppressing swell noise in marine seismic data using deep learning models and a professional PyQt5 GUI application.

## Overview

This project implements an end-to-end system for:
- Loading and processing SEG-Y marine seismic data
- Generating synthetic training datasets with realistic swell noise patterns
- Training deep learning models (U-Net, Autoencoder) for noise suppression
- Real-time inference and visualization
- Interactive PyQt5 desktop application for model training and prediction

## Features

✅ **Complete Data Pipeline**
- SEG-Y file loading using obspy/segyio
- Advanced preprocessing (normalization, windowing, filtering)
- Synthetic data generation with configurable noise models
- Data augmentation techniques

✅ **ML Models**
- U-Net architecture optimized for seismic denoising
- Autoencoder-based unsupervised learning approach
- Custom loss functions for seismic signals
- Comprehensive evaluation metrics

✅ **Training System**
- Full training loop with validation
- Early stopping and model checkpointing
- Real-time monitoring with TensorBoard
- Hyperparameter management

✅ **PyQt5 GUI Application**
- Multi-tab interface for easy workflow
- Real-time visualization of seismic data
- Interactive training monitoring
- Model inference and result export
- Configuration management

✅ **Production-Ready**
- Comprehensive testing framework (pytest)
- Proper error handling and logging
- Well-documented code
- CI/CD workflows (GitHub Actions)

## Quick Start

### Prerequisites
- Python 3.9+
- GPU support (CUDA/cuDNN) recommended

### Installation

```bash
# Clone the repository
git clone https://github.com/AnshulChoudhary-iitism/Machine_learning_noise_supression.git
cd Machine_learning_noise_supression

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Usage

#### 1. Launch GUI Application
```bash
python scripts/run_gui.py
```

#### 2. Generate Synthetic Data
```bash
python scripts/generate_synthetic_data.py \
    --output data/synthetic_data.npy \
    --num_samples 1000 \
    --trace_length 2048
```

#### 3. Train Model (CLI)
```bash
python scripts/train.py \
    --config config.yaml \
    --data data/processed/ \
    --model unet \
    --epochs 100 \
    --batch_size 32
```

#### 4. Evaluate Model
```bash
python scripts/evaluate.py \
    --model data/models/best_model.h5 \
    --test_data data/processed/test/
```

## Project Structure

```
Machine_learning_noise_supression/
├── src/
│   ├── data/              # Data loading and preprocessing
│   ├── features/          # Feature extraction
│   ├── models/            # ML model architectures
│   ├── training/          # Training pipeline
│   ├── inference/         # Inference engine
│   ├── ui/                # PyQt5 GUI application
│   ├── config.py          # Configuration management
│   └── utils.py           # Utility functions
├── scripts/               # CLI tools
├── tests/                 # Test suite
├── notebooks/             # Jupyter notebooks
├── docs/                  # Documentation
├── data/                  # Data directory (local)
├── requirements.txt       # Python dependencies
├── setup.py              # Package setup
└── config.yaml           # Configuration file
```

## Documentation

- **[Architecture Guide](docs/architecture.md)** - System design and component interactions
- **[Data Format](docs/data_format.md)** - SEG-Y specifications
- **[Model Development](docs/model_development.md)** - ML models and training
- **[Usage Guide](docs/usage_guide.md)** - Detailed usage examples

## Models

### U-Net
Encoder-decoder architecture with skip connections, optimized for seismic image-to-image translation.

**Features:**
- Residual blocks for better gradient flow
- Multi-scale feature extraction
- Skip connections for fine-grained details

### Autoencoder
Unsupervised learning approach for noise suppression through reconstruction-based denoising.

**Features:**
- Bottleneck architecture
- Symmetric encoder-decoder
- Reconstruction loss optimization

## Training

```python
from src.training.trainer import Trainer
from src.models.unet import UNet
import tensorflow as tf

# Initialize model
model = UNet(input_shape=(2048, 1))

# Create trainer
trainer = Trainer(model, config='config.yaml')

# Train
history = trainer.train(
    train_data=train_dataset,
    val_data=val_dataset,
    epochs=100,
    batch_size=32
)
```

## Inference

```python
from src.inference.predictor import Predictor

# Load model
predictor = Predictor('data/models/best_model.h5')

# Predict
suppressed_data = predictor.predict(noisy_data)
```

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_models.py -v

# Run with coverage
pytest tests/ --cov=src
```

## Configuration

Edit `config.yaml` to customize:
- Model architecture parameters
- Training hyperparameters
- Data preprocessing options
- Synthetic data generation settings
- GUI preferences

```yaml
model:
  type: unet
  input_shape: [2048, 1]
  filters: [32, 64, 128, 256]

training:
  learning_rate: 0.001
  batch_size: 32
  epochs: 100
  early_stopping_patience: 10

data:
  normalization: true
  window_size: 2048
  overlap: 0.5
```

## Dependencies

- **Data Processing**: numpy, scipy, pandas
- **Deep Learning**: TensorFlow/Keras
- **Seismic I/O**: obspy, segyio
- **ML Utilities**: scikit-learn
- **Visualization**: matplotlib, plotly
- **GUI**: PyQt5
- **Testing**: pytest

## Performance

Expected improvements on typical marine seismic data:
- **SNR Improvement**: 5-15 dB
- **Inference Speed**: ~100-500 ms per trace (GPU-dependent)
- **Model Size**: 50-150 MB

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{ml_noise_suppression,
  title={ML-Assisted Swell Noise Suppression in Towed Marine Seismic Data},
  author={Anshul Choudhary},
  year={2025},
  url={https://github.com/AnshulChoudhary-iitism/Machine_learning_noise_supression}
}
```

## Authors

- **Anshul Choudhary** - Initial development

## Acknowledgments

- Seismic data processing techniques from geophysics literature
- Deep learning architectures from computer vision research
- ObsPy and segyio communities for seismic data handling

## Support

For issues, questions, or suggestions, please open an [GitHub issue](https://github.com/AnshulChoudhary-iitism/Machine_learning_noise_supression/issues).

## Roadmap

- [ ] Integrate with real seismic data benchmarks
- [ ] Add Transformer-based architectures
- [ ] Support for 3D seismic data
- [ ] Real-time processing optimization
- [ ] Model compression and quantization
- [ ] Distributed training support
- [ ] Cloud deployment guides

---

**Project Status**: Active Development 🚀
