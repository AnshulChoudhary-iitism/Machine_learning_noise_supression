"""
System architecture documentation for ML Swell Noise Suppression
"""

# ML-Assisted Swell Noise Suppression - System Architecture

## Overview

This document describes the complete architecture of the ML-assisted swell noise suppression system for towed marine seismic data.

## System Components

### 1. Data Pipeline (src/data/)

#### Loaders
- **SEG-Y Loading**: Uses `obspy` and `segyio` to read standard seismic data formats
- **NumPy Loading**: Support for `.npy` and `.npz` formats for processed data
- **Format Auto-detection**: Automatically detects file format

#### Preprocessing
- **Normalization**: Multiple methods (min-max, z-score, robust)
- **Windowing**: Overlapping trace segmentation
- **Filtering**: Butterworth bandpass filter
- **DC Offset Removal**: Elimination of DC components
- **Denormalization**: Reversible preprocessing for reconstruction

#### Synthetic Data Generation
- **Swell Noise**: Multi-harmonic ocean swell model (0.5-5 Hz)
- **Seismic Signals**: Ricker wavelet-based synthetic reflections
- **Gaussian Noise**: White noise addition
- **SNR Control**: Configurable signal-to-noise ratio
- **Realistic Patterns**: Multiple overlapping reflections with random amplitudes

#### Augmentation
- **Time Shifting**: Random temporal offset
- **Amplitude Scaling**: Random amplitude multiplication
- **Noise Injection**: Controlled noise addition
- **Random Flipping**: Trace reversal
- **Cropping**: Random trace segmentation

### 2. Feature Extraction (src/features/)

#### Spectral Analysis
- **FFT**: Fast Fourier Transform for frequency domain analysis
- **Spectrograms**: Short-time Fourier Transform (STFT)
- **Wavelet Transform**: Continuous wavelet transform (CWT)
- **Amplitude Spectrum**: Magnitude of frequency components
- **Phase Spectrum**: Phase information preservation

#### Temporal Features
- **Amplitude Envelope**: Analytic signal magnitude
- **Instantaneous Frequency**: Time-varying frequency analysis
- **Instantaneous Phase**: Phase unwrapping

#### Statistics
- **Spectral Centroid**: Center of mass of spectrum
- **Spectral Spread**: Bandwidth measure
- **Spectral Rolloff**: Frequency threshold for 85% energy

### 3. ML Models (src/models/)

#### U-Net Architecture
```
Input [2048, 1]
  ↓
[Encoder] 4-level hierarchical downsampling with skip connections
  ↓ Maxpooling (stride 2)
  ↓ Conv1D (filters: 32→64→128→256)
  ↓ Batch Normalization
  ↓ Dropout (rate: 0.2)
  ↓
[Bottleneck] Compressed representation
  ↓
[Decoder] 4-level hierarchical upsampling
  ↓ Upsampling (factor 2)
  ↓ Concatenate with encoder features (skip connection)
  ↓ Conv1D (filters: 256→128→64→32)
  ↓
Output [2048, 1] with Tanh activation
```

**Advantages:**
- Skip connections preserve fine-grained details
- Hierarchical feature extraction at multiple scales
- Effective for image-to-image translation
- Proven architecture for seismic denoising

#### Autoencoder Architecture
```
Input [2048, 1]
  ↓
[Encoder] Progressive compression
  ↓ Conv1D (32, 64, 128)
  ↓ MaxPooling (stride 2)
  ↓ Batch Normalization + Dropout
  ↓
[Bottleneck] Latent representation (64 filters)
  ↓
[Decoder] Progressive expansion
  ↓ Upsampling (factor 2)
  ↓ Conv1D (128, 64, 32)
  ↓ Batch Normalization + Dropout
  ↓
Output [2048, 1] with Tanh activation
```

**Advantages:**
- Unsupervised learning approach
- Learns latent representations
- Good for anomaly detection
- Efficient compression-decompression

### 4. Training System (src/training/)

#### Trainer Class
- **Model Compilation**: Setup optimizer, loss, and metrics
- **Training Loop**: Full epoch training with validation
- **Callbacks**:
  - ModelCheckpoint: Save best models
  - EarlyStopping: Prevent overfitting
  - ReduceLROnPlateau: Dynamic learning rate adjustment
  - TensorBoard: Real-time monitoring

#### Loss Functions
| Function | Purpose | Formula |
|----------|---------|---------|
| MSE | Reconstruction error | (1/n)Σ(y-ŷ)² |
| MAE | Robust reconstruction | (1/n)Σ\|y-ŷ\| |
| Huber | Balanced robustness | Combination of L1/L2 |
| Perceptual | Frequency-domain matching | FFT-based L1 loss |
| SNR | Signal quality | 10·log₁₀(P_signal/P_noise) |
| Combined | Weighted combination | α·L1 + β·Perceptual |

#### Metrics
| Metric | Purpose |
|--------|---------|
| MSE | Mean squared error |
| MAE | Mean absolute error |
| SNR | Signal-to-noise ratio (dB) |
| Phase Similarity | Phase coherence preservation |
| Freq. Domain MAE | Spectral fidelity |
| Improved SNR | SNR gain from denoising |

### 5. Inference Engine (src/inference/)

#### Predictor Class
- **Batch Prediction**: Efficient GPU-based inference
- **Single Trace Prediction**: Individual trace denoising
- **Streaming Prediction**: Sequential batch processing
- **Ensemble Prediction**: Multiple model voting (mean, median, uncertainty)

#### Post-Processing
- **Amplitude Clipping**: Limit to [-1, 1] range
- **Smoothing**: Moving average filtering
- **Bandpass Filtering**: Artifact removal
- **Morphological Denoising**: Median filtering
- **Spike Removal**: Statistical outlier detection and removal
- **DC Removal**: Eliminate residual offset

### 6. PyQt5 GUI Application (src/ui/)

#### Main Window
- Multi-tab interface for complete workflow
- Professional styling and layout
- Real-time status updates

#### Tab 1: Data Loading
- Load SEG-Y files from disk
- Generate synthetic training data
- Data visualization (2D seismic display)
- Statistical summary

#### Tab 2: Model Training
- Model selection (U-Net / Autoencoder)
- Hyperparameter input (epochs, batch size, learning rate, loss)
- Real-time training progress
- Loss curve visualization
- Model checkpointing

#### Tab 3: Model Inference
- Load trained models
- Run batch inference
- Before/after comparison visualization
- SNR improvement display
- Export denoised data

#### Tab 4: Configuration
- Edit YAML configuration
- Load/save custom configurations
- Real-time parameter management

#### Custom Widgets
- **SeismicPlotter**: 2D seismic display with colormap
- **TrainingMonitor**: Real-time loss plotting
- **ParameterForm**: Dynamic form generation
- **FileSelector**: Cross-platform file browser

### 7. CLI Scripts (scripts/)

#### train.py
- Train models from command line
- Flexible parameter configuration
- Support for synthetic and real data

#### evaluate.py
- Evaluate trained models
- Compute metrics (MSE, MAE, SNR)
- Generate evaluation reports

#### generate_synthetic_data.py
- Create synthetic training datasets
- Configurable noise parameters
- Batch generation support

#### run_gui.py
- Launch PyQt5 application
- Handle logging setup

## Data Flow

```
Raw Data (SEG-Y)
    ↓
[Data Loading] → Load & Parse
    ↓
[Preprocessing]
├─ Normalization
├─ Windowing
├─ DC Removal
└─ Filtering
    ↓
[Augmentation] (Optional)
├─ Time Shift
├─ Amplitude Scale
├─ Noise Injection
└─ Flipping
    ↓
[Feature Extraction]
├─ FFT
├─ Spectrograms
└─ Statistical Features
    ↓
[ML Model]
├─ U-Net or Autoencoder
└─ Forward Pass
    ↓
[Inference]
├─ Batch Processing
└─ Streaming Mode
    ↓
[Post-Processing]
├─ Clipping
├─ Smoothing
├─ Filtering
└─ Spike Removal
    ↓
Output (Denoised Seismic Data)
```

## Deployment Architecture

### Training Phase
```
GPU (TensorFlow)
├─ Data Loading
├─ Preprocessing
├─ Model Training
├─ Validation
└─ Checkpointing → Best Model (H5)
```

### Inference Phase
```
GPU or CPU
├─ Model Loading
├─ Batch Processing
├─ Real-time Prediction
└─ Post-Processing → Output
```

## Performance Considerations

- **Batch Size**: Larger batches (32-64) for training, smaller (16-32) for inference
- **Data Format**: 1D convolutions are more efficient than 2D for traces
- **Memory Usage**: ~500MB for model + data in typical configs
- **Inference Speed**: ~100-500ms per trace depending on GPU

## Extensibility

The architecture is designed for easy extension:

1. **Add New Models**: Inherit from `BaseModel` or `EncoderDecoderModel`
2. **Add New Loss Functions**: Create class inheriting from `keras.losses.Loss`
3. **Add New Metrics**: Inherit from `keras.metrics.Metric`
4. **Add New Augmentations**: Add methods to `SeismicAugmentor`
5. **Add New Features**: Extend `SeismicFeatureExtractor`
