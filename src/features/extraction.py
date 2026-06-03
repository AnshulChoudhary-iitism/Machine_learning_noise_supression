"""
Feature extraction for seismic data analysis.
Includes spectral analysis, time-frequency representations, and seismic attributes.
"""

import logging
import numpy as np
from scipy import signal, fft
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class SeismicFeatureExtractor:
    """Extract features from seismic data."""
    
    def __init__(self, sample_rate: float = 1000.0):
        """
        Initialize feature extractor.
        
        Args:
            sample_rate: Sampling rate (Hz)
        """
        self.sample_rate = sample_rate
        self.dt = 1.0 / sample_rate
    
    def compute_fft(
        self,
        data: np.ndarray,
        n_fft: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute FFT of seismic data.
        
        Args:
            data: Input signal
            n_fft: FFT size
        
        Returns:
            Tuple of (frequencies, magnitude spectrum)
        """
        n_fft = n_fft or len(data)
        
        if data.ndim == 1:
            spectrum = np.abs(fft.fft(data, n=n_fft))
        else:
            spectrum = np.abs(fft.fft(data, n=n_fft, axis=1))
        
        freqs = fft.fftfreq(n_fft, self.dt)[:n_fft // 2]
        spectrum = spectrum[:n_fft // 2] if data.ndim == 1 else spectrum[:, :n_fft // 2]
        
        return freqs, spectrum
    
    def compute_spectrogram(
        self,
        data: np.ndarray,
        window: str = 'hann',
        nperseg: int = 256,
        noverlap: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute spectrogram using STFT.
        
        Args:
            data: Input signal
            window: Window function
            nperseg: Segment length
            noverlap: Overlap length
        
        Returns:
            Tuple of (times, frequencies, spectrogram)
        """
        noverlap = noverlap or nperseg // 2
        
        freqs, times, spec = signal.spectrogram(
            data,
            fs=self.sample_rate,
            window=window,
            nperseg=nperseg,
            noverlap=noverlap
        )
        
        return times, freqs, spec
    
    def compute_wavelet_transform(
        self,
        data: np.ndarray,
        wavelet: str = 'morlet',
        scales: Optional[np.ndarray] = None,
        freq_range: Tuple[float, float] = (1, 100)
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute continuous wavelet transform.
        
        Args:
            data: Input signal
            wavelet: Wavelet type
            scales: Wavelet scales
            freq_range: Frequency range for scales
        
        Returns:
            Tuple of (scales, frequencies, cwt_result)
        """
        if scales is None:
            min_freq, max_freq = freq_range
            scales = np.logspace(
                np.log10(self.sample_rate / max_freq),
                np.log10(self.sample_rate / min_freq),
                100
            )
        
        coefficients = signal.cwt(data, signal.morlet2, scales)
        frequencies = self.sample_rate / (2 * np.pi * scales)
        
        return scales, frequencies, coefficients
    
    def compute_amplitude_envelope(self, data: np.ndarray) -> np.ndarray:
        """
        Compute amplitude envelope (analytic signal).
        
        Args:
            data: Input signal
        
        Returns:
            Amplitude envelope
        """
        analytic = signal.hilbert(data, axis=-1 if data.ndim > 1 else 0)
        envelope = np.abs(analytic)
        return envelope
    
    def compute_instantaneous_frequency(self, data: np.ndarray) -> np.ndarray:
        """
        Compute instantaneous frequency.
        
        Args:
            data: Input signal
        
        Returns:
            Instantaneous frequency
        """
        analytic = signal.hilbert(data, axis=-1 if data.ndim > 1 else 0)
        phase = np.unwrap(np.angle(analytic), axis=-1 if data.ndim > 1 else 0)
        
        # Derivative of phase
        if data.ndim == 1:
            inst_freq = np.diff(phase) / (2 * np.pi * self.dt)
        else:
            inst_freq = np.diff(phase, axis=1) / (2 * np.pi * self.dt)
        
        return inst_freq
    
    def compute_amplitude_spectrum(
        self,
        data: np.ndarray,
        n_fft: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute amplitude spectrum.
        
        Args:
            data: Input signal
            n_fft: FFT size
        
        Returns:
            Amplitude spectrum
        """
        freqs, spectrum = self.compute_fft(data, n_fft)
        return spectrum / len(data)
    
    def compute_phase_spectrum(
        self,
        data: np.ndarray,
        n_fft: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute phase spectrum.
        
        Args:
            data: Input signal
            n_fft: FFT size
        
        Returns:
            Phase spectrum
        """
        n_fft = n_fft or len(data)
        
        if data.ndim == 1:
            fft_result = fft.fft(data, n=n_fft)
        else:
            fft_result = fft.fft(data, n=n_fft, axis=1)
        
        phase = np.angle(fft_result)
        return phase[:n_fft // 2] if data.ndim == 1 else phase[:, :n_fft // 2]
    
    def extract_all_features(self, data: np.ndarray) -> dict:
        """
        Extract all available features.
        
        Args:
            data: Input signal
        
        Returns:
            Dictionary of features
        """
        features = {
            'mean': np.mean(data),
            'std': np.std(data),
            'min': np.min(data),
            'max': np.max(data),
            'rms': np.sqrt(np.mean(data ** 2)),
            'amplitude_envelope': self.compute_amplitude_envelope(data),
        }
        
        # Spectral features
        freqs, spectrum = self.compute_fft(data)
        features['spectrum'] = spectrum
        features['frequencies'] = freqs
        
        return features


def extract_features(
    data: np.ndarray,
    feature_types: list = ['statistical', 'spectral'],
    sample_rate: float = 1000.0
) -> dict:
    """
    Extract features from seismic data.
    
    Args:
        data: Input seismic data
        feature_types: Types of features to extract
        sample_rate: Sampling rate (Hz)
    
    Returns:
        Dictionary of extracted features
    """
    extractor = SeismicFeatureExtractor(sample_rate)
    features = {}
    
    if 'statistical' in feature_types:
        features['mean'] = np.mean(data)
        features['std'] = np.std(data)
        features['rms'] = np.sqrt(np.mean(data ** 2))
    
    if 'spectral' in feature_types:
        freqs, spectrum = extractor.compute_fft(data)
        features['spectrum'] = spectrum
        features['frequencies'] = freqs
    
    if 'temporal' in feature_types:
        features['envelope'] = extractor.compute_amplitude_envelope(data)
    
    return features
