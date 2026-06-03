"""
CLI script for generating synthetic seismic data
"""

import argparse
import logging
from pathlib import Path

import numpy as np
from src.data.synthetic_generator import generate_synthetic_seismic_data
from src.utils import setup_logging

logger = setup_logging(__name__)


def main():
    """Generate synthetic seismic training data."""
    parser = argparse.ArgumentParser(description="Generate synthetic seismic data")
    
    parser.add_argument('--output', type=str, required=True,
                        help='Output file path')
    
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of training samples')
    
    parser.add_argument('--trace_length', type=int, default=2048,
                        help='Length of each trace')
    
    parser.add_argument('--sample_rate', type=float, default=1000.0,
                        help='Sample rate (Hz)')
    
    parser.add_argument('--snr', type=float, default=5.0,
                        help='Target SNR (dB)')
    
    parser.add_argument('--separate', action='store_true',
                        help='Save noisy and clean data separately')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Generating synthetic seismic data...")
    logger.info(f"  Number of samples: {args.num_samples}")
    logger.info(f"  Trace length: {args.trace_length}")
    logger.info(f"  Sample rate: {args.sample_rate} Hz")
    logger.info(f"  Target SNR: {args.snr} dB")
    
    # Generate data
    noisy_data, clean_data = generate_synthetic_seismic_data(
        n_samples=args.num_samples,
        trace_length=args.trace_length,
        sample_rate=args.sample_rate,
        seed=args.seed
    )
    
    logger.info(f"Generated data shape: {noisy_data.shape}")
    
    # Save data
    if args.separate:
        # Save noisy and clean data separately
        noisy_path = str(args.output).replace('.npy', '_noisy.npy')
        clean_path = str(args.output).replace('.npy', '_clean.npy')
        
        np.save(noisy_path, noisy_data)
        np.save(clean_path, clean_data)
        
        logger.info(f"Saved noisy data to {noisy_path}")
        logger.info(f"Saved clean data to {clean_path}")
    else:
        # Save combined data
        combined_data = np.vstack([noisy_data, clean_data])
        np.save(args.output, combined_data)
        
        logger.info(f"Saved combined data to {args.output}")
        logger.info(f"  Combined shape: {combined_data.shape}")
        logger.info(f"  First half: noisy data")
        logger.info(f"  Second half: clean data")
    
    logger.info("Data generation completed successfully!")


if __name__ == "__main__":
    main()
