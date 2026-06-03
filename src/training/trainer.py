"""
Model trainer for seismic denoising.
Handles training loop, validation, and model checkpointing.
"""

import logging
import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class Trainer:
    """Train denoising models."""
    
    def __init__(
        self,
        model: keras.Model,
        config: Optional[Dict[str, Any]] = None,
        checkpoint_dir: str = "data/models/",
        log_dir: str = "logs/"
    ):
        """
        Initialize trainer.
        
        Args:
            model: Keras model to train
            config: Configuration dictionary
            checkpoint_dir: Directory for saving checkpoints
            log_dir: Directory for logs
        """
        self.model = model
        self.config = config or {}
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        
        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.history = None
        self.best_model_path = None
    
    def compile(
        self,
        loss: str = 'mse',
        optimizer: str = 'adam',
        learning_rate: float = 0.001,
        metrics: list = None
    ):
        """
        Compile model.
        
        Args:
            loss: Loss function
            optimizer: Optimizer
            learning_rate: Learning rate
            metrics: Evaluation metrics
        """
        if optimizer.lower() == 'adam':
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer.lower() == 'sgd':
            opt = keras.optimizers.SGD(learning_rate=learning_rate)
        else:
            opt = optimizer
        
        if metrics is None:
            metrics = ['mae']
        
        self.model.compile(
            optimizer=opt,
            loss=loss,
            metrics=metrics
        )
        
        logger.info(f"Model compiled with {optimizer} optimizer, {loss} loss")
    
    def train(
        self,
        train_data: Tuple[np.ndarray, np.ndarray],
        val_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: int = 1,
        early_stopping_patience: int = 10,
        reduce_lr_patience: int = 5
    ) -> Dict[str, Any]:
        """
        Train model.
        
        Args:
            train_data: Tuple of (X_train, y_train)
            val_data: Tuple of (X_val, y_val) or None
            epochs: Number of training epochs
            batch_size: Batch size
            verbose: Verbosity level
            early_stopping_patience: Early stopping patience
            reduce_lr_patience: Learning rate reduction patience
        
        Returns:
            Training history
        """
        X_train, y_train = train_data
        
        callbacks = []
        
        # Checkpoint callback
        checkpoint_path = self.checkpoint_dir / "best_model.h5"
        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            str(checkpoint_path),
            monitor='val_loss' if val_data else 'loss',
            save_best_only=True,
            verbose=verbose
        )
        callbacks.append(checkpoint_callback)
        self.best_model_path = checkpoint_path
        
        # Early stopping callback
        early_stopping_callback = keras.callbacks.EarlyStopping(
            monitor='val_loss' if val_data else 'loss',
            patience=early_stopping_patience,
            verbose=verbose,
            restore_best_weights=True
        )
        callbacks.append(early_stopping_callback)
        
        # Learning rate reduction
        reduce_lr_callback = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss' if val_data else 'loss',
            factor=0.5,
            patience=reduce_lr_patience,
            verbose=verbose,
            min_lr=1e-7
        )
        callbacks.append(reduce_lr_callback)
        
        # TensorBoard callback
        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir=str(self.log_dir / datetime.now().strftime("%Y%m%d-%H%M%S")),
            histogram_freq=1
        )
        callbacks.append(tensorboard_callback)
        
        # Train model
        logger.info(f"Starting training for {epochs} epochs")
        logger.info(f"Training samples: {len(X_train)}, Batch size: {batch_size}")
        if val_data:
            logger.info(f"Validation samples: {len(val_data[0])}")
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=val_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        logger.info("Training completed")
        self.save_history()
        
        return self.history.history
    
    def evaluate(
        self,
        test_data: Tuple[np.ndarray, np.ndarray],
        batch_size: int = 32
    ) -> Dict[str, float]:
        """
        Evaluate model on test data.
        
        Args:
            test_data: Tuple of (X_test, y_test)
            batch_size: Batch size
        
        Returns:
            Dictionary of evaluation metrics
        """
        X_test, y_test = test_data
        
        logger.info(f"Evaluating model on {len(X_test)} test samples")
        
        results = self.model.evaluate(
            X_test, y_test,
            batch_size=batch_size,
            verbose=1
        )
        
        # Get metric names
        metric_names = ['loss'] + self.model.metrics_names
        
        eval_dict = dict(zip(metric_names, results if isinstance(results, list) else [results]))
        
        logger.info(f"Evaluation results: {eval_dict}")
        
        return eval_dict
    
    def predict(
        self,
        data: np.ndarray,
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            data: Input data
            batch_size: Batch size
        
        Returns:
            Predictions
        """
        return self.model.predict(data, batch_size=batch_size, verbose=0)
    
    def save_model(self, path: Optional[str] = None) -> None:
        """
        Save trained model.
        
        Args:
            path: Output path (uses best model if None)
        """
        path = path or str(self.checkpoint_dir / "model.h5")
        self.model.save(path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """
        Load trained model.
        
        Args:
            path: Path to model file
        """
        self.model = keras.models.load_model(path)
        logger.info(f"Model loaded from {path}")
    
    def save_history(self, path: Optional[str] = None) -> None:
        """
        Save training history.
        
        Args:
            path: Output path
        """
        if self.history is None:
            logger.warning("No training history to save")
            return
        
        path = path or str(self.log_dir / "training_history.json")
        
        with open(path, 'w') as f:
            json.dump(self.history.history, f, indent=2)
        
        logger.info(f"Training history saved to {path}")
    
    def get_summary(self) -> str:
        """Get model summary."""
        summary_lines = []
        self.model.summary(print_fn=lambda x: summary_lines.append(x))
        return "\n".join(summary_lines)
