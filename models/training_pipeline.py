"""
Training pipeline for model training and evaluation.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Any, Optional, Callable, Tuple
import numpy as np
from pathlib import Path
import logging
import json
from dataclasses import dataclass
from datetime import datetime
import asyncio
from tqdm import tqdm

from .model_integrator import ModelWrapper, TrainingConfig, ModelType

logger = logging.getLogger(__name__)

@dataclass
class TrainingMetrics:
    """Metrics from training."""
    loss: float
    accuracy: float
    epoch: int
    step: int
    learning_rate: float
    timestamp: datetime
    additional_metrics: Dict[str, float] = None

class EarlyStopping:
    """Early stopping handler."""
    
    def __init__(self, patience: int, threshold: float):
        self.patience = patience
        self.threshold = threshold
        self.best_loss = float('inf')
        self.counter = 0
        self.should_stop = False
        
    def __call__(self, loss: float) -> bool:
        """Update early stopping state."""
        if loss < self.best_loss - self.threshold:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.should_stop = True
            
        return self.should_stop

class MetricsTracker:
    """Tracks and logs training metrics."""
    
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics: List[TrainingMetrics] = []
        
    def add_metrics(self, metrics: TrainingMetrics):
        """Add new metrics."""
        self.metrics.append(metrics)
        self._save_metrics(metrics)
        
    def _save_metrics(self, metrics: TrainingMetrics):
        """Save metrics to file."""
        metrics_file = self.log_dir / 'metrics.jsonl'
        with open(metrics_file, 'a') as f:
            json.dump({
                'loss': metrics.loss,
                'accuracy': metrics.accuracy,
                'epoch': metrics.epoch,
                'step': metrics.step,
                'learning_rate': metrics.learning_rate,
                'timestamp': metrics.timestamp.isoformat(),
                **(metrics.additional_metrics or {})
            }, f)
            f.write('\n')
            
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of tracked metrics."""
        if not self.metrics:
            return {}
            
        recent = self.metrics[-100:]
        return {
            'latest_loss': recent[-1].loss,
            'avg_loss': np.mean([m.loss for m in recent]),
            'latest_accuracy': recent[-1].accuracy,
            'avg_accuracy': np.mean([m.accuracy for m in recent]),
            'total_steps': sum(1 for m in self.metrics),
            'latest_learning_rate': recent[-1].learning_rate
        }

class ModelTrainer:
    """Handles model training and evaluation."""
    
    def __init__(self, 
                 model_wrapper: ModelWrapper,
                 training_config: TrainingConfig,
                 train_dataloader: DataLoader,
                 val_dataloader: Optional[DataLoader] = None,
                 metrics_tracker: Optional[MetricsTracker] = None):
        
        self.model_wrapper = model_wrapper
        self.config = training_config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.metrics_tracker = metrics_tracker or MetricsTracker('logs/training')
        
        self.early_stopping = EarlyStopping(
            patience=training_config.early_stopping_patience,
            threshold=training_config.early_stopping_threshold
        )
        
        self.device = model_wrapper.config.device
        
    async def train(self) -> Dict[str, Any]:
        """Train the model."""
        model = self.model_wrapper.model
        model.train()
        
        for epoch in range(self.config.num_epochs):
            epoch_metrics = await self._train_epoch(epoch)
            
            if self.val_dataloader:
                val_metrics = await self._evaluate()
                
                if self.early_stopping(val_metrics.loss):
                    logger.info("Early stopping triggered")
                    break
                    
            if epoch % self.config.save_steps == 0:
                await self.model_wrapper.save()
                
        return self.metrics_tracker.get_summary()
        
    async def _train_epoch(self, epoch: int) -> TrainingMetrics:
        """Train for one epoch."""
        total_loss = 0
        correct = 0
        total = 0
        
        progress = tqdm(self.train_dataloader, desc=f"Epoch {epoch}")
        for step, batch in enumerate(progress):
            loss, accuracy = await self._train_step(batch)
            
            total_loss += loss
            correct += accuracy * len(batch[0])
            total += len(batch[0])
            
            if step % self.config.evaluation_steps == 0:
                metrics = TrainingMetrics(
                    loss=total_loss / (step + 1),
                    accuracy=correct / total,
                    epoch=epoch,
                    step=step,
                    learning_rate=self.model_wrapper.scheduler.get_last_lr()[0],
                    timestamp=datetime.now()
                )
                self.metrics_tracker.add_metrics(metrics)
                
            progress.set_postfix({
                'loss': total_loss / (step + 1),
                'accuracy': correct / total
            })
            
        return TrainingMetrics(
            loss=total_loss / len(self.train_dataloader),
            accuracy=correct / total,
            epoch=epoch,
            step=len(self.train_dataloader),
            learning_rate=self.model_wrapper.scheduler.get_last_lr()[0],
            timestamp=datetime.now()
        )
        
    async def _train_step(self, batch: Tuple) -> Tuple[float, float]:
        """Perform one training step."""
        inputs, labels = self._prepare_batch(batch)
        
        self.model_wrapper.optimizer.zero_grad()
        
        outputs = self.model_wrapper.model(**inputs)
        loss = outputs.loss
        
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(
            self.model_wrapper.model.parameters(),
            self.config.max_grad_norm
        )
        
        self.model_wrapper.optimizer.step()
        self.model_wrapper.scheduler.step()
        
        # Calculate accuracy
        predictions = outputs.logits.argmax(-1)
        accuracy = (predictions == labels).float().mean().item()
        
        return loss.item(), accuracy
        
    async def _evaluate(self) -> TrainingMetrics:
        """Evaluate the model."""
        model = self.model_wrapper.model
        model.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Evaluating"):
                inputs, labels = self._prepare_batch(batch)
                outputs = model(**inputs)
                
                total_loss += outputs.loss.item()
                predictions = outputs.logits.argmax(-1)
                correct += (predictions == labels).sum().item()
                total += len(labels)
                
        model.train()
        
        return TrainingMetrics(
            loss=total_loss / len(self.val_dataloader),
            accuracy=correct / total,
            epoch=-1,
            step=-1,
            learning_rate=-1,
            timestamp=datetime.now()
        )
        
    def _prepare_batch(self, batch: Tuple) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Prepare batch for training/evaluation."""
        if self.model_wrapper.config.type == ModelType.CODE:
            inputs = self.model_wrapper.tokenizer(
                batch[0],
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
        elif self.model_wrapper.config.type == ModelType.LANGUAGE:
            inputs = self.model_wrapper.tokenizer(
                batch[0],
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
        else:
            inputs = {k: v.to(self.device) for k, v in batch[0].items()}
            
        labels = batch[1].to(self.device)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        return inputs, labels

class TrainingOrchestrator:
    """Orchestrates training across multiple models."""
    
    def __init__(self, save_dir: str = 'models'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.active_trainers: Dict[str, ModelTrainer] = {}
        
    def create_trainer(self,
                      model_wrapper: ModelWrapper,
                      training_config: TrainingConfig,
                      train_dataloader: DataLoader,
                      val_dataloader: Optional[DataLoader] = None) -> ModelTrainer:
        """Create a new trainer."""
        trainer = ModelTrainer(
            model_wrapper=model_wrapper,
            training_config=training_config,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            metrics_tracker=MetricsTracker(
                self.save_dir / model_wrapper.config.name / 'logs'
            )
        )
        
        self.active_trainers[model_wrapper.config.name] = trainer
        return trainer
        
    async def train_all(self) -> Dict[str, Any]:
        """Train all active models."""
        results = {}
        for name, trainer in self.active_trainers.items():
            logger.info(f"Training model: {name}")
            try:
                results[name] = await trainer.train()
            except Exception as e:
                logger.error(f"Error training model {name}: {str(e)}")
                results[name] = {'error': str(e)}
                
        return results
        
    def get_trainer(self, model_name: str) -> Optional[ModelTrainer]:
        """Get trainer by model name."""
        return self.active_trainers.get(model_name)
        
    def remove_trainer(self, model_name: str):
        """Remove a trainer."""
        if model_name in self.active_trainers:
            del self.active_trainers[model_name] 