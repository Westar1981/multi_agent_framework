"""
Model training infrastructure with support for different model types.
"""

import torch
from torch.utils.data import DataLoader
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from tqdm import tqdm
import json
import wandb
from dataclasses import dataclass

from .model_integrator import ModelType, ModelIntegrator
from .data_pipeline import create_dataloaders, get_data_config, DataConfig

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Training configuration."""
    num_epochs: int
    learning_rate: float
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 1e-4
    use_wandb: bool = False
    wandb_project: str = "multi_agent_framework"
    save_dir: Optional[str] = None

class ModelTrainer:
    """Trainer for different model types."""
    
    def __init__(
        self,
        model_type: ModelType,
        training_config: Optional[TrainingConfig] = None,
        data_config: Optional[DataConfig] = None
    ):
        self.model_type = model_type
        self.training_config = training_config or self._get_default_training_config()
        self.data_config = data_config or get_data_config(model_type)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def _get_default_training_config(self) -> TrainingConfig:
        """Get default training configuration."""
        return TrainingConfig(
            num_epochs=10,
            learning_rate=2e-5,
            weight_decay=0.01,
            warmup_steps=1000,
            max_grad_norm=1.0
        )
        
    def train(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ) -> Dict[str, Any]:
        """Train the model."""
        
        if optimizer is None:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.training_config.learning_rate,
                weight_decay=self.training_config.weight_decay
            )
            
        if scheduler is None:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.training_config.learning_rate,
                epochs=self.training_config.num_epochs,
                steps_per_epoch=len(train_loader)
            )
            
        # Initialize wandb if enabled
        if self.training_config.use_wandb:
            wandb.init(
                project=self.training_config.wandb_project,
                config={
                    "model_type": self.model_type.value,
                    "training_config": self.training_config.__dict__,
                    "data_config": self.data_config.__dict__
                }
            )
            
        model = model.to(self.device)
        best_val_loss = float('inf')
        patience_counter = 0
        training_history = {
            'train_loss': [],
            'val_loss': [],
            'best_model_state': None
        }
        
        for epoch in range(self.training_config.num_epochs):
            # Training
            model.train()
            train_loss = 0
            train_steps = 0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
            for batch in progress_bar:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = model(**batch)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    self.training_config.max_grad_norm
                )
                optimizer.step()
                scheduler.step()
                
                # Update metrics
                train_loss += loss.item()
                train_steps += 1
                progress_bar.set_postfix({'loss': loss.item()})
                
                if self.training_config.use_wandb:
                    wandb.log({
                        'train_loss': loss.item(),
                        'learning_rate': scheduler.get_last_lr()[0]
                    })
                    
            avg_train_loss = train_loss / train_steps
            training_history['train_loss'].append(avg_train_loss)
            
            # Validation
            if val_loader is not None:
                model.eval()
                val_loss = 0
                val_steps = 0
                
                with torch.no_grad():
                    for batch in tqdm(val_loader, desc="Validation"):
                        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                                for k, v in batch.items()}
                        outputs = model(**batch)
                        loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                        val_loss += loss.item()
                        val_steps += 1
                        
                avg_val_loss = val_loss / val_steps
                training_history['val_loss'].append(avg_val_loss)
                
                if self.training_config.use_wandb:
                    wandb.log({'val_loss': avg_val_loss})
                    
                # Early stopping check
                if avg_val_loss < best_val_loss - self.training_config.early_stopping_threshold:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    training_history['best_model_state'] = model.state_dict()
                    
                    # Save best model if save_dir is specified
                    if self.training_config.save_dir:
                        save_path = Path(self.training_config.save_dir)
                        save_path.mkdir(parents=True, exist_ok=True)
                        torch.save(model.state_dict(), save_path / 'best_model.pt')
                else:
                    patience_counter += 1
                    
                # Early stopping
                if patience_counter >= self.training_config.early_stopping_patience:
                    logger.info("Early stopping triggered")
                    break
                    
            logger.info(
                f"Epoch {epoch + 1}: "
                f"train_loss={avg_train_loss:.4f}, "
                f"val_loss={avg_val_loss:.4f if val_loader else 'N/A'}"
            )
            
        if self.training_config.use_wandb:
            wandb.finish()
            
        return training_history
        
    @classmethod
    async def from_config(
        cls,
        config_path: str,
        model_type: ModelType
    ) -> 'ModelTrainer':
        """Create trainer from configuration file."""
        with open(config_path) as f:
            config = json.load(f)
            
        training_config = TrainingConfig(**config['training'])
        data_config = DataConfig(**config['data'])
        
        return cls(
            model_type=model_type,
                    training_config=training_config,
            data_config=data_config
        )
        
async def train_model(
    model_type: ModelType,
    data_path: str,
    config_path: Optional[str] = None,
    **kwargs
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """High-level function to train a model."""
    
    # Initialize trainer and integrator
    trainer = await ModelTrainer.from_config(config_path, model_type) if config_path \
        else ModelTrainer(model_type)
    integrator = await ModelIntegrator.create(model_type)
    
    # Create data loaders
    train_loader, val_loader = create_dataloaders(
        data_path,
        model_type,
        tokenizer=integrator.tokenizer,
        config=trainer.data_config
    )
    
    # Train model
    history = trainer.train(
        model=integrator.model,
        train_loader=train_loader,
        val_loader=val_loader,
        **kwargs
    )
    
    return integrator.model, history 