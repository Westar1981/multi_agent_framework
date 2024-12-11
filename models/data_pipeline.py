"""
Data pipeline for model training with specialized data loaders and preprocessing.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import json
import logging
from dataclasses import dataclass
from transformers import PreTrainedTokenizer
import dgl
from tqdm import tqdm
import pandas as pd
import ast
import re

from .model_integrator import ModelType

logger = logging.getLogger(__name__)

@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    batch_size: int
    max_length: int
    num_workers: int = 4
    shuffle: bool = True
    validation_split: float = 0.2
    cache_dir: Optional[str] = None

class CodeDataset(Dataset):
    """Dataset for code analysis and generation."""
    
    def __init__(self, data_path: str, tokenizer: PreTrainedTokenizer, config: DataConfig):
        self.tokenizer = tokenizer
        self.config = config
        self.samples = self._load_data(data_path)
        
    def _load_data(self, data_path: str) -> List[Dict[str, Any]]:
        """Load code samples from data path."""
        samples = []
        data_path = Path(data_path)
        
        # Load from jsonl file
        if data_path.suffix == '.jsonl':
            with open(data_path) as f:
                for line in tqdm(f, desc="Loading code samples"):
                    sample = json.loads(line)
                    if self._validate_sample(sample):
                        samples.append(sample)
                        
        # Load from directory of Python files
        elif data_path.is_dir():
            for file_path in tqdm(list(data_path.rglob('*.py')), desc="Loading Python files"):
                try:
                    with open(file_path) as f:
                        code = f.read()
                    tree = ast.parse(code)
                    functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
                    
                    for func in functions:
                        samples.append({
                            'code': ast.unparse(func),
                            'docstring': ast.get_docstring(func) or '',
                            'name': func.name,
                            'file': str(file_path)
                        })
                except Exception as e:
                    logger.warning(f"Error processing {file_path}: {str(e)}")
                    
        return samples
        
    def _validate_sample(self, sample: Dict[str, Any]) -> bool:
        """Validate code sample."""
        required_fields = {'code', 'docstring'}
        return all(field in sample for field in required_fields)
        
    def __len__(self) -> int:
        return len(self.samples)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Tokenize code and docstring
        code_tokens = self.tokenizer(
            sample['code'],
            padding='max_length',
            truncation=True,
            max_length=self.config.max_length,
            return_tensors='pt'
        )
        
        docstring_tokens = self.tokenizer(
            sample['docstring'],
            padding='max_length',
            truncation=True,
            max_length=self.config.max_length // 2,
            return_tensors='pt'
        )
        
        return {
            'code_ids': code_tokens['input_ids'].squeeze(0),
            'code_mask': code_tokens['attention_mask'].squeeze(0),
            'docstring_ids': docstring_tokens['input_ids'].squeeze(0),
            'docstring_mask': docstring_tokens['attention_mask'].squeeze(0)
        }

class KnowledgeGraphDataset(Dataset):
    """Dataset for knowledge graph learning."""
    
    def __init__(self, data_path: str, config: DataConfig):
        self.config = config
        self.graph = self._load_graph(data_path)
        self.node_features = self._prepare_node_features()
        self.edge_types = self._get_edge_types()
        
    def _load_graph(self, data_path: str) -> dgl.DGLGraph:
        """Load knowledge graph from data path."""
        data_path = Path(data_path)
        
        if data_path.suffix == '.json':
            with open(data_path) as f:
                data = json.load(f)
                
            # Create graph
            src_nodes = []
            dst_nodes = []
            edge_types = []
            
            for triple in data['triples']:
                src_nodes.append(triple['subject'])
                dst_nodes.append(triple['object'])
                edge_types.append(triple['relation'])
                
            return dgl.graph((src_nodes, dst_nodes))
            
        elif data_path.suffix == '.csv':
            df = pd.read_csv(data_path)
            src_nodes = df['subject'].values
            dst_nodes = df['object'].values
            edge_types = df['relation'].values
            
            return dgl.graph((src_nodes, dst_nodes))
            
        raise ValueError(f"Unsupported file format: {data_path.suffix}")
        
    def _prepare_node_features(self) -> torch.Tensor:
        """Prepare node feature vectors."""
        num_nodes = self.graph.number_of_nodes()
        return torch.randn(num_nodes, self.config.max_length)
        
    def _get_edge_types(self) -> List[str]:
        """Get unique edge types."""
        return sorted(list(set(self.graph.edata['type'].tolist())))
        
    def __len__(self) -> int:
        return self.graph.number_of_nodes()
        
    def __getitem__(self, idx: int) -> Tuple[dgl.DGLGraph, torch.Tensor]:
        # Sample subgraph around node
        subgraph = dgl.node_subgraph(self.graph, [idx])
        node_feats = self.node_features[idx]
        
        return subgraph, node_feats

class LanguageDataset(Dataset):
    """Dataset for language understanding."""
    
    def __init__(self, data_path: str, tokenizer: PreTrainedTokenizer, config: DataConfig):
        self.tokenizer = tokenizer
        self.config = config
        self.samples = self._load_data(data_path)
        
    def _load_data(self, data_path: str) -> List[Dict[str, Any]]:
        """Load text samples from data path."""
        samples = []
        data_path = Path(data_path)
        
        if data_path.suffix == '.jsonl':
            with open(data_path) as f:
                for line in f:
                    sample = json.loads(line)
                    if self._validate_sample(sample):
                        samples.append(sample)
                        
        elif data_path.suffix == '.txt':
            with open(data_path) as f:
                text = f.read()
                # Split into sentences
                sentences = re.split(r'[.!?]+', text)
                samples = [{'text': s.strip()} for s in sentences if s.strip()]
                
        return samples
        
    def _validate_sample(self, sample: Dict[str, Any]) -> bool:
        """Validate text sample."""
        return 'text' in sample
        
    def __len__(self) -> int:
        return len(self.samples)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        tokens = self.tokenizer(
            sample['text'],
            padding='max_length',
            truncation=True,
            max_length=self.config.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'labels': torch.tensor(sample.get('label', 0))
        }

class MetaLearningDataset(Dataset):
    """Dataset for meta-learning."""
    
    def __init__(self, data_path: str, config: DataConfig):
        self.config = config
        self.tasks = self._load_tasks(data_path)
        
    def _load_tasks(self, data_path: str) -> List[Dict[str, Any]]:
        """Load meta-learning tasks."""
        tasks = []
        data_path = Path(data_path)
        
        with open(data_path) as f:
            data = json.load(f)
            
        for task in data['tasks']:
            if self._validate_task(task):
                tasks.append({
                    'support_set': torch.tensor(task['support_set']),
                    'query_set': torch.tensor(task['query_set']),
                    'task_type': task['task_type']
                })
                
        return tasks
        
    def _validate_task(self, task: Dict[str, Any]) -> bool:
        """Validate meta-learning task."""
        required_fields = {'support_set', 'query_set', 'task_type'}
        return all(field in task for field in required_fields)
        
    def __len__(self) -> int:
        return len(self.tasks)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        task = self.tasks[idx]
        return {
            'support_set': task['support_set'],
            'query_set': task['query_set'],
            'task_type': torch.tensor(task['task_type'])
        }

def create_dataloaders(
    data_path: str,
    model_type: ModelType,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    config: Optional[DataConfig] = None
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Create train and validation data loaders."""
    
    if config is None:
        config = DataConfig(
            batch_size=32,
            max_length=512,
            num_workers=4
        )
        
    # Create dataset based on model type
    if model_type == ModelType.CODE:
        if tokenizer is None:
            raise ValueError("Tokenizer required for code dataset")
        dataset = CodeDataset(data_path, tokenizer, config)
        
    elif model_type == ModelType.KNOWLEDGE:
        dataset = KnowledgeGraphDataset(data_path, config)
        
    elif model_type == ModelType.LANGUAGE:
        if tokenizer is None:
            raise ValueError("Tokenizer required for language dataset")
        dataset = LanguageDataset(data_path, tokenizer, config)
        
    elif model_type == ModelType.META:
        dataset = MetaLearningDataset(data_path, config)
        
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
        
    # Split into train and validation
    val_size = int(len(dataset) * config.validation_split)
    train_size = len(dataset) - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    ) if val_size > 0 else None
    
    return train_loader, val_loader

def get_data_config(model_type: ModelType, config_path: Optional[str] = None) -> DataConfig:
    """Get data configuration for model type."""
    if config_path:
        with open(config_path) as f:
            config = json.load(f)
        return DataConfig(**config)
        
    # Default configurations
    defaults = {
        ModelType.CODE: DataConfig(
            batch_size=32,
            max_length=512,
            num_workers=4
        ),
        ModelType.KNOWLEDGE: DataConfig(
            batch_size=64,
            max_length=256,
            num_workers=4
        ),
        ModelType.LANGUAGE: DataConfig(
            batch_size=16,
            max_length=512,
            num_workers=4
        ),
        ModelType.META: DataConfig(
            batch_size=32,
            max_length=128,
            num_workers=4
        )
    }
    
    return defaults.get(model_type, DataConfig(
        batch_size=32,
        max_length=512,
        num_workers=4
    )) 