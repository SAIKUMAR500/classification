import os
import json
import pickle
import base64
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from io import BytesIO

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, silhouette_score
)
from sklearn.preprocessing import StandardScaler
from pydantic import BaseModel, Field, ConfigDict

from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseLLM
from huggingface_hub import Repository, snapshot_download

class DataAnalysisAgent(BaseModel):
    """Enhanced version with full Hugging Face Hub support"""
    
    name: str = "Data Analysis Expert"
    current_dataset: Optional[pd.DataFrame] = None
    analysis_results: Dict[str, Any] = {}
    conversation_history: List[str] = []
    chain: Optional[LLMChain] = None
    output_dir: str = "analysis_output"
    scaler: Optional[StandardScaler] = None
    last_request_time: float = 0.0
    request_delay: float = 1.0
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def __init__(self, **data):
        super().__init__(**data)
        self._setup_output_directory()
    
    def _setup_output_directory(self):
        """Create or clean the output directory"""
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir)
    
    def load_dataset(self, data: pd.DataFrame) -> None:
        """Load a dataset into the agent"""
        self.current_dataset = data
        self.analysis_results = {}
        self.conversation_history.append("System: New dataset loaded")
        self.scaler = StandardScaler()
    
    # [Include all your existing methods here...]
    # generate_report, _format_table, _get_metric_formula, etc.
    # ...

    def save_pretrained(self, save_directory: str, push_to_hub: bool = False, **kwargs):
        """
        Save the agent to a directory for later use.
        
        Args:
            save_directory: Path to save the agent
            push_to_hub: Whether to push to Hugging Face Hub
            kwargs: Additional arguments for Repository
        """
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config = {
            "name": self.name,
            "output_dir": self.output_dir,
            "request_delay": self.request_delay,
        }
        with open(save_path / "config.json", "w") as f:
            json.dump(config, f)
            
        # Save dataset if exists
        if self.current_dataset is not None:
            self.current_dataset.to_parquet(save_path / "dataset.parquet")
            
        # Save analysis results
        if self.analysis_results:
            with open(save_path / "analysis_results.json", "w") as f:
                json.dump(self.analysis_results, f, default=str)
                
        # Save conversation history
        with open(save_path / "conversation_history.json", "w") as f:
            json.dump(self.conversation_history, f)
            
        # Save visualization files
        if hasattr(self, 'output_dir') and os.path.exists(self.output_dir):
            viz_dir = save_path / "visualizations"
            viz_dir.mkdir(exist_ok=True)
            for viz_file in os.listdir(self.output_dir):
                shutil.copy2(
                    os.path.join(self.output_dir, viz_file),
                    viz_dir / viz_file
                )
        
        if push_to_hub:
            repo = Repository(save_directory, **kwargs)
            repo.push_to_hub()

    @classmethod
    def from_pretrained(cls, save_directory: str, llm: Optional[BaseLLM] = None, **kwargs):
        """
        Load an agent from a directory.
        
        Args:
            save_directory: Path to load the agent from
            llm: Optional LLM to use (required if you want to use the chain)
        """
        save_path = Path(save_directory)
        
        # Load configuration
        with open(save_path / "config.json", "r") as f:
            config = json.load(f)
            
        # Initialize agent
        agent = cls(**config)
        
        # Load dataset if exists
        dataset_path = save_path / "dataset.parquet"
        if dataset_path.exists():
            agent.current_dataset = pd.read_parquet(dataset_path)
            
        # Load analysis results
        results_path = save_path / "analysis_results.json"
        if results_path.exists():
            with open(results_path, "r") as f:
                agent.analysis_results = json.load(f)
                
        # Load conversation history
        history_path = save_path / "conversation_history.json"
        if history_path.exists():
            with open(history_path, "r") as f:
                agent.conversation_history = json.load(f)
                
        # Load visualizations
        viz_dir = save_path / "visualizations"
        if viz_dir.exists():
            agent._setup_output_directory()
            for viz_file in os.listdir(viz_dir):
                shutil.copy2(
                    viz_dir / viz_file,
                    os.path.join(agent.output_dir, viz_file)
        
        # Reinitialize LLM chain if provided
        if llm is not None:
            agent.chain = create_analysis_agent(llm).chain
                
        return agent

def create_analysis_agent(llm: BaseLLM) -> DataAnalysisAgent:
    """Factory function to create a DataAnalysisAgent with LLM chain"""
    prompt = PromptTemplate(
        template="""[Your existing prompt template here...]""",
        input_variables=["dataset_info", "analysis_results", "conversation_history"]
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return DataAnalysisAgent(chain=chain)
