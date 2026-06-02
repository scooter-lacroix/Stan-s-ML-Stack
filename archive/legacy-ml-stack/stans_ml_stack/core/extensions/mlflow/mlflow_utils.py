#!/usr/bin/env python3
# =============================================================================
# MLflow Utilities
# =============================================================================
# This module provides utilities for using MLflow with AMD GPUs.
#
# Author: Stanley Chisango (Scooter Lacroix)
# Email: scooterlacroix@gmail.com
# GitHub: https://github.com/scooter-lacroix
# X: https://x.com/scooter_lacroix
# Patreon: https://patreon.com/ScooterLacroix
# 
# If this code saved you time, consider buying me a coffee! ☕
# "Code is like humor. When you have to explain it, it's bad!" - Cory House
# Date: 2023-04-19
# =============================================================================

import os
import sys
import torch
import logging
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("mlflow_utils")

def check_mlflow_installation():
    """Check if MLflow is installed.
    
    Returns:
        bool: True if MLflow is installed, False otherwise
    """
    try:
        import mlflow
        logger.info(f"MLflow is installed (version {mlflow.__version__})")
        return True
    except ImportError:
        logger.error("MLflow is not installed")
        logger.info("Please install MLflow first")
        return False

def install_mlflow():
    """Install MLflow.
    
    Returns:
        bool: True if installation is successful, False otherwise
    """
    try:
        import subprocess
        
        # Install MLflow
        logger.info("Installing MLflow")
        subprocess.run(
            ["pip", "install", "mlflow"],
            check=True
        )
        
        logger.info("MLflow installed successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to install MLflow: {e}")
        return False

def start_mlflow_server(host="127.0.0.1", port=5000, backend_store_uri=None, default_artifact_root=None):
    """Start MLflow server.
    
    Args:
        host: Host
        port: Port
        backend_store_uri: Backend store URI
        default_artifact_root: Default artifact root
    
    Returns:
        subprocess.Popen: Server process
    """
    try:
        import subprocess
        
        # Set command
        command = ["mlflow", "server", "--host", host, "--port", str(port)]
        
        # Add backend store URI if provided
        if backend_store_uri:
            command.extend(["--backend-store-uri", backend_store_uri])
        
        # Add default artifact root if provided
        if default_artifact_root:
            command.extend(["--default-artifact-root", default_artifact_root])
        
        # Start server
        logger.info(f"Starting MLflow server on {host}:{port}")
        
        server_process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Wait for server to start
        time.sleep(5)
        
        logger.info(f"MLflow server started successfully")
        return server_process
    except Exception as e:
        logger.error(f"Failed to start MLflow server: {e}")
        return None

def set_mlflow_tracking_uri(tracking_uri):
    """Set MLflow tracking URI.
    
    Args:
        tracking_uri: Tracking URI
    
    Returns:
        bool: True if tracking URI is set, False otherwise
    """
    try:
        import mlflow
        
        # Set tracking URI
        logger.info(f"Setting MLflow tracking URI: {tracking_uri}")
        
        mlflow.set_tracking_uri(tracking_uri)
        
        logger.info(f"MLflow tracking URI set successfully")
        return True
    except ImportError:
        logger.error("MLflow is not installed")
        return False
    except Exception as e:
        logger.error(f"Failed to set MLflow tracking URI: {e}")
        return False

def create_mlflow_experiment(experiment_name, artifact_location=None):
    """Create MLflow experiment.
    
    Args:
        experiment_name: Experiment name
        artifact_location: Artifact location
    
    Returns:
        str: Experiment ID
    """
    try:
        import mlflow
        
        # Create experiment
        logger.info(f"Creating MLflow experiment: {experiment_name}")
        
        # Check if experiment exists
        experiment = mlflow.get_experiment_by_name(experiment_name)
        
        if experiment:
            experiment_id = experiment.experiment_id
            logger.info(f"Experiment already exists with ID: {experiment_id}")
        else:
            experiment_id = mlflow.create_experiment(
                name=experiment_name,
                artifact_location=artifact_location
            )
            logger.info(f"Experiment created with ID: {experiment_id}")
        
        return experiment_id
    except ImportError:
        logger.error("MLflow is not installed")
        return None
    except Exception as e:
        logger.error(f"Failed to create MLflow experiment: {e}")
        return None

def start_mlflow_run(experiment_name=None, run_name=None, tags=None):
    """Start MLflow run.
    
    Args:
        experiment_name: Experiment name
        run_name: Run name
        tags: Tags
    
    Returns:
        mlflow.ActiveRun: MLflow run
    """
    try:
        import mlflow
        
        # Set experiment if provided
        if experiment_name:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            
            if experiment:
                experiment_id = experiment.experiment_id
            else:
                experiment_id = create_mlflow_experiment(experiment_name)
            
            mlflow.set_experiment(experiment_id)
        
        # Start run
        logger.info(f"Starting MLflow run")
        
        run = mlflow.start_run(run_name=run_name, tags=tags)
        
        logger.info(f"MLflow run started successfully: {run.info.run_id}")
        return run
    except ImportError:
        logger.error("MLflow is not installed")
        return None
    except Exception as e:
        logger.error(f"Failed to start MLflow run: {e}")
        return None

def log_mlflow_params(params):
    """Log MLflow parameters.
    
    Args:
        params: Parameters
    
    Returns:
        bool: True if parameters are logged, False otherwise
    """
    try:
        import mlflow
        
        # Log parameters
        logger.info(f"Logging MLflow parameters")
        
        mlflow.log_params(params)
        
        logger.info(f"MLflow parameters logged successfully")
        return True
    except ImportError:
        logger.error("MLflow is not installed")
        return False
    except Exception as e:
        logger.error(f"Failed to log MLflow parameters: {e}")
        return False

def log_mlflow_metrics(metrics, step=None):
    """Log MLflow metrics.
    
    Args:
        metrics: Metrics
        step: Step
    
    Returns:
        bool: True if metrics are logged, False otherwise
    """
    try:
        import mlflow
        
        # Log metrics
        logger.info(f"Logging MLflow metrics")
        
        mlflow.log_metrics(metrics, step=step)
        
        logger.info(f"MLflow metrics logged successfully")
        return True
    except ImportError:
        logger.error("MLflow is not installed")
        return False
    except Exception as e:
        logger.error(f"Failed to log MLflow metrics: {e}")
        return False

def log_mlflow_artifact(artifact_path):
    """Log MLflow artifact.
    
    Args:
        artifact_path: Artifact path
    
    Returns:
        bool: True if artifact is logged, False otherwise
    """
    try:
        import mlflow
        
        # Log artifact
        logger.info(f"Logging MLflow artifact: {artifact_path}")
        
        mlflow.log_artifact(artifact_path)
        
        logger.info(f"MLflow artifact logged successfully")
        return True
    except ImportError:
        logger.error("MLflow is not installed")
        return False
    except Exception as e:
        logger.error(f"Failed to log MLflow artifact: {e}")
        return False

def log_mlflow_figure(figure, artifact_path):
    """Log MLflow figure.
    
    Args:
        figure: Figure
        artifact_path: Artifact path
    
    Returns:
        bool: True if figure is logged, False otherwise
    """
    try:
        import mlflow
        
        # Log figure
        logger.info(f"Logging MLflow figure: {artifact_path}")
        
        mlflow.log_figure(figure, artifact_path)
        
        logger.info(f"MLflow figure logged successfully")
        return True
    except ImportError:
        logger.error("MLflow is not installed")
        return False
    except Exception as e:
        logger.error(f"Failed to log MLflow figure: {e}")
        return False

def log_mlflow_model(model, artifact_path, conda_env=None, signature=None, input_example=None):
    """Log MLflow model.
    
    Args:
        model: Model
        artifact_path: Artifact path
        conda_env: Conda environment
        signature: Model signature
        input_example: Input example
    
    Returns:
        bool: True if model is logged, False otherwise
    """
    try:
        import mlflow
        
        # Log model
        logger.info(f"Logging MLflow model: {artifact_path}")
        
        mlflow.pytorch.log_model(
            model,
            artifact_path,
            conda_env=conda_env,
            signature=signature,
            input_example=input_example
        )
        
        logger.info(f"MLflow model logged successfully")
        return True
    except ImportError:
        logger.error("MLflow is not installed")
        return False
    except Exception as e:
        logger.error(f"Failed to log MLflow model: {e}")
        return False

def end_mlflow_run():
    """End MLflow run.
    
    Returns:
        bool: True if run is ended, False otherwise
    """
    try:
        import mlflow
        
        # End run
        logger.info(f"Ending MLflow run")
        
        mlflow.end_run()
        
        logger.info(f"MLflow run ended successfully")
        return True
    except ImportError:
        logger.error("MLflow is not installed")
        return False
    except Exception as e:
        logger.error(f"Failed to end MLflow run: {e}")
        return False

def load_mlflow_model(model_uri):
    """Load MLflow model.
    
    Args:
        model_uri: Model URI
    
    Returns:
        torch.nn.Module: PyTorch model
    """
    try:
        import mlflow
        
        # Load model
        logger.info(f"Loading MLflow model: {model_uri}")
        
        model = mlflow.pytorch.load_model(model_uri)
        
        logger.info(f"MLflow model loaded successfully")
        return model
    except ImportError:
        logger.error("MLflow is not installed")
        return None
    except Exception as e:
        logger.error(f"Failed to load MLflow model: {e}")
        return None

def get_mlflow_run(run_id):
    """Get MLflow run.
    
    Args:
        run_id: Run ID
    
    Returns:
        mlflow.entities.Run: MLflow run
    """
    try:
        import mlflow
        
        # Get run
        logger.info(f"Getting MLflow run: {run_id}")
        
        run = mlflow.get_run(run_id)
        
        logger.info(f"MLflow run retrieved successfully")
        return run
    except ImportError:
        logger.error("MLflow is not installed")
        return None
    except Exception as e:
        logger.error(f"Failed to get MLflow run: {e}")
        return None

def search_mlflow_runs(experiment_ids=None, filter_string=None, max_results=100):
    """Search MLflow runs.
    
    Args:
        experiment_ids: Experiment IDs
        filter_string: Filter string
        max_results: Maximum number of results
    
    Returns:
        list: List of MLflow runs
    """
    try:
        import mlflow
        from mlflow.entities import ViewType
        
        # Search runs
        logger.info(f"Searching MLflow runs")
        
        runs = mlflow.search_runs(
            experiment_ids=experiment_ids,
            filter_string=filter_string,
            max_results=max_results,
            view_type=ViewType.ACTIVE_ONLY
        )
        
        logger.info(f"MLflow runs search completed successfully: {len(runs)} runs found")
        return runs
    except ImportError:
        logger.error("MLflow is not installed")
        return None
    except Exception as e:
        logger.error(f"Failed to search MLflow runs: {e}")
        return None

def delete_mlflow_experiment(experiment_id):
    """Delete MLflow experiment.
    
    Args:
        experiment_id: Experiment ID
    
    Returns:
        bool: True if experiment is deleted, False otherwise
    """
    try:
        import mlflow
        
        # Delete experiment
        logger.info(f"Deleting MLflow experiment: {experiment_id}")
        
        mlflow.delete_experiment(experiment_id)
        
        logger.info(f"MLflow experiment deleted successfully")
        return True
    except ImportError:
        logger.error("MLflow is not installed")
        return False
    except Exception as e:
        logger.error(f"Failed to delete MLflow experiment: {e}")
        return False

def log_gpu_metrics_to_mlflow(step=None):
    """Log GPU metrics to MLflow.
    
    Args:
        step: Step
    
    Returns:
        bool: True if metrics are logged, False otherwise
    """
    try:
        import mlflow
        import subprocess
        import json
        
        # Get GPU metrics
        result = subprocess.run(
            ["rocm-smi", "--showuse", "--showmeminfo", "vram", "--showtemp", "--showpower", "--json"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        if result.returncode != 0:
            logger.error("Failed to get GPU metrics")
            logger.error(result.stderr)
            return False
        
        # Parse JSON output
        gpu_metrics = json.loads(result.stdout)
        
        # Log GPU metrics
        logger.info(f"Logging GPU metrics to MLflow")
        
        metrics = {}
        
        for i, gpu in enumerate(gpu_metrics):
            gpu_use = gpu.get("GPU use (%)", 0)
            memory_use = gpu.get("GPU memory use (%)", 0)
            temperature = gpu.get("Temperature", {}).get("edge", 0)
            power = gpu.get("Power", {}).get("average", 0)
            
            metrics[f"gpu_{i}_use"] = gpu_use
            metrics[f"gpu_{i}_memory_use"] = memory_use
            metrics[f"gpu_{i}_temperature"] = temperature
            metrics[f"gpu_{i}_power"] = power
        
        mlflow.log_metrics(metrics, step=step)
        
        logger.info(f"GPU metrics logged successfully")
        return True
    except ImportError:
        logger.error("MLflow is not installed")
        return False
    except Exception as e:
        logger.error(f"Failed to log GPU metrics: {e}")
        return False

def create_mlflow_experiment_with_tags(experiment_name, tags=None, artifact_location=None):
    """Create MLflow experiment with tags.
    
    Args:
        experiment_name: Experiment name
        tags: Tags
        artifact_location: Artifact location
    
    Returns:
        str: Experiment ID
    """
    try:
        import mlflow
        
        # Create experiment
        logger.info(f"Creating MLflow experiment with tags: {experiment_name}")
        
        # Check if experiment exists
        experiment = mlflow.get_experiment_by_name(experiment_name)
        
        if experiment:
            experiment_id = experiment.experiment_id
            logger.info(f"Experiment already exists with ID: {experiment_id}")
        else:
            experiment_id = mlflow.create_experiment(
                name=experiment_name,
                artifact_location=artifact_location,
                tags=tags
            )
            logger.info(f"Experiment created with ID: {experiment_id}")
        
        return experiment_id
    except ImportError:
        logger.error("MLflow is not installed")
        return None
    except Exception as e:
        logger.error(f"Failed to create MLflow experiment with tags: {e}")
        return None

def log_mlflow_confusion_matrix(y_true, y_pred, labels=None):
    """Log confusion matrix to MLflow.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Label names
    
    Returns:
        bool: True if confusion matrix is logged, False otherwise
    """
    try:
        import mlflow
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(ax=ax)
        
        # Log figure
        mlflow.log_figure(fig, "confusion_matrix.png")
        
        # Close figure
        plt.close(fig)
        
        logger.info(f"Confusion matrix logged successfully")
        return True
    except ImportError:
        logger.error("MLflow, matplotlib, or scikit-learn is not installed")
        return False
    except Exception as e:
        logger.error(f"Failed to log confusion matrix: {e}")
        return False

def log_mlflow_roc_curve(y_true, y_score):
    """Log ROC curve to MLflow.
    
    Args:
        y_true: True labels
        y_score: Predicted scores
    
    Returns:
        bool: True if ROC curve is logged, False otherwise
    """
    try:
        import mlflow
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, auc
        
        # Compute ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
        ax.plot([0, 1], [0, 1], "k--")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Receiver Operating Characteristic")
        ax.legend(loc="lower right")
        
        # Log figure
        mlflow.log_figure(fig, "roc_curve.png")
        
        # Close figure
        plt.close(fig)
        
        logger.info(f"ROC curve logged successfully")
        return True
    except ImportError:
        logger.error("MLflow, matplotlib, or scikit-learn is not installed")
        return False
    except Exception as e:
        logger.error(f"Failed to log ROC curve: {e}")
        return False

if __name__ == "__main__":
    # Check MLflow installation
    check_mlflow_installation()
    
    # Example usage
    if check_mlflow_installation():
        # Set tracking URI
        set_mlflow_tracking_uri("http://localhost:5000")
        
        # Create experiment
        experiment_id = create_mlflow_experiment("amd_gpu_test")
        
        # Start run
        run = start_mlflow_run(run_name="test_run")
        
        # Log parameters
        log_mlflow_params({
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 10
        })
        
        # Log metrics
        for epoch in range(10):
            log_mlflow_metrics({
                "train_loss": 1.0 - 0.1 * epoch,
                "train_accuracy": 0.5 + 0.05 * epoch,
                "val_loss": 1.2 - 0.1 * epoch,
                "val_accuracy": 0.4 + 0.05 * epoch
            }, step=epoch)
        
        # Log GPU metrics
        log_gpu_metrics_to_mlflow()
        
        # End run
        end_mlflow_run()
