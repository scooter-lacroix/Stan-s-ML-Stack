#!/usr/bin/env python3
# =============================================================================
# RAPIDS for AMD GPUs
# =============================================================================
# This module provides utilities for using RAPIDS with AMD GPUs.
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
import logging
import subprocess
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("rapids_amd")

def check_rapids_installation():
    """Check if RAPIDS is installed.
    
    Returns:
        bool: True if RAPIDS is installed, False otherwise
    """
    try:
        import cudf
        import cuml
        import cugraph
        
        logger.info(f"RAPIDS is installed (cuDF version {cudf.__version__})")
        logger.info(f"cuML version: {cuml.__version__}")
        logger.info(f"cuGraph version: {cugraph.__version__}")
        
        return True
    except ImportError:
        logger.error("RAPIDS is not installed")
        logger.info("Please install RAPIDS first")
        return False

def install_rapids_for_amd():
    """Install RAPIDS for AMD GPUs.
    
    Returns:
        bool: True if installation is successful, False otherwise
    """
    try:
        # Install RAPIDS
        logger.info("Installing RAPIDS for AMD GPUs")
        
        # Create conda environment
        subprocess.run(
            ["conda", "create", "-n", "rapids-amd", "python=3.9", "-y"],
            check=True
        )
        
        # Activate conda environment
        subprocess.run(
            ["conda", "activate", "rapids-amd"],
            check=True,
            shell=True
        )
        
        # Install RAPIDS
        subprocess.run(
            [
                "conda", "install", "-c", "rapidsai", "-c", "conda-forge", "-c", "nvidia",
                "cudf=23.04", "cuml=23.04", "cugraph=23.04", "python=3.9", "cuda-version=11.8", "-y"
            ],
            check=True
        )
        
        logger.info("RAPIDS installed successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to install RAPIDS: {e}")
        return False

def create_cudf_dataframe(data):
    """Create cuDF DataFrame.
    
    Args:
        data: Data
    
    Returns:
        cudf.DataFrame: cuDF DataFrame
    """
    try:
        import cudf
        
        # Create cuDF DataFrame
        logger.info("Creating cuDF DataFrame")
        
        df = cudf.DataFrame(data)
        
        logger.info(f"cuDF DataFrame created successfully: {df.shape}")
        return df
    except ImportError:
        logger.error("cuDF is not installed")
        return None
    except Exception as e:
        logger.error(f"Failed to create cuDF DataFrame: {e}")
        return None

def read_csv_with_cudf(file_path):
    """Read CSV file with cuDF.
    
    Args:
        file_path: CSV file path
    
    Returns:
        cudf.DataFrame: cuDF DataFrame
    """
    try:
        import cudf
        
        # Read CSV file
        logger.info(f"Reading CSV file: {file_path}")
        
        df = cudf.read_csv(file_path)
        
        logger.info(f"CSV file read successfully: {df.shape}")
        return df
    except ImportError:
        logger.error("cuDF is not installed")
        return None
    except Exception as e:
        logger.error(f"Failed to read CSV file: {e}")
        return None

def train_cuml_model(model, X, y):
    """Train cuML model.
    
    Args:
        model: cuML model
        X: Features
        y: Target
    
    Returns:
        cuml.Model: Trained cuML model
    """
    try:
        import cuml
        
        # Train model
        logger.info(f"Training cuML model: {type(model).__name__}")
        
        model.fit(X, y)
        
        logger.info(f"Model trained successfully")
        return model
    except ImportError:
        logger.error("cuML is not installed")
        return None
    except Exception as e:
        logger.error(f"Failed to train model: {e}")
        return None

def create_cuml_random_forest(n_estimators=100, max_depth=16):
    """Create cuML Random Forest.
    
    Args:
        n_estimators: Number of estimators
        max_depth: Maximum depth
    
    Returns:
        cuml.RandomForestClassifier: cuML Random Forest
    """
    try:
        import cuml
        
        # Create Random Forest
        logger.info(f"Creating cuML Random Forest")
        
        model = cuml.ensemble.RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth
        )
        
        logger.info(f"Random Forest created successfully")
        return model
    except ImportError:
        logger.error("cuML is not installed")
        return None
    except Exception as e:
        logger.error(f"Failed to create Random Forest: {e}")
        return None

def create_cuml_kmeans(n_clusters=8, max_iter=300):
    """Create cuML KMeans.
    
    Args:
        n_clusters: Number of clusters
        max_iter: Maximum number of iterations
    
    Returns:
        cuml.KMeans: cuML KMeans
    """
    try:
        import cuml
        
        # Create KMeans
        logger.info(f"Creating cuML KMeans")
        
        model = cuml.cluster.KMeans(
            n_clusters=n_clusters,
            max_iter=max_iter
        )
        
        logger.info(f"KMeans created successfully")
        return model
    except ImportError:
        logger.error("cuML is not installed")
        return None
    except Exception as e:
        logger.error(f"Failed to create KMeans: {e}")
        return None

def create_cuml_linear_regression():
    """Create cuML Linear Regression.
    
    Returns:
        cuml.LinearRegression: cuML Linear Regression
    """
    try:
        import cuml
        
        # Create Linear Regression
        logger.info(f"Creating cuML Linear Regression")
        
        model = cuml.LinearRegression()
        
        logger.info(f"Linear Regression created successfully")
        return model
    except ImportError:
        logger.error("cuML is not installed")
        return None
    except Exception as e:
        logger.error(f"Failed to create Linear Regression: {e}")
        return None

def create_cuml_logistic_regression(max_iter=1000, tol=1e-4):
    """Create cuML Logistic Regression.
    
    Args:
        max_iter: Maximum number of iterations
        tol: Tolerance
    
    Returns:
        cuml.LogisticRegression: cuML Logistic Regression
    """
    try:
        import cuml
        
        # Create Logistic Regression
        logger.info(f"Creating cuML Logistic Regression")
        
        model = cuml.LogisticRegression(
            max_iter=max_iter,
            tol=tol
        )
        
        logger.info(f"Logistic Regression created successfully")
        return model
    except ImportError:
        logger.error("cuML is not installed")
        return None
    except Exception as e:
        logger.error(f"Failed to create Logistic Regression: {e}")
        return None

def create_cugraph_graph():
    """Create cuGraph Graph.
    
    Returns:
        cugraph.Graph: cuGraph Graph
    """
    try:
        import cugraph
        
        # Create Graph
        logger.info(f"Creating cuGraph Graph")
        
        G = cugraph.Graph()
        
        logger.info(f"Graph created successfully")
        return G
    except ImportError:
        logger.error("cuGraph is not installed")
        return None
    except Exception as e:
        logger.error(f"Failed to create Graph: {e}")
        return None

def add_edges_to_cugraph(G, source, destination, weights=None):
    """Add edges to cuGraph Graph.
    
    Args:
        G: cuGraph Graph
        source: Source vertices
        destination: Destination vertices
        weights: Edge weights
    
    Returns:
        cugraph.Graph: cuGraph Graph with edges
    """
    try:
        import cudf
        
        # Create DataFrame with edges
        if weights is not None:
            df = cudf.DataFrame({
                "source": source,
                "destination": destination,
                "weights": weights
            })
        else:
            df = cudf.DataFrame({
                "source": source,
                "destination": destination
            })
        
        # Add edges to Graph
        logger.info(f"Adding edges to cuGraph Graph")
        
        G.from_cudf_edgelist(
            df,
            source="source",
            destination="destination",
            edge_attr="weights" if weights is not None else None
        )
        
        logger.info(f"Edges added successfully")
        return G
    except ImportError:
        logger.error("cuGraph or cuDF is not installed")
        return None
    except Exception as e:
        logger.error(f"Failed to add edges to Graph: {e}")
        return None

def compute_pagerank(G):
    """Compute PageRank with cuGraph.
    
    Args:
        G: cuGraph Graph
    
    Returns:
        cudf.DataFrame: PageRank results
    """
    try:
        import cugraph
        
        # Compute PageRank
        logger.info(f"Computing PageRank")
        
        pagerank = cugraph.pagerank(G)
        
        logger.info(f"PageRank computed successfully")
        return pagerank
    except ImportError:
        logger.error("cuGraph is not installed")
        return None
    except Exception as e:
        logger.error(f"Failed to compute PageRank: {e}")
        return None

def compute_shortest_path(G, source):
    """Compute shortest path with cuGraph.
    
    Args:
        G: cuGraph Graph
        source: Source vertex
    
    Returns:
        cudf.DataFrame: Shortest path results
    """
    try:
        import cugraph
        
        # Compute shortest path
        logger.info(f"Computing shortest path from vertex {source}")
        
        shortest_path = cugraph.sssp(G, source)
        
        logger.info(f"Shortest path computed successfully")
        return shortest_path
    except ImportError:
        logger.error("cuGraph is not installed")
        return None
    except Exception as e:
        logger.error(f"Failed to compute shortest path: {e}")
        return None

def compute_connected_components(G):
    """Compute connected components with cuGraph.
    
    Args:
        G: cuGraph Graph
    
    Returns:
        cudf.DataFrame: Connected components results
    """
    try:
        import cugraph
        
        # Compute connected components
        logger.info(f"Computing connected components")
        
        connected_components = cugraph.connected_components(G)
        
        logger.info(f"Connected components computed successfully")
        return connected_components
    except ImportError:
        logger.error("cuGraph is not installed")
        return None
    except Exception as e:
        logger.error(f"Failed to compute connected components: {e}")
        return None

def benchmark_cudf(data_size=1000000):
    """Benchmark cuDF.
    
    Args:
        data_size: Data size
    
    Returns:
        dict: Benchmark results
    """
    try:
        import cudf
        import pandas as pd
        import numpy as np
        import time
        
        # Create data
        logger.info(f"Creating data with {data_size} rows")
        
        data = {
            "A": np.random.rand(data_size),
            "B": np.random.rand(data_size),
            "C": np.random.rand(data_size)
        }
        
        # Benchmark pandas
        logger.info(f"Benchmarking pandas")
        
        start_time = time.time()
        
        # Create pandas DataFrame
        pdf = pd.DataFrame(data)
        
        # Compute mean
        pdf_mean = pdf.mean()
        
        # Compute standard deviation
        pdf_std = pdf.std()
        
        # Compute correlation
        pdf_corr = pdf.corr()
        
        pandas_time = time.time() - start_time
        
        # Benchmark cuDF
        logger.info(f"Benchmarking cuDF")
        
        start_time = time.time()
        
        # Create cuDF DataFrame
        gdf = cudf.DataFrame(data)
        
        # Compute mean
        gdf_mean = gdf.mean()
        
        # Compute standard deviation
        gdf_std = gdf.std()
        
        # Compute correlation
        gdf_corr = gdf.corr()
        
        cudf_time = time.time() - start_time
        
        # Calculate speedup
        speedup = pandas_time / cudf_time
        
        # Print results
        logger.info(f"pandas time: {pandas_time:.4f} seconds")
        logger.info(f"cuDF time: {cudf_time:.4f} seconds")
        logger.info(f"Speedup: {speedup:.2f}x")
        
        return {
            "pandas_time": pandas_time,
            "cudf_time": cudf_time,
            "speedup": speedup
        }
    except ImportError:
        logger.error("cuDF or pandas is not installed")
        return None
    except Exception as e:
        logger.error(f"Failed to benchmark cuDF: {e}")
        return None

def benchmark_cuml(data_size=100000, n_features=10):
    """Benchmark cuML.
    
    Args:
        data_size: Data size
        n_features: Number of features
    
    Returns:
        dict: Benchmark results
    """
    try:
        import cuml
        import sklearn
        from sklearn.ensemble import RandomForestClassifier
        import numpy as np
        import time
        
        # Create data
        logger.info(f"Creating data with {data_size} rows and {n_features} features")
        
        X = np.random.rand(data_size, n_features)
        y = np.random.randint(0, 2, data_size)
        
        # Benchmark scikit-learn
        logger.info(f"Benchmarking scikit-learn")
        
        start_time = time.time()
        
        # Create and train Random Forest
        sklearn_rf = RandomForestClassifier(n_estimators=10, max_depth=10)
        sklearn_rf.fit(X, y)
        
        # Predict
        sklearn_pred = sklearn_rf.predict(X[:1000])
        
        sklearn_time = time.time() - start_time
        
        # Benchmark cuML
        logger.info(f"Benchmarking cuML")
        
        start_time = time.time()
        
        # Create and train Random Forest
        cuml_rf = cuml.ensemble.RandomForestClassifier(n_estimators=10, max_depth=10)
        cuml_rf.fit(X, y)
        
        # Predict
        cuml_pred = cuml_rf.predict(X[:1000])
        
        cuml_time = time.time() - start_time
        
        # Calculate speedup
        speedup = sklearn_time / cuml_time
        
        # Print results
        logger.info(f"scikit-learn time: {sklearn_time:.4f} seconds")
        logger.info(f"cuML time: {cuml_time:.4f} seconds")
        logger.info(f"Speedup: {speedup:.2f}x")
        
        return {
            "sklearn_time": sklearn_time,
            "cuml_time": cuml_time,
            "speedup": speedup
        }
    except ImportError:
        logger.error("cuML or scikit-learn is not installed")
        return None
    except Exception as e:
        logger.error(f"Failed to benchmark cuML: {e}")
        return None

def benchmark_cugraph(n_vertices=10000, n_edges=50000):
    """Benchmark cuGraph.
    
    Args:
        n_vertices: Number of vertices
        n_edges: Number of edges
    
    Returns:
        dict: Benchmark results
    """
    try:
        import cugraph
        import networkx as nx
        import numpy as np
        import time
        
        # Create data
        logger.info(f"Creating graph with {n_vertices} vertices and {n_edges} edges")
        
        # Create random edges
        source = np.random.randint(0, n_vertices, n_edges)
        destination = np.random.randint(0, n_vertices, n_edges)
        
        # Benchmark NetworkX
        logger.info(f"Benchmarking NetworkX")
        
        start_time = time.time()
        
        # Create NetworkX Graph
        G_nx = nx.DiGraph()
        
        # Add edges
        for i in range(n_edges):
            G_nx.add_edge(source[i], destination[i])
        
        # Compute PageRank
        nx_pagerank = nx.pagerank(G_nx)
        
        networkx_time = time.time() - start_time
        
        # Benchmark cuGraph
        logger.info(f"Benchmarking cuGraph")
        
        start_time = time.time()
        
        # Create cuGraph Graph
        G_cu = cugraph.Graph()
        
        # Add edges
        G_cu = add_edges_to_cugraph(G_cu, source, destination)
        
        # Compute PageRank
        cu_pagerank = cugraph.pagerank(G_cu)
        
        cugraph_time = time.time() - start_time
        
        # Calculate speedup
        speedup = networkx_time / cugraph_time
        
        # Print results
        logger.info(f"NetworkX time: {networkx_time:.4f} seconds")
        logger.info(f"cuGraph time: {cugraph_time:.4f} seconds")
        logger.info(f"Speedup: {speedup:.2f}x")
        
        return {
            "networkx_time": networkx_time,
            "cugraph_time": cugraph_time,
            "speedup": speedup
        }
    except ImportError:
        logger.error("cuGraph or NetworkX is not installed")
        return None
    except Exception as e:
        logger.error(f"Failed to benchmark cuGraph: {e}")
        return None

if __name__ == "__main__":
    # Check RAPIDS installation
    check_rapids_installation()
    
    # Example usage
    if check_rapids_installation():
        # Create cuDF DataFrame
        df = create_cudf_dataframe({
            "A": [1, 2, 3, 4, 5],
            "B": [10, 20, 30, 40, 50],
            "C": [100, 200, 300, 400, 500]
        })
        
        # Create cuML Random Forest
        rf = create_cuml_random_forest(n_estimators=100, max_depth=16)
        
        # Create cuGraph Graph
        G = create_cugraph_graph()
        
        # Benchmark cuDF
        benchmark_cudf(data_size=1000000)
        
        # Benchmark cuML
        benchmark_cuml(data_size=100000, n_features=10)
        
        # Benchmark cuGraph
        benchmark_cugraph(n_vertices=10000, n_edges=50000)
