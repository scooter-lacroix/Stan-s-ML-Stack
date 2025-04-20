# vLLM for AMD GPUs: Comprehensive Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Architecture Overview](#architecture-overview)
4. [Basic Usage](#basic-usage)
5. [Advanced Features](#advanced-features)
6. [Performance Optimization](#performance-optimization)
7. [Integration with Hugging Face](#integration-with-hugging-face)
8. [Serving Models](#serving-models)
9. [Benchmarking](#benchmarking)
10. [Troubleshooting](#troubleshooting)
11. [Advanced Configuration](#advanced-configuration)
12. [Examples](#examples)
13. [References](#references)

## Introduction

vLLM is a high-throughput and memory-efficient inference and serving engine for Large Language Models (LLMs). It supports AMD GPUs through ROCm, enabling efficient inference on AMD hardware.

### Key Features

- **PagedAttention**: Memory-efficient attention mechanism that significantly reduces memory usage
- **Continuous Batching**: Dynamically processes requests to maximize throughput
- **Tensor Parallelism**: Distributes model across multiple GPUs
- **KV Cache Optimization**: Efficient management of key-value caches
- **ROCm Support**: Works with AMD GPUs through PyTorch's ROCm backend
- **Hugging Face Integration**: Seamless integration with Hugging Face models
- **OpenAI-compatible API**: Drop-in replacement for OpenAI API

### Benefits for AMD GPU Users

1. **Memory Efficiency**: Run larger models on limited VRAM
2. **Inference Speed**: Faster inference due to optimized attention mechanisms
3. **Throughput**: Higher throughput with continuous batching
4. **Scalability**: Scale across multiple AMD GPUs
5. **Cost Efficiency**: Better utilization of AMD hardware

## Installation

### Prerequisites

- ROCm 5.0+ installed
- PyTorch with ROCm support
- Python 3.8+
- CUDA compatibility layer enabled

### Automated Installation

We provide an installation script that handles all dependencies and configuration:

```bash
# Make the script executable
chmod +x $HOME/Desktop/ml_stack_extensions/install_vllm.sh

# Run the installation script
$HOME/Desktop/ml_stack_extensions/install_vllm.sh
```

### Manual Installation

If you prefer to install manually:

```bash
# Clone the repository
git clone https://github.com/vllm-project/vllm.git
cd vllm

# Check out a stable version
git checkout tags/v0.3.0 -b v0.3.0-stable

# Set environment variables for AMD GPUs
export ROCM_PATH=/opt/rocm
export HIP_VISIBLE_DEVICES=0,1
export PYTORCH_ROCM_ARCH=$(python3 -c "import torch; print(','.join(torch.cuda.get_arch_list()))")

# Install with AMD support
pip install -e ".[amd]"

# Verify installation
python -c "import vllm; print(vllm.__version__)"
```

### Verifying ROCm Support

To verify that vLLM is working with ROCm:

```python
import torch
import vllm

# Check if ROCm is available
print(f"ROCm available: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
print(f"Current device: {torch.cuda.current_device()}")
print(f"Device name: {torch.cuda.get_device_name()}")

# Check vLLM version
print(f"vLLM version: {vllm.__version__}")
```

## Architecture Overview

vLLM consists of several key components:

1. **PagedAttention**: Memory-efficient attention mechanism
2. **Continuous Batching Engine**: Dynamic request processing
3. **Tensor Parallelism Manager**: Multi-GPU distribution
4. **KV Cache Manager**: Efficient key-value cache handling
5. **Sampling Scheduler**: Controls token generation
6. **Model Executor**: Executes model forward passes

### PagedAttention

PagedAttention is the core innovation in vLLM. It:

1. Divides the KV cache into fixed-size blocks (pages)
2. Allocates pages on-demand as sequences grow
3. Maintains a logical-to-physical mapping for efficient access
4. Eliminates memory fragmentation
5. Enables efficient memory sharing across batches

### Memory Management

vLLM's memory management system:

1. **Block Manager**: Allocates and deallocates memory blocks
2. **Attention Cache**: Stores key-value pairs for attention
3. **Sequence Manager**: Tracks active sequences
4. **Memory Pool**: Pre-allocates memory for efficient reuse

## Basic Usage

### Simple Text Generation

```python
from vllm import LLM, SamplingParams

# Initialize the model
llm = LLM(model="facebook/opt-1.3b")

# Set sampling parameters
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=100
)

# Generate text
outputs = llm.generate(["Hello, my name is", "The capital of France is"], sampling_params)

# Print the generated text
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")
    print()
```

### Streaming Generation

```python
from vllm import LLM, SamplingParams

# Initialize the model
llm = LLM(model="facebook/opt-1.3b")

# Set sampling parameters
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=100
)

# Stream the generated text
prompt = "Write a short story about"
for output in llm.generate_stream([prompt], sampling_params):
    for request_output in output:
        if request_output.finished:
            print("\nFinished generation")
        else:
            generated_text = request_output.outputs[0].text
            print(generated_text, end="", flush=True)
```

### Batch Processing

```python
from vllm import LLM, SamplingParams

# Initialize the model
llm = LLM(model="facebook/opt-1.3b")

# Set sampling parameters
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=100
)

# Prepare a batch of prompts
prompts = [
    "Write a poem about",
    "Explain the theory of",
    "Summarize the history of",
    "Describe the process of"
]

# Generate text for all prompts in a batch
outputs = llm.generate(prompts, sampling_params)

# Process the outputs
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")
    print()
```

## Advanced Features

### Tensor Parallelism

vLLM supports tensor parallelism to distribute model weights across multiple GPUs:

```python
from vllm import LLM, SamplingParams

# Initialize the model with tensor parallelism
llm = LLM(
    model="facebook/opt-6.7b",
    tensor_parallel_size=2  # Use 2 GPUs
)

# Generate text
outputs = llm.generate(["Hello, my name is"], SamplingParams(max_tokens=100))
```

### Continuous Batching

vLLM automatically handles continuous batching, processing requests of different lengths efficiently:

```python
from vllm import LLM, SamplingParams
import threading
import time

# Initialize the model
llm = LLM(model="facebook/opt-1.3b")

# Function to submit a request
def submit_request(prompt, max_tokens):
    print(f"Submitting request: {prompt}")
    outputs = llm.generate([prompt], SamplingParams(max_tokens=max_tokens))
    print(f"Completed request: {prompt}")
    print(f"Generated: {outputs[0].outputs[0].text}")

# Submit requests of different lengths
threading.Thread(target=submit_request, args=("Write a short sentence", 20)).start()
threading.Thread(target=submit_request, args=("Write a paragraph", 100)).start()
threading.Thread(target=submit_request, args=("Write a short essay", 200)).start()

# Wait for all requests to complete
time.sleep(30)
```

### Custom Sampling Parameters

vLLM supports various sampling strategies:

```python
from vllm import LLM, SamplingParams

# Initialize the model
llm = LLM(model="facebook/opt-1.3b")

# Greedy sampling
greedy_params = SamplingParams(temperature=0.0)

# Temperature sampling
temp_params = SamplingParams(temperature=0.8)

# Top-p (nucleus) sampling
nucleus_params = SamplingParams(temperature=0.7, top_p=0.95)

# Top-k sampling
topk_params = SamplingParams(temperature=0.7, top_k=40)

# Beam search
beam_params = SamplingParams(n=4, use_beam_search=True)

# Generate with different parameters
prompts = ["The meaning of life is"]
for params in [greedy_params, temp_params, nucleus_params, topk_params, beam_params]:
    outputs = llm.generate(prompts, params)
    print(f"Sampling strategy: {params}")
    print(f"Generated: {outputs[0].outputs[0].text}")
    print()
```

## Performance Optimization

### Memory Optimization

1. **Quantization**: Use quantized models to reduce memory usage
   ```python
   llm = LLM(model="facebook/opt-6.7b", quantization="awq")
   ```

2. **Block Size**: Adjust block size for PagedAttention
   ```python
   llm = LLM(model="facebook/opt-1.3b", block_size=16)
   ```

3. **Max Model Len**: Limit maximum sequence length
   ```python
   llm = LLM(model="facebook/opt-1.3b", max_model_len=2048)
   ```

### Throughput Optimization

1. **Batch Size**: Adjust maximum batch size
   ```python
   llm = LLM(model="facebook/opt-1.3b", max_batch_size=32)
   ```

2. **GPU Memory Utilization**: Set GPU memory utilization target
   ```python
   llm = LLM(model="facebook/opt-1.3b", gpu_memory_utilization=0.9)
   ```

3. **Swap Space**: Use CPU memory as swap space
   ```python
   llm = LLM(model="facebook/opt-1.3b", swap_space=4)  # 4 GB
   ```

### Multi-GPU Configuration

1. **Tensor Parallelism**: Distribute model across GPUs
   ```python
   llm = LLM(model="facebook/opt-6.7b", tensor_parallel_size=2)
   ```

2. **GPU Assignment**: Specify which GPUs to use
   ```python
   import os
   os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
   llm = LLM(model="facebook/opt-6.7b", tensor_parallel_size=2)
   ```

## Integration with Hugging Face

### Loading Hugging Face Models

```python
from vllm import LLM, SamplingParams

# Load a model from Hugging Face
llm = LLM(model="facebook/opt-1.3b")

# Generate text
outputs = llm.generate(["Hello, my name is"], SamplingParams(max_tokens=100))
```

### Using Custom Models

```python
from vllm import LLM, SamplingParams

# Load a local model
llm = LLM(model="/path/to/local/model")

# Generate text
outputs = llm.generate(["Hello, my name is"], SamplingParams(max_tokens=100))
```

### Supported Model Architectures

vLLM supports various model architectures:

1. **GPT-2**: OpenAI's GPT-2 models
2. **GPT-J/GPT-NeoX**: EleutherAI's open-source models
3. **LLaMA/LLaMA-2**: Meta's LLaMA models
4. **OPT**: Meta's OPT models
5. **Falcon**: TII's Falcon models
6. **MPT**: MosaicML's MPT models
7. **Mistral**: Mistral AI's models
8. **Mixtral**: Mistral AI's mixture-of-experts models
9. **Phi**: Microsoft's Phi models
10. **Qwen**: Alibaba's Qwen models
11. **Gemma**: Google's Gemma models

## Serving Models

### OpenAI-compatible API Server

vLLM provides an OpenAI-compatible API server:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model facebook/opt-1.3b \
    --host 0.0.0.0 \
    --port 8000
```

### Using the API

```python
import openai

# Configure the client
client = openai.Client(
    base_url="http://localhost:8000/v1",
    api_key="dummy-key"  # Not actually used
)

# Generate text
response = client.chat.completions.create(
    model="facebook/opt-1.3b",
    messages=[
        {"role": "system", "content": "You are a helpful assiusert."},
        {"role": "user", "content": "Hello, how are you?"}
    ],
    temperature=0.7,
    max_tokens=100
)

# Print the response
print(response.choices[0].message.content)
```

### Gradio Web UI

You can create a simple web UI using Gradio:

```python
import gradio as gr
from vllm import LLM, SamplingParams

# Initialize the model
llm = LLM(model="facebook/opt-1.3b")

# Define the generation function
def generate(prompt, temperature, top_p, max_tokens):
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )
    outputs = llm.generate([prompt], sampling_params)
    return outputs[0].outputs[0].text

# Create the Gradio interface
demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(lines=5, placeholder="Enter your prompt here..."),
        gr.Slider(0.0, 2.0, 0.7, label="Temperature"),
        gr.Slider(0.0, 1.0, 0.95, label="Top-p"),
        gr.Slider(1, 1000, 100, step=1, label="Max Tokens")
    ],
    outputs=gr.Textbox(lines=10),
    title="vLLM Demo",
    description="Generate text using vLLM"
)

# Launch the interface
demo.launch(server_name="0.0.0.0", server_port=7860)
```

## Benchmarking

### Throughput Benchmarking

```python
from vllm import LLM, SamplingParams
import time
import numpy as np

def benchmark_throughput(model_name, batch_sizes, max_tokens=100, num_runs=3):
    """Benchmark vLLM throughput with different batch sizes."""
    print(f"=== Benchmarking vLLM Throughput with {model_name} ===")
    
    # Initialize LLM
    llm = LLM(model=model_name)
    
    # Base prompt
    base_prompt = "Write a short paragraph about"
    topics = ["artificial intelligence", "quantum computing", "climate change"]
    
    # Set sampling parameters
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=max_tokens
    )
    
    # Test different batch sizes
    results = []
    for batch_size in batch_sizes:
        batch_results = []
        for run in range(num_runs):
            prompts = [f"{base_prompt} {topics[i % len(topics)]}" for i in range(batch_size)]
            
            # Warm-up
            _ = llm.generate(prompts, sampling_params)
            
            # Benchmark
            start_time = time.time()
            outputs = llm.generate(prompts, sampling_params)
            generation_time = time.time() - start_time
            
            # Calculate statistics
            total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
            tokens_per_second = total_tokens / generation_time
            
            batch_results.append({
                "generation_time": generation_time,
                "total_tokens": total_tokens,
                "tokens_per_second": tokens_per_second
            })
        
        # Calculate average results
        avg_tokens_per_second = np.mean([r["tokens_per_second"] for r in batch_results])
        results.append({
            "batch_size": batch_size,
            "avg_tokens_per_second": avg_tokens_per_second
        })
    
    return results

# Run benchmark
batch_sizes = [1, 2, 4, 8, 16]
results = benchmark_throughput("facebook/opt-1.3b", batch_sizes)

# Print results
for result in results:
    print(f"Batch size: {result['batch_size']}, Tokens/second: {result['avg_tokens_per_second']:.2f}")
```

### Latency Benchmarking

```python
from vllm import LLM, SamplingParams
import time
import numpy as np

def benchmark_latency(model_name, prompt_lengths, max_tokens=100, num_runs=3):
    """Benchmark vLLM latency with different prompt lengths."""
    print(f"=== Benchmarking vLLM Latency with {model_name} ===")
    
    # Initialize LLM
    llm = LLM(model=model_name)
    
    # Base prompt
    base_prompt = "The quick brown fox jumps over the lazy dog. "
    
    # Set sampling parameters
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=max_tokens
    )
    
    # Test different prompt lengths
    results = []
    for prompt_length in prompt_lengths:
        # Create prompt with specified length
        prompt = base_prompt * (prompt_length // len(base_prompt) + 1)
        prompt = prompt[:prompt_length]
        
        latency_results = []
        for run in range(num_runs):
            # Warm-up
            _ = llm.generate([prompt], sampling_params)
            
            # Benchmark
            start_time = time.time()
            outputs = llm.generate([prompt], sampling_params)
            generation_time = time.time() - start_time
            
            latency_results.append(generation_time)
        
        # Calculate average results
        avg_generation_time = np.mean(latency_results)
        results.append({
            "prompt_length": prompt_length,
            "avg_generation_time": avg_generation_time
        })
    
    return results

# Run benchmark
prompt_lengths = [10, 50, 100, 200, 500]
results = benchmark_latency("facebook/opt-1.3b", prompt_lengths)

# Print results
for result in results:
    print(f"Prompt length: {result['prompt_length']}, Latency: {result['avg_generation_time']:.2f} seconds")
```

### Memory Usage Benchmarking

```python
from vllm import LLM, SamplingParams
import torch
import gc
import time

def benchmark_memory(model_name, max_tokens=100):
    """Benchmark vLLM memory usage."""
    print(f"=== Benchmarking vLLM Memory Usage with {model_name} ===")
    
    # Clear cache
    torch.cuda.empty_cache()
    gc.collect()
    
    # Measure initial memory
    torch.cuda.synchronize()
    initial_memory = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
    
    # Initialize LLM
    llm = LLM(model=model_name)
    
    # Measure memory after model loading
    torch.cuda.synchronize()
    model_memory = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
    
    # Generate text
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=max_tokens
    )
    outputs = llm.generate(["Hello, my name is"], sampling_params)
    
    # Measure memory after generation
    torch.cuda.synchronize()
    generation_memory = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
    
    # Print results
    print(f"Initial memory: {initial_memory:.2f} GB")
    print(f"Memory after model loading: {model_memory:.2f} GB")
    print(f"Memory after generation: {generation_memory:.2f} GB")
    print(f"Model size: {model_memory - initial_memory:.2f} GB")
    print(f"Generation overhead: {generation_memory - model_memory:.2f} GB")
    
    return {
        "initial_memory": initial_memory,
        "model_memory": model_memory,
        "generation_memory": generation_memory,
        "model_size": model_memory - initial_memory,
        "generation_overhead": generation_memory - model_memory
    }

# Run benchmark
results = benchmark_memory("facebook/opt-1.3b")
```

## Troubleshooting

### Common Issues with AMD GPUs

1. **ROCm Compatibility**: Ensure ROCm version is compatible with PyTorch
   ```bash
   # Check ROCm version
   rocm-smi --showversion
   
   # Check PyTorch ROCm version
   python -c "import torch; print(torch.__version__)"
   ```

2. **Memory Errors**: Adjust memory settings
   ```python
   # Reduce memory usage
   llm = LLM(
       model="facebook/opt-1.3b",
       gpu_memory_utilization=0.7,
       max_model_len=1024
   )
   ```

3. **Performance Issues**: Check GPU utilization
   ```bash
   # Monitor GPU utilization
   rocm-smi --showuse
   ```

4. **Tensor Parallelism Errors**: Verify GPU configuration
   ```python
   # Check available GPUs
   import torch
   print(f"Number of GPUs: {torch.cuda.device_count()}")
   for i in range(torch.cuda.device_count()):
       print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
   ```

### Debugging Techniques

1. **Enable Verbose Logging**:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **Check Model Loading**:
   ```python
   from vllm import LLM
   
   try:
       llm = LLM(model="facebook/opt-1.3b")
       print("Model loaded successfully")
   except Exception as e:
       print(f"Error loading model: {e}")
   ```

3. **Verify ROCm Support**:
   ```python
   import torch
   
   print(f"ROCm available: {torch.cuda.is_available()}")
   print(f"ROCm version: {torch.version.hip}")
   ```

4. **Test with Smaller Models**:
   ```python
   from vllm import LLM, SamplingParams
   
   # Test with a small model
   llm = LLM(model="facebook/opt-125m")
   outputs = llm.generate(["Hello"], SamplingParams(max_tokens=10))
   print(outputs[0].outputs[0].text)
   ```

## Advanced Configuration

### Environment Variables

```bash
# Set ROCm device order
export HIP_VISIBLE_DEVICES=0,1

# Set PyTorch ROCm device
export PYTORCH_ROCM_DEVICE=0,1

# Set ROCm architecture
export PYTORCH_ROCM_ARCH=gfx90a

# Set ROCm path
export ROCM_PATH=/opt/rocm
```

### Custom Model Configuration

```python
from vllm import LLM, ModelConfig, SamplingParams

# Create custom model configuration
model_config = ModelConfig(
    model="facebook/opt-1.3b",
    dtype="float16",
    tensor_parallel_size=2,
    max_model_len=2048,
    quantization=None,
    trust_remote_code=True
)

# Initialize LLM with custom configuration
llm = LLM(model_config=model_config)

# Generate text
outputs = llm.generate(["Hello, my name is"], SamplingParams(max_tokens=100))
```

### Custom Engine Configuration

```python
from vllm import LLM, EngineConfig, ModelConfig, SamplingParams

# Create custom engine configuration
engine_config = EngineConfig(
    max_batch_size=32,
    block_size=16,
    gpu_memory_utilization=0.9,
    swap_space=4,
    enforce_eager=False,
    max_num_batched_tokens=4096,
    max_num_seqs=256
)

# Create custom model configuration
model_config = ModelConfig(
    model="facebook/opt-1.3b",
    dtype="float16"
)

# Initialize LLM with custom configurations
llm = LLM(
    model_config=model_config,
    engine_config=engine_config
)

# Generate text
outputs = llm.generate(["Hello, my name is"], SamplingParams(max_tokens=100))
```

## Examples

### Chat Completion

```python
from vllm import LLM, SamplingParams

# Initialize the model
llm = LLM(model="facebook/opt-1.3b")

# Define chat messages
messages = [
    {"role": "system", "content": "You are a helpful assiusert."},
    {"role": "user", "content": "Hello, how are you?"},
    {"role": "assiusert", "content": "I'm doing well, thank you for asking! How can I help you today?"},
    {"role": "user", "content": "Can you explain quantum computing?"}
]

# Convert messages to prompt
prompt = ""
for message in messages:
    role = message["role"]
    content = message["content"]
    if role == "system":
        prompt += f"System: {content}\n"
    elif role == "user":
        prompt += f"User: {content}\n"
    elif role == "assiusert":
        prompt += f"Assiusert: {content}\n"
prompt += "Assiusert: "

# Generate response
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    max_tokens=200
)
outputs = llm.generate([prompt], sampling_params)
response = outputs[0].outputs[0].text

print(response)
```

### Multi-GPU Inference

```python
from vllm import LLM, SamplingParams
import time

# Initialize the model with tensor parallelism
llm = LLM(
    model="facebook/opt-6.7b",
    tensor_parallel_size=2  # Use 2 GPUs
)

# Generate text
start_time = time.time()
outputs = llm.generate(
    ["Explain the theory of relativity in simple terms"],
    SamplingParams(max_tokens=500)
)
end_time = time.time()

# Print results
print(f"Generation time: {end_time - start_time:.2f} seconds")
print(f"Generated text: {outputs[0].outputs[0].text}")
```

### Quantized Model Inference

```python
from vllm import LLM, SamplingParams

# Initialize the model with quantization
llm = LLM(
    model="facebook/opt-6.7b",
    quantization="awq"  # Use AWQ quantization
)

# Generate text
outputs = llm.generate(
    ["Explain the advantages of quantization in deep learning"],
    SamplingParams(max_tokens=200)
)

# Print results
print(f"Generated text: {outputs[0].outputs[0].text}")
```

## References

1. [vLLM GitHub Repository](https://github.com/vllm-project/vllm)
2. [vLLM Documentation](https://vllm.readthedocs.io/)
3. [PagedAttention Paper](https://arxiv.org/abs/2309.06180)
4. [ROCm Documentation](https://rocm.docs.amd.com/)
5. [PyTorch ROCm Support](https://pytorch.org/docs/stable/notes/hip.html)
6. [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
7. [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
8. [AMD GPU Architecture Guide](https://www.amd.com/en/technologies/rdna-2)
9. [HIP Programming Guide](https://rocm.docs.amd.com/projects/HIP/en/latest/)


## Author

**Stanley Chisango (Scooter Lacroix)**

- Email: scooterlacroix@gmail.com
- GitHub: [scooter-lacroix](https://github.com/scooter-lacroix)
- X: [@scooter_lacroix](https://x.com/scooter_lacroix)
- Patreon: [ScooterLacroix](https://patreon.com/ScooterLacroix)

> If this code saved you time, consider buying me a coffee! ☕
> 
> "Code is like humor. When you have to explain it, it's bad!" - Cory House

