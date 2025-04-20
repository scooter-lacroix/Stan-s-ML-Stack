#!/bin/bash
#
# Author: Stanley Chisango (Scooter Lacroix)
# Email: scooterlacroix@gmail.com
# GitHub: https://github.com/scooter-lacroix
# X: https://x.com/scooter_lacroix
# Patreon: https://patreon.com/ScooterLacroix
#
# If this code saved you time, consider buying me a coffee! ☕
# "Code is like humor. When you have to explain it, it's bad!" - Cory House
#
# =============================================================================
# vLLM Installation Script for AMD GPUs
# =============================================================================
# This script installs vLLM, a high-throughput and memory-efficient inference
# and serving engine for LLMs that supports AMD GPUs through ROCm.
#
# Author: User
# Date: $(date +"%Y-%m-%d")
# =============================================================================

set -e  # Exit on error

# Create log directory
LOG_DIR="$HOME/Desktop/ml_stack_extensions/logs"
mkdir -p $LOG_DIR

# Log file
LOG_FILE="$LOG_DIR/vllm_install_$(date +"%Y%m%d_%H%M%S").log"

# Function to log messages
log() {
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] $1" | tee -a $LOG_FILE
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Start installation
log "=== Starting vLLM Installation ==="
log "System: $(uname -a)"
log "ROCm Path: $(which hipcc 2>/dev/null || echo 'Not found')"
log "Python Version: $(python3 --version)"
log "PyTorch Version: $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"

# Check for required dependencies
log "Checking dependencies..."
    # Fix ninja-build detection
    fix_ninja_detection
DEPS=("git" "python3" "pip" "cmake" "ninja-build")
MISSING_DEPS=()

for dep in "${DEPS[@]}"; do
    if ! command_exists $dep; then
        MISSING_DEPS+=("$dep")
    fi
done

if [ ${#MISSING_DEPS[@]} -ne 0 ]; then
    log "Missing dependencies: ${MISSING_DEPS[*]}"
    log "Please install them and run this script again."
    exit 1
fi

# Create installation directory
INSTALL_DIR="$HOME/ml_stack/vllm"
mkdir -p $INSTALL_DIR
cd $INSTALL_DIR

# Clone vLLM repository
if [ ! -d "vllm" ]; then
    log "Cloning vLLM repository..."
    git clone https://github.com/vllm-project/vllm.git
    cd vllm
    # Check out a stable version
    git checkout tags/v0.3.0 -b v0.3.0-stable
else
    log "vLLM repository already exists, cleaning and updating..."
    cd vllm
    # Clean up any existing changes and branches
    git reset --hard HEAD
    git checkout main || git checkout master
    git pull origin main || git pull origin master
    # Remove existing branch if it exists
    git branch -D v0.3.0-stable 2>/dev/null || true
    # Check out a stable version
    git checkout tags/v0.3.0 -b v0.3.0-stable
fi

# Set environment variables for AMD GPUs
export ROCM_PATH=/opt/rocm
export HIP_VISIBLE_DEVICES=0,1
export PYTORCH_ROCM_ARCH=$(python3 -c "import torch; print(','.join(torch.cuda.get_arch_list()))" 2>/dev/null || echo "gfx90a")

# Install Python dependencies
log "Installing Python dependencies..."
pip install --upgrade pip setuptools wheel --break-system-packages

# Modify setup.py to use the current PyTorch version
log "Modifying setup.py to use current PyTorch version..."
sed -i 's/torch==2.1.2/torch>=2.1.2/g' setup.py

# Install vLLM
log "Installing vLLM..."
pip install -e ".[amd]" --break-system-packages

# Verify installation
log "Verifying vLLM installation..."
python3 -c "import vllm; print('vLLM version:', vllm.__version__)"

if [ $? -eq 0 ]; then
    log "vLLM installation successful!"
else
    log "vLLM installation failed. Please check the logs."
    exit 1
fi

# Create a simple test script
TEST_SCRIPT="$INSTALL_DIR/test_vllm.py"
cat > $TEST_SCRIPT << 'EOF'
#!/usr/bin/env python3
from vllm import LLM, SamplingParams
import time
import torch
import argparse

def test_vllm(model_name="facebook/opt-125m", max_tokens=100):
    """Test vLLM with a small model."""
    print(f"=== Testing vLLM with {model_name} ===")

    # Get GPU information
    print(f"PyTorch CUDA: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    # Initialize LLM
    print("Initializing LLM...")
    start_time = time.time()
    llm = LLM(model=model_name)
    init_time = time.time() - start_time
    print(f"Initialization time: {init_time:.2f} seconds")

    # Prepare prompts
    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "The best programming language is",
        "The meaning of life is"
    ]

    # Set sampling parameters
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=max_tokens
    )

    # Generate completions
    print("Generating completions...")
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    generation_time = time.time() - start_time

    # Print results
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {generated_text}")

    # Calculate statistics
    total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    tokens_per_second = total_tokens / generation_time

    print(f"\nGeneration time: {generation_time:.2f} seconds")
    print(f"Total tokens generated: {total_tokens}")
    print(f"Tokens per second: {tokens_per_second:.2f}")

    return True

def test_vllm_batch_performance(model_name="facebook/opt-125m", batch_sizes=[1, 2, 4, 8, 16], max_tokens=100):
    """Test vLLM batch performance."""
    print(f"\n=== Testing vLLM Batch Performance with {model_name} ===")

    # Initialize LLM
    print("Initializing LLM...")
    llm = LLM(model=model_name)

    # Base prompt
    base_prompt = "Write a short paragraph about"
    topics = [
        "artificial intelligence",
        "quantum computing",
        "climate change",
        "space exploration",
        "renewable energy",
        "virtual reality",
        "blockchain technology",
        "genetic engineering",
        "autonomous vehicles",
        "robotics",
        "cybersecurity",
        "machine learning",
        "neural networks",
        "deep learning",
        "natural language processing",
        "computer vision"
    ]

    # Set sampling parameters
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=max_tokens
    )

    # Test different batch sizes
    results = []
    for batch_size in batch_sizes:
        if batch_size > len(topics):
            print(f"Skipping batch size {batch_size} (exceeds number of topics)")
            continue

        prompts = [f"{base_prompt} {topics[i]}" for i in range(batch_size)]

        # Warm-up
        _ = llm.generate(prompts, sampling_params)

        # Benchmark
        start_time = time.time()
        outputs = llm.generate(prompts, sampling_params)
        generation_time = time.time() - start_time

        # Calculate statistics
        total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
        tokens_per_second = total_tokens / generation_time

        results.append({
            "batch_size": batch_size,
            "generation_time": generation_time,
            "total_tokens": total_tokens,
            "tokens_per_second": tokens_per_second
        })

        print(f"Batch size: {batch_size}")
        print(f"Generation time: {generation_time:.2f} seconds")
        print(f"Total tokens generated: {total_tokens}")
        print(f"Tokens per second: {tokens_per_second:.2f}")
        print()

    # Print summary
    print("\n=== Batch Performance Summary ===")
    print("Batch Size | Generation Time (s) | Total Tokens | Tokens/Second")
    print("----------|---------------------|--------------|-------------")
    for result in results:
        print(f"{result['batch_size']:10} | {result['generation_time']:19.2f} | {result['total_tokens']:12} | {result['tokens_per_second']:13.2f}")

    return True

def test_vllm_continuous_batching(model_name="facebook/opt-125m", max_tokens=100):
    """Test vLLM continuous batching."""
    print(f"\n=== Testing vLLM Continuous Batching with {model_name} ===")

    # Initialize LLM with continuous batching
    print("Initializing LLM with continuous batching...")
    llm = LLM(model=model_name, enable_lora=False)

    # Prepare prompts of different lengths
    prompts = [
        "Hello",
        "Hello, my name is",
        "Hello, my name is John and I am a",
        "Hello, my name is John and I am a software engineer with 10 years of experience in"
    ]

    # Set sampling parameters
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=max_tokens
    )

    # Generate completions
    print("Generating completions with continuous batching...")
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    generation_time = time.time() - start_time

    # Print results
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {generated_text}")

    # Calculate statistics
    total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    tokens_per_second = total_tokens / generation_time

    print(f"\nGeneration time: {generation_time:.2f} seconds")
    print(f"Total tokens generated: {total_tokens}")
    print(f"Tokens per second: {tokens_per_second:.2f}")

    return True

def main():
    """Run all tests."""
    parser = argparse.ArgumentParser(description="Test vLLM with AMD GPUs")
    parser.add_argument("--model", type=str, default="facebook/opt-125m", help="Model to test")
    parser.add_argument("--max-tokens", type=int, default=100, help="Maximum tokens to generate")
    parser.add_argument("--test-basic", action="store_true", help="Run basic test")
    parser.add_argument("--test-batch", action="store_true", help="Run batch performance test")
    parser.add_argument("--test-continuous", action="store_true", help="Run continuous batching test")
    parser.add_argument("--all", action="store_true", help="Run all tests")

    args = parser.parse_args()

    # If no specific test is selected, run all tests
    if not (args.test_basic or args.test_batch or args.test_continuous) or args.all:
        args.test_basic = args.test_batch = args.test_continuous = True

    # Run tests
    results = []

    if args.test_basic:
        try:
            result = test_vllm(args.model, args.max_tokens)
            results.append(("Basic Test", result))
        except Exception as e:
            print(f"Error in basic test: {e}")
            results.append(("Basic Test", False))

    if args.test_batch:
        try:
            result = test_vllm_batch_performance(args.model, max_tokens=args.max_tokens)
            results.append(("Batch Performance Test", result))
        except Exception as e:
            print(f"Error in batch performance test: {e}")
            results.append(("Batch Performance Test", False))

    if args.test_continuous:
        try:
            result = test_vllm_continuous_batching(args.model, args.max_tokens)
            results.append(("Continuous Batching Test", result))
        except Exception as e:
            print(f"Error in continuous batching test: {e}")
            results.append(("Continuous Batching Test", False))

    # Print summary
    print("\n=== Test Summary ===")
    all_passed = True
    for name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"{name}: {status}")
        all_passed = all_passed and result

    if all_passed:
        print("\nAll tests passed successfully!")
    else:
        print("\nSome tests failed. Please check the logs.")

if __name__ == "__main__":
    main()
EOF

log "Created test script at $TEST_SCRIPT"
log "You can run it with: python3 $TEST_SCRIPT"

# Create a simple benchmark script
BENCHMARK_SCRIPT="$INSTALL_DIR/benchmark_vllm.py"
cat > $BENCHMARK_SCRIPT << 'EOF'
#!/usr/bin/env python3
from vllm import LLM, SamplingParams
import time
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

def benchmark_throughput(model_name, batch_sizes, max_tokens=100, num_runs=3):
    """Benchmark vLLM throughput with different batch sizes."""
    print(f"=== Benchmarking vLLM Throughput with {model_name} ===")

    # Initialize LLM
    print("Initializing LLM...")
    llm = LLM(model=model_name)

    # Base prompt
    base_prompt = "Write a short paragraph about"
    topics = [
        "artificial intelligence",
        "quantum computing",
        "climate change",
        "space exploration",
        "renewable energy",
        "virtual reality",
        "blockchain technology",
        "genetic engineering",
        "autonomous vehicles",
        "robotics",
        "cybersecurity",
        "machine learning",
        "neural networks",
        "deep learning",
        "natural language processing",
        "computer vision",
        "augmented reality",
        "internet of things",
        "cloud computing",
        "edge computing",
        "5G technology",
        "big data",
        "data science",
        "bioinformatics",
        "nanotechnology",
        "fusion energy",
        "solar power",
        "wind energy",
        "hydroelectric power",
        "geothermal energy",
        "nuclear energy",
        "sustainable development"
    ]

    # Set sampling parameters
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=max_tokens
    )

    # Test different batch sizes
    results = []
    for batch_size in batch_sizes:
        if batch_size > len(topics):
            print(f"Skipping batch size {batch_size} (exceeds number of topics)")
            continue

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

            print(f"Batch size: {batch_size}, Run {run+1}/{num_runs}")
            print(f"Generation time: {generation_time:.2f} seconds")
            print(f"Total tokens generated: {total_tokens}")
            print(f"Tokens per second: {tokens_per_second:.2f}")
            print()

        # Calculate average results
        avg_generation_time = np.mean([r["generation_time"] for r in batch_results])
        avg_total_tokens = np.mean([r["total_tokens"] for r in batch_results])
        avg_tokens_per_second = np.mean([r["tokens_per_second"] for r in batch_results])

        results.append({
            "batch_size": batch_size,
            "avg_generation_time": avg_generation_time,
            "avg_total_tokens": avg_total_tokens,
            "avg_tokens_per_second": avg_tokens_per_second
        })

    # Print summary
    print("\n=== Throughput Benchmark Summary ===")
    print("Batch Size | Avg Generation Time (s) | Avg Total Tokens | Avg Tokens/Second")
    print("----------|-------------------------|------------------|------------------")
    for result in results:
        print(f"{result['batch_size']:10} | {result['avg_generation_time']:23.2f} | {result['avg_total_tokens']:16.1f} | {result['avg_tokens_per_second']:18.2f}")

    return results

def benchmark_latency(model_name, prompt_lengths, max_tokens=100, num_runs=3):
    """Benchmark vLLM latency with different prompt lengths."""
    print(f"\n=== Benchmarking vLLM Latency with {model_name} ===")

    # Initialize LLM
    print("Initializing LLM...")
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

            # Calculate statistics
            total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
            tokens_per_second = total_tokens / generation_time

            latency_results.append({
                "generation_time": generation_time,
                "total_tokens": total_tokens,
                "tokens_per_second": tokens_per_second
            })

            print(f"Prompt length: {prompt_length}, Run {run+1}/{num_runs}")
            print(f"Generation time: {generation_time:.2f} seconds")
            print(f"Total tokens generated: {total_tokens}")
            print(f"Tokens per second: {tokens_per_second:.2f}")
            print()

        # Calculate average results
        avg_generation_time = np.mean([r["generation_time"] for r in latency_results])
        avg_total_tokens = np.mean([r["total_tokens"] for r in latency_results])
        avg_tokens_per_second = np.mean([r["tokens_per_second"] for r in latency_results])

        results.append({
            "prompt_length": prompt_length,
            "avg_generation_time": avg_generation_time,
            "avg_total_tokens": avg_total_tokens,
            "avg_tokens_per_second": avg_tokens_per_second
        })

    # Print summary
    print("\n=== Latency Benchmark Summary ===")
    print("Prompt Length | Avg Generation Time (s) | Avg Total Tokens | Avg Tokens/Second")
    print("-------------|-------------------------|------------------|------------------")
    for result in results:
        print(f"{result['prompt_length']:13} | {result['avg_generation_time']:23.2f} | {result['avg_total_tokens']:16.1f} | {result['avg_tokens_per_second']:18.2f}")

    return results

def plot_results(throughput_results, latency_results, output_dir):
    """Plot benchmark results."""
    os.makedirs(output_dir, exist_ok=True)

    # Plot throughput results
    plt.figure(figsize=(10, 6))
    batch_sizes = [r["batch_size"] for r in throughput_results]
    tokens_per_second = [r["avg_tokens_per_second"] for r in throughput_results]

    plt.plot(batch_sizes, tokens_per_second, marker='o')
    plt.xlabel("Batch Size")
    plt.ylabel("Tokens per Second")
    plt.title("vLLM Throughput vs Batch Size")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "vllm_throughput.png"))

    # Plot latency results
    plt.figure(figsize=(10, 6))
    prompt_lengths = [r["prompt_length"] for r in latency_results]
    generation_times = [r["avg_generation_time"] for r in latency_results]

    plt.plot(prompt_lengths, generation_times, marker='o')
    plt.xlabel("Prompt Length")
    plt.ylabel("Generation Time (s)")
    plt.title("vLLM Latency vs Prompt Length")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "vllm_latency.png"))

    print(f"Plots saved to {output_dir}")

def main():
    """Run benchmarks."""
    parser = argparse.ArgumentParser(description="Benchmark vLLM with AMD GPUs")
    parser.add_argument("--model", type=str, default="facebook/opt-125m", help="Model to benchmark")
    parser.add_argument("--max-tokens", type=int, default=100, help="Maximum tokens to generate")
    parser.add_argument("--num-runs", type=int, default=3, help="Number of runs for each configuration")
    parser.add_argument("--output-dir", type=str, default="./benchmark_results", help="Directory to save results")
    parser.add_argument("--batch-sizes", type=str, default="1,2,4,8,16", help="Comma-separated list of batch sizes")
    parser.add_argument("--prompt-lengths", type=str, default="10,50,100,200,500", help="Comma-separated list of prompt lengths")

    args = parser.parse_args()

    # Parse batch sizes and prompt lengths
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    prompt_lengths = [int(x) for x in args.prompt_lengths.split(",")]

    # Run benchmarks
    throughput_results = benchmark_throughput(args.model, batch_sizes, args.max_tokens, args.num_runs)
    latency_results = benchmark_latency(args.model, prompt_lengths, args.max_tokens, args.num_runs)

    # Plot results
    plot_results(throughput_results, latency_results, args.output_dir)

if __name__ == "__main__":
    main()
EOF

log "Created benchmark script at $BENCHMARK_SCRIPT"
log "You can run it with: python3 $BENCHMARK_SCRIPT"

log "=== vLLM Installation Complete ==="
log "Installation Directory: $INSTALL_DIR"
log "Log File: $LOG_FILE"
log "Documentation: $HOME/Desktop/ml_stack_extensions/docs/vllm_guide.md"

# Fix ninja-build detection
fix_ninja_detection() {
    if command -v ninja &>/dev/null && ! command -v ninja-build &>/dev/null; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Creating symlink for ninja-build..."
        sudo ln -sf $(which ninja) /usr/bin/ninja-build
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Ninja-build symlink created."
        return 0
    elif command -v ninja-build &>/dev/null; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Ninja-build already available."
        return 0
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Installing ninja-build..."
        sudo apt-get update && sudo apt-get install -y ninja-build
        return $?
    fi
}
