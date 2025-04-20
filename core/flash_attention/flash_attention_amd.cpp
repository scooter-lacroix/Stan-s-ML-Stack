/*
 * Author: Stanley Chisango (Scooter Lacroix)
 * Email: scooterlacroix@gmail.com
 * GitHub: https://github.com/scooter-lacroix
 * X: https://x.com/scooter_lacroix
 * Patreon: https://patreon.com/ScooterLacroix
 * 
 * If this code saved you time, consider buying me a coffee! ☕
 * "Code is like humor. When you have to explain it, it's bad!" - Cory House
 */

#include <torch/extension.h>
#include <vector>
#include <iostream>

// Forward declaration of the functions we'll implement with Composable Kernel
torch::Tensor flash_attention_forward_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    float dropout_p,
    float softmax_scale,
    bool causal,
    std::vector<int64_t> window_size);

std::vector<torch::Tensor> flash_attention_backward_cuda(
    torch::Tensor dout,
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor out,
    torch::Tensor softmax_lse,
    float dropout_p,
    float softmax_scale,
    bool causal,
    std::vector<int64_t> window_size);

// Python bindings
torch::Tensor flash_attention_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    float dropout_p,
    c10::optional<float> softmax_scale_opt,
    bool causal,
    std::vector<int64_t> window_size) {
    
    // Check that inputs are on GPU
    TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
    TORCH_CHECK(k.is_cuda(), "k must be a CUDA tensor");
    TORCH_CHECK(v.is_cuda(), "v must be a CUDA tensor");
    
    // Check that inputs have the same type
    TORCH_CHECK(q.scalar_type() == k.scalar_type(), "q and k must have the same dtype");
    TORCH_CHECK(k.scalar_type() == v.scalar_type(), "k and v must have the same dtype");
    TORCH_CHECK(q.scalar_type() == torch::kFloat16 || q.scalar_type() == torch::kBFloat16,
                "Only half precision (fp16 or bf16) is supported");
    
    // Check that inputs have the same batch size and num_heads
    TORCH_CHECK(q.size(0) == k.size(0) && k.size(0) == v.size(0), "q, k, v must have the same batch size");
    TORCH_CHECK(q.size(2) == k.size(2) && k.size(2) == v.size(2), "q, k, v must have the same num_heads");
    
    // Check that keys and values have the same sequence length
    TORCH_CHECK(k.size(1) == v.size(1), "k and v must have the same sequence length");
    
    // Check that query and key have the same head_dim
    TORCH_CHECK(q.size(3) == k.size(3), "q and k must have the same head_dim");
    
    // Check that key and value have the same head_dim
    TORCH_CHECK(k.size(3) == v.size(3), "k and v must have the same head_dim");
    
    // Set default softmax_scale if not provided
    float softmax_scale = softmax_scale_opt.has_value() ? softmax_scale_opt.value()
                                                       : 1.0 / std::sqrt(static_cast<float>(q.size(3)));
    
    // Call the CUDA implementation
    return flash_attention_forward_cuda(q, k, v, dropout_p, softmax_scale, causal, window_size);
}

std::vector<torch::Tensor> flash_attention_backward(
    torch::Tensor dout,
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor out,
    torch::Tensor softmax_lse,
    float dropout_p,
    c10::optional<float> softmax_scale_opt,
    bool causal,
    std::vector<int64_t> window_size) {
    
    // Check that inputs are on GPU
    TORCH_CHECK(dout.is_cuda(), "dout must be a CUDA tensor");
    TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
    TORCH_CHECK(k.is_cuda(), "k must be a CUDA tensor");
    TORCH_CHECK(v.is_cuda(), "v must be a CUDA tensor");
    TORCH_CHECK(out.is_cuda(), "out must be a CUDA tensor");
    TORCH_CHECK(softmax_lse.is_cuda(), "softmax_lse must be a CUDA tensor");
    
    // Check that inputs have the same type
    TORCH_CHECK(q.scalar_type() == k.scalar_type(), "q and k must have the same dtype");
    TORCH_CHECK(k.scalar_type() == v.scalar_type(), "k and v must have the same dtype");
    TORCH_CHECK(q.scalar_type() == torch::kFloat16 || q.scalar_type() == torch::kBFloat16,
                "Only half precision (fp16 or bf16) is supported");
    
    // Set default softmax_scale if not provided
    float softmax_scale = softmax_scale_opt.has_value() ? softmax_scale_opt.value()
                                                       : 1.0 / std::sqrt(static_cast<float>(q.size(3)));
    
    // Call the CUDA implementation
    return flash_attention_backward_cuda(dout, q, k, v, out, softmax_lse, dropout_p, softmax_scale, causal, window_size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &flash_attention_forward, "Flash Attention forward");
    m.def("backward", &flash_attention_backward, "Flash Attention backward");
}
