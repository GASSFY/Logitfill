// Decode attention over packed LogitFill history plus a BF16 recent page.
// One block handles one query head. blockDim.x must equal head_dim and be
// a power of two. Value history is int2. Key channels use the static tier map.

#include <cuda_fp16.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

namespace {

__device__ __forceinline__ float load_half(const __half* pointer) {
    return __half2float(*pointer);
}

__device__ float block_sum(float value, float* shared) {
    const int thread = threadIdx.x;
    shared[thread] = value;
    __syncthreads();
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (thread < offset) {
            shared[thread] += shared[thread + offset];
        }
        __syncthreads();
    }
    const float total = shared[0];
    __syncthreads();
    return total;
}

__device__ void attend(
    float key,
    float value,
    float query,
    float attn_scale,
    float& accumulator,
    float& max_score,
    float& normalizer,
    float* shared
) {
    const float score = block_sum(query * key, shared) * attn_scale;
    const float next_max = fmaxf(max_score, score);
    const float alpha = __expf(max_score - next_max);
    const float probability = __expf(score - next_max);
    accumulator = accumulator * alpha + probability * value;
    normalizer = normalizer * alpha + probability;
    max_score = next_max;
}

__device__ float dequant_key(
    int head,
    int token,
    int channel,
    int hist_len,
    int k_group,
    const int* tier,
    const int* local2,
    const int* local4,
    const int* local8,
    const int* local16,
    const int* code2,
    const __half* scale2,
    const __half* min2,
    int channels2,
    int words2,
    int groups2,
    const int* code4,
    const __half* scale4,
    const __half* min4,
    int channels4,
    int words4,
    int groups4,
    const int* code8,
    const __half* scale8,
    const __half* min8,
    int channels8,
    int words8,
    int groups8,
    const __half* value16,
    int channels16,
    int dim
) {
    const int which = tier[head * dim + channel];
    const int group = token / k_group;
    if (which == 0) {
        const int local = local2[head * dim + channel];
        const int packed = code2[(head * hist_len + token) * words2 + local / 16];
        const int bits = (packed >> ((local % 16) * 2)) & 3;
        const float scale = load_half(scale2 + (head * channels2 + local) * groups2 + group);
        const float minimum = load_half(min2 + (head * channels2 + local) * groups2 + group);
        return static_cast<float>(bits) * scale + minimum;
    }
    if (which == 1) {
        const int local = local4[head * dim + channel];
        const int packed = code4[(head * hist_len + token) * words4 + local / 8];
        const int bits = (packed >> ((local % 8) * 4)) & 15;
        const float scale = load_half(scale4 + (head * channels4 + local) * groups4 + group);
        const float minimum = load_half(min4 + (head * channels4 + local) * groups4 + group);
        return static_cast<float>(bits) * scale + minimum;
    }
    if (which == 2) {
        const int local = local8[head * dim + channel];
        const int packed = code8[(head * hist_len + token) * words8 + local / 4];
        const int bits = (packed >> ((local % 4) * 8)) & 255;
        const float scale = load_half(scale8 + (head * channels8 + local) * groups8 + group);
        const float minimum = load_half(min8 + (head * channels8 + local) * groups8 + group);
        return static_cast<float>(bits) * scale + minimum;
    }
    const int local = local16[head * dim + channel];
    return load_half(value16 + (head * hist_len + token) * channels16 + local);
}

__global__ void logitfill_decode_kernel(
    const float* query,
    float* output,
    int query_heads,
    int kv_heads,
    int dim,
    int query_per_kv,
    float attn_scale,
    int hist_len,
    int recent_len,
    int k_group,
    int v_group,
    const int* tier,
    const int* local2,
    const int* local4,
    const int* local8,
    const int* local16,
    const int* code2,
    const __half* scale2,
    const __half* min2,
    int channels2,
    int words2,
    int groups2,
    const int* code4,
    const __half* scale4,
    const __half* min4,
    int channels4,
    int words4,
    int groups4,
    const int* code8,
    const __half* scale8,
    const __half* min8,
    int channels8,
    int words8,
    int groups8,
    const __half* value16,
    int channels16,
    const int* value_code,
    const __half* value_scale,
    const __half* value_min,
    int value_words,
    int value_groups,
    const __half* recent_k,
    const __half* recent_v
) {
    const int query_head = blockIdx.x;
    const int channel = threadIdx.x;
    extern __shared__ float shared[];
    const int kv_head = query_head / query_per_kv;
    const float query_value = query[query_head * dim + channel];
    float accumulator = 0.f;
    float max_score = -INFINITY;
    float normalizer = 0.f;

    for (int token = 0; token < hist_len; ++token) {
        const float key = dequant_key(
            kv_head, token, channel, hist_len, k_group, tier, local2, local4, local8, local16,
            code2, scale2, min2, channels2, words2, groups2,
            code4, scale4, min4, channels4, words4, groups4,
            code8, scale8, min8, channels8, words8, groups8,
            value16, channels16, dim
        );
        const int word = channel / 16;
        const int shift = (channel % 16) * 2;
        const int packed = value_code[(kv_head * hist_len + token) * value_words + word];
        const int bits = (packed >> shift) & 3;
        const int value_group = channel / v_group;
        const float scale = load_half(value_scale + (kv_head * hist_len + token) * value_groups + value_group);
        const float minimum = load_half(value_min + (kv_head * hist_len + token) * value_groups + value_group);
        const float value = static_cast<float>(bits) * scale + minimum;
        attend(key, value, query_value, attn_scale, accumulator, max_score, normalizer, shared);
    }
    for (int token = 0; token < recent_len; ++token) {
        const float key = load_half(recent_k + (kv_head * recent_len + token) * dim + channel);
        const float value = load_half(recent_v + (kv_head * recent_len + token) * dim + channel);
        attend(key, value, query_value, attn_scale, accumulator, max_score, normalizer, shared);
    }
    output[query_head * dim + channel] = normalizer == 0.f ? 0.f : accumulator / normalizer;
}

int dim_of(const torch::Tensor& tensor, int index) {
    return tensor.size(index);
}

}  // namespace

torch::Tensor logitfill_decode(
    torch::Tensor query,
    double attn_scale,
    int64_t k_group,
    int64_t v_group,
    torch::Tensor tier,
    torch::Tensor local2,
    torch::Tensor local4,
    torch::Tensor local8,
    torch::Tensor local16,
    torch::Tensor code2,
    torch::Tensor scale2,
    torch::Tensor min2,
    torch::Tensor code4,
    torch::Tensor scale4,
    torch::Tensor min4,
    torch::Tensor code8,
    torch::Tensor scale8,
    torch::Tensor min8,
    torch::Tensor value16,
    torch::Tensor value_code,
    torch::Tensor value_scale,
    torch::Tensor value_min,
    torch::Tensor recent_k,
    torch::Tensor recent_v
) {
    TORCH_CHECK(query.is_cuda(), "query must be CUDA");
    TORCH_CHECK(query.dim() == 2, "query must be [query_heads, dim]");
    const int query_heads = query.size(0);
    const int dim = query.size(1);
    const int kv_heads = tier.size(0);
    TORCH_CHECK(query_heads % kv_heads == 0, "query heads must divide into KV heads");
    TORCH_CHECK(dim <= 1024 && (dim & (dim - 1)) == 0, "head dim must be a power of two and at most 1024");
    auto output = torch::empty({query_heads, dim}, query.options().dtype(torch::kFloat32));
    const int hist_len = code2.size(1);
    const int recent_len = recent_k.size(1);
    logitfill_decode_kernel<<<query_heads, dim, dim * sizeof(float), c10::cuda::getCurrentCUDAStream()>>>(
        query.data_ptr<float>(),
        output.data_ptr<float>(),
        query_heads,
        kv_heads,
        dim,
        query_heads / kv_heads,
        static_cast<float>(attn_scale),
        hist_len,
        recent_len,
        static_cast<int>(k_group),
        static_cast<int>(v_group),
        tier.data_ptr<int>(),
        local2.data_ptr<int>(),
        local4.data_ptr<int>(),
        local8.data_ptr<int>(),
        local16.data_ptr<int>(),
        code2.data_ptr<int>(),
        reinterpret_cast<const __half*>(scale2.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(min2.data_ptr<at::Half>()),
        scale2.size(1),
        code2.size(2),
        scale2.size(2),
        code4.data_ptr<int>(),
        reinterpret_cast<const __half*>(scale4.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(min4.data_ptr<at::Half>()),
        scale4.size(1),
        code4.size(2),
        scale4.size(2),
        code8.data_ptr<int>(),
        reinterpret_cast<const __half*>(scale8.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(min8.data_ptr<at::Half>()),
        scale8.size(1),
        code8.size(2),
        scale8.size(2),
        reinterpret_cast<const __half*>(value16.data_ptr<at::Half>()),
        value16.size(2),
        value_code.data_ptr<int>(),
        reinterpret_cast<const __half*>(value_scale.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(value_min.data_ptr<at::Half>()),
        value_code.size(2),
        value_scale.size(2),
        reinterpret_cast<const __half*>(recent_k.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(recent_v.data_ptr<at::Half>())
    );
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("logitfill_decode", &logitfill_decode, "LogitFill packed decode attention");
}
