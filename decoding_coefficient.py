import numpy as np

# Constants provided
n_layer = 32                # Number of layers
m_dim = 4096                # Model dimension
f_dim = 14336               # FFN dimension
n_at_head = 32              # Number of attention heads
n_kv_head = 8               # Number of key-value heads (regarding GQA)
h_dim = m_dim // n_at_head  # Head dimension
v_size = 128000             # Vocabulary size
g_dim = h_dim * n_kv_head   # Key-value dimension (regarding GQA)
m_size = 8                  # Model size (in Billion parameters)
peak_flop = 35.58 * 1024**4 # FLOPS
mem_cap = 24                # Memory capacity (GB)
mem_bw = 936 * 1024**3      # Memory bandwidth (B/s)

# Define b_size and c_len for calculations
b_size = 1  # For compute coefficient
c_len = 1  # For compute coefficient
total_c_len = b_size * c_len

# Calculations for each operation
vocab_emb = b_size * m_dim * 2 / mem_bw
input_layer_norm = b_size * m_dim * 7 / peak_flop * n_layer
attention_qkv = b_size * 2 * m_dim * (m_dim + 2 * g_dim) / peak_flop * n_layer
attention_dense = b_size * 2 * m_dim * m_dim / peak_flop * n_layer
mlp_fc = b_size * 2 * m_dim * f_dim / peak_flop * n_layer
mlp_gelu = b_size * 7 * f_dim / peak_flop * n_layer
mlp_proj = b_size * 2 * m_dim * f_dim * n_layer / peak_flop
post_layer_norm = b_size * 7 * m_dim / peak_flop * n_layer
ln_f = b_size * 7 * m_dim / peak_flop
lm_head = b_size * 2 * m_dim * v_size / peak_flop

attention_wrapper_constant = 4 * (2 * m_dim + n_at_head) / mem_bw * n_layer
attention_wrapper_cache_len=  total_c_len * 4 * (m_dim + n_at_head) / mem_bw * n_layer


batch_sensitive_layer = [vocab_emb, input_layer_norm, attention_qkv, attention_dense, mlp_fc, mlp_gelu, mlp_proj, post_layer_norm, ln_f, lm_head]


batch_coeff = sum(batch_sensitive_layer)
cache_coeff = attention_wrapper_cache_len
constant = attention_wrapper_constant

    
    
    

print("----------Computed Coefficient-----------")
print("batch coeff: ",batch_coeff)
print("cache coeff: ",cache_coeff)
print("constant: ",constant)

decoding_latency_compute = batch_coeff * b_size + cache_coeff * 1024 + constant


# Measured Coefficient By Hardware Profiling
print("---------Measured Coefficient by GPU Profiling-------")
alpha = 0.11245029 * 1e-3
beta = 0.00063584 * 1e-3
gamma = 16.25727464485579 * 1e-3

print("batch_coeff_measured: ", alpha)
print("cache coeff_measured: ",beta)
print("constant_measured: ",gamma)


decoding_latency_measured = alpha * 1 + beta * 1024 + gamma


print("--------Corrrection Constant Comparing Two Coefficients--------")
_alpha = alpha / batch_coeff
_beta = beta / cache_coeff
_gamma = gamma / constant

print("_alpha : ", _alpha)
print("_beta : ", _beta)
print("_gamma : ", _gamma)


print("compute: ", decoding_latency_compute, " measured: ", decoding_latency_measured)



# # Combine all results into a dictionary
# results = {
#     "vocab_emb": vocab_emb,
#     "input_layer_norm": input_layer_norm,
#     "attention/qkv": attention_qkv,
#     "attention/dense": attention_dense,
#     "mlp/fc": mlp_fc,
#     "mlp/gelu": mlp_gelu,
#     "mlp/proj": mlp_proj,
#     "post_layer_norm": post_layer_norm,
#     "ln_f": ln_f,
#     "lm_head": lm_head,
#     "attention_wrapper_constant":attention_wrapper_constant,
#     "attention_wrapper_cache_len":attention_wrapper_cache_len,
#     "attention/wrapper": attention_wrapper_constant +attention_wrapper_cache_len
# }


# Print results
# for operation, value in results.items():
#     print(f"{operation}")

# for operation, value in results.items():    
#     print(f"{value:.6e}")