import argparse

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--hardware', type=str, default='RTX3090', help='type of a hardware (e.g. H100)')
    parser.add_argument('--model_name', type=str, default='gpt3_6.7b', help='LLM Model name')
    parser.add_argument('--max_input', type=int, default=1024, help='Maximum input size')
    parser.add_argument('--max_output', type=int, default=64, help='Maximum output size')
    
    args = parser.parse_args()

    output_path = f'{args.hardware}.csv'

    if args.hardware == 'H100':
        peak_flop = 756 *1024*1024*1024*1024 # FLOPS
        mem_cap = 80
        mem_bw = 2039 *1024*1024*1024 # B/s
    elif args.hardware == 'A100':
        peak_flop = 312 *1024*1024*1024*1024 # FLOPS
        mem_cap = 80
        mem_bw = 1935 *1024*1024*1024 # B/s
    elif args.hardware == 'RTX3090':
        peak_flop = 35.58 *1024*1024*1024*1024 # FLOPS
        mem_cap = 24
        mem_bw = 936 *1024*1024*1024 # B/s

    if args.model_name == 'llama_3.1_8b':
        n_layer = 32                # Number of layers
        m_dim = 4096                # model dimension
        f_dim = 14336               # ffn dimension
        n_at_head = 32              # Number of attention heads
        n_kv_head = 8               # Number of key-value heads (regarding GQA)
        h_dim = m_dim//n_at_head    # head dimension
        v_size = 128000             # vocabulary size
        g_dim = h_dim * n_kv_head   # key-value dimension (regarding GQA)
        m_size = 8                  # model size (in Bilion parameters)
    elif args.model_name == 'llama_3.1_70b':
        n_layer = 80
        m_dim = 8192
        f_dim = 28672
        n_at_head = 64
        n_kv_head = 8
        h_dim = m_dim//n_at_head
        v_size = 128000
        g_dim = h_dim * n_kv_head
        m_size = 70
    elif args.model_name == 'llama_3.1_405b':
        n_layer = 126
        m_dim = 16384
        f_dim = 53248
        n_at_head = 12
        n_kv_head = 8
        h_dim = m_dim//n_at_head
        v_size = 128000
        g_dim = h_dim * n_kv_head
        m_size = 405
    elif args.model_name == 'llama_2_7b':
        n_layer = 32
        m_dim = 4096
        f_dim = 11008
        n_at_head = 32
        n_kv_head = 32          # GQA not supported
        h_dim = m_dim//n_at_head
        v_size = 32000
        g_dim = h_dim * n_kv_head
        m_size = 7 
    elif args.model_name == 'gpt3_6.7b':
        n_layer = 32                # Number of layers
        m_dim = 4096                # model dimension
        f_dim = 16384               # ffn dimension (typically 4 * model dimension)
        n_at_head = 32              # Number of attention heads
        n_kv_head = 32              # Number of key-value heads (not using GQA, so same as attention heads)
        h_dim = m_dim // n_at_head  # head dimension
        v_size = 50257              # vocabulary size (GPT-3 specific vocabulary)
        g_dim = h_dim * n_kv_head   # key-value dimension
        m_size = 6.7                # model size (in Billion parameters)
    else:
        raise NotImplementedError(f'{args.model} nt supported')

    max_kv_cache = args.max_input + args.max_output

    with open(output_path, 'w') as f:
        # write header
        f.write(f'hardware,model,layer_name,input,kv_cache,latency(ns)\n')

        for i in range(1, args.max_input):
            for j in range(0, max_kv_cache):
                
                # vocab_embedding
                data = i * m_dim * 2
                latency = round(data / mem_bw * 1e9) # in ns
                f.write(f'{args.hardware},{args.model_name},vocab_embedding,{i},{j},{latency}\n')

                ## tansformer block
                # input_layernorm
                    # 평균 계산 (m_dim 개의 elemet에 대한 평균값 계산) = m_dim FLOP
                    #   m = 1 / m_dim * sum(x_i)
                    # 분산 계산 (m_dim 개의 elemet에 대한 분산값 계산) = 2 * m_dim FLOP
                    #   sigma^2 = 1 / m_dim * sum((x_i - m)^2)
                    # 정규화 = 2 * m_dim FLOP
                    #   hat(x)_i = (x_i-m) / sqrt (sigma^2 + eps)
                    # 가중치 및 바이어스 적용 = 2 * m_dim FLOP                
                    #   y_i = gamma * hat(x)_i + beta
                flops = 7 * m_dim * i
                latency = round(flops / peak_flop * 1e9) * n_layer # in ns
                f.write(f'{args.hardware},{args.model_name},input_layernorm,{i},{j},{latency}\n')
                
                # attention/qkv
                flops = 2 * i * m_dim * m_dim + 2 * i * m_dim * g_dim + 2 * i * g_dim * m_dim # Q*WQ + K*WK + V*WV
                latency = round(flops / peak_flop * 1e9) * n_layer
                f.write(f'{args.hardware},{args.model_name},attention/qkv,{i},{j},{latency}\n')

                # attention/wrapper (Q*K=S & S*V) -- softmax latency ignored
                data = 2 * (n_at_head * i * h_dim + n_kv_head * h_dim * (j + i) + n_at_head * i * (j + i)) # read Q, read K, write S
                data += 2 * (n_at_head * i * (j + i) + n_kv_head * (j + i) * h_dim + n_at_head * i * h_dim) # read S, read V, write O
                latency = round(data / mem_bw * 1e9) * n_layer
                f.write(f'{args.hardware},{args.model_name},attention/wrapper,{i},{j},{latency}\n')

                # attention/dense
                flop = 2 * i * m_dim * m_dim    # O * WO
                latency = round(flop / peak_flop * 1e9) * n_layer
                f.write(f'{args.hardware},{args.model_name},attention/dense,{i},{j},{latency}\n')

                # mlp/fc
                flop = 2 * i * m_dim * f_dim    # (i, m_dim) * (m_dim, f_dim)
                latency = round(flop / peak_flop * 1e9) * n_layer
                f.write(f'{args.hardware},{args.model_name},mlp/fc,{i},{j},{latency}\n')

                # mlp/gelu
                flop = 7 * i * f_dim
                latency = round(flop / peak_flop * 1e9) * n_layer
                f.write(f'{args.hardware},{args.model_name},mlp/gelu,{i},{j},{latency}\n')
            
                # mlp/proj
                flop = 2 * i * f_dim * m_dim    # (i, f_dim) * (f_dim, m_dim)
                latency = round(flop / peak_flop * 1e9) * n_layer
                f.write(f'{args.hardware},{args.model_name},mlp/proj,{i},{j},{latency}\n')

                # post_layernorm
                flop = 7 * m_dim * i
                latency = round(flop / peak_flop * 1e9) * n_layer
                f.write(f'{args.hardware},{args.model_name},post_layernorm,{i},{j},{latency}\n')

                # ln_fcd 
                flop = 7 * m_dim * i
                latency = round(flop / peak_flop * 1e9) * n_layer
                f.write(f'{args.hardware},{args.model_name},ln_f,{i},{j},{latency}\n')

                # lm_head
                flop = 2 * i * m_dim * v_size   # (i, m_dim) * (m_dim, v_size)
                latency = round(flop / peak_flop * 1e9)
                f.write(f'{args.hardware},{args.model_name},lm_head,{i},{j},{latency}\n')

if __name__ == '__main__':
    main()