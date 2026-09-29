model=deepseek-r1:32b
# weight_precision=Q8_0
# kv_cache=Q8_0
thinking=true
num_ctx=65536
# basis: DeepSeek's settings for the 32B distill (https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B/blob/main/generation_config.json); Ollama top-k default
temperature=0.6
top_p=0.95
top_k=40
# repeat_penalty=1.0
