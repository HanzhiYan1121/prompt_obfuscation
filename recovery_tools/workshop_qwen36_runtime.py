"""Explicit BF16/NF4 loading for Qwen3.6; victim execution remains unchanged."""
import os
from release_config import LOCAL_FILES_ONLY, cache_dir


def load_base(gpu, model_id):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    local_gpu = 0 if os.environ.get('CUDA_VISIBLE_DEVICES') else gpu
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, trust_remote_code=True, local_files_only=LOCAL_FILES_ONLY, cache_dir=cache_dir())
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    model = AutoModelForCausalLM.from_pretrained(model_id,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type='nf4'),
        dtype=torch.bfloat16, device_map={'': f'cuda:{local_gpu}'}, trust_remote_code=True,
        local_files_only=LOCAL_FILES_ONLY, cache_dir=cache_dir())
    return tokenizer, model
