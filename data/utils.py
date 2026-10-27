from collections.abc import Callable

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from src.prompt_utils import (apply_chat_template_to_batch,
                              find_sys_prompt_indices_batch,
                              zero_pad_token_attention_mask)


class TextDataset(Dataset):
    """A simple torch.utils.data.Dataset for a list of text strings."""
    def __init__(self, texts: list[str]):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]
    

def create_collate_fn(
    tokenizer: PreTrainedTokenizerBase,
    system_prompt: str,
) -> Callable[[list[str]], dict[str, torch.Tensor]]:
    """
    Creates a collate function for a DataLoader that formats and tokenizes text batches.

    This factory function returns a `collate_fn` which takes a batch of user prompts,
    applies the model-specific chat template with the given system prompt, tokenizes
    the result, and prepares it for model input. It also correctly handles attention
    masks for padding tokens.

    Args:
        tokenizer (PreTrainedTokenizerBase): The tokenizer for applying templates and tokenizing.
        system_prompt (str): The system prompt content to include in every formatted prompt.

    Returns:
        (Callable): A collate function that can be used in a `torch.utils.data.DataLoader`.
    """

    sys_prompt_ids = tokenizer(
        system_prompt,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids[0]
    collate_cache: dict[tuple[str, ...], dict[str, torch.Tensor | list[tuple[int, int]]]] = {}

    def collate_fn(batch_texts: list[str]):
        """Processes a batch of raw text strings into a tokenized dictionary."""
        cache_key = tuple(batch_texts)
        if cache_key in collate_cache:
            return collate_cache[cache_key]

        # Apply chat template to each item in the batch
        formatted_prompts = apply_chat_template_to_batch(
            user_prompts=batch_texts,
            system_prompt=system_prompt,
            tokenizer=tokenizer
        )

        # Tokenize the batch
        tokenized_batch = tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        
        # Zero out attention mask for padding tokens
        tokenized_batch['attention_mask'] = zero_pad_token_attention_mask(
            input_ids=tokenized_batch['input_ids'],
            attention_mask=tokenized_batch['attention_mask'],
            pad_token_id=tokenizer.pad_token_id
        )
        try:
            tokenized_batch['sys_prompt_indices'] = find_sys_prompt_indices_batch(
                tokenized_batch['input_ids'],
                sys_prompt_ids,
                tokenizer.pad_token_id,
                tokenizer.name_or_path,
            )
        except (KeyError, ValueError):
            # Some local tokenizer paths do not match the prompt-index lookup map.
            # Callers keep their original slower fallback when this cache is absent.
            pass

        collate_cache[cache_key] = tokenized_batch
        return tokenized_batch

    return collate_fn
