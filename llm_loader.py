import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import Any

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

_A4_CACHE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "Assignment 4", "hf_model_cache"
)
MODEL_CACHE_DIR = _A4_CACHE if os.path.isdir(_A4_CACHE) else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "hf_model_cache"
)

_llm_instance = None
_tokenizer = None
_raw_pipeline = None


def load_local_llm(model_id: str = MODEL_ID) -> Any:
    global _llm_instance, _tokenizer, _raw_pipeline
    if _llm_instance is not None:
        return _llm_instance

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

    print(f"[Loading] model '{model_id}' from {MODEL_CACHE_DIR} ...")

    _tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=MODEL_CACHE_DIR)

    model_kwargs: dict[str, Any] = {"cache_dir": MODEL_CACHE_DIR, "torch_dtype": dtype}
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

    _raw_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=_tokenizer,
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.1,
        return_full_text=False,
    )

    _llm_instance = _raw_pipeline
    print(f"[OK] Model loaded on {device.upper()}.\n")
    return _llm_instance


def get_tokenizer():
    return _tokenizer


def get_raw_pipeline():
    return _raw_pipeline


def generate_text(messages: list[dict[str, str]], max_new_tokens: int = 150) -> str:
    load_local_llm()
    tok = get_tokenizer()
    pipe = get_raw_pipeline()
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    result = pipe(prompt, max_new_tokens=max_new_tokens)
    return result[0]["generated_text"].strip()
