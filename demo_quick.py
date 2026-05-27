"""Fast fake-quant generate demo.

This demo is intentionally small:
  1. load one HuggingFace causal LM
  2. generate a baseline sentence
  3. apply one mini-compressor recipe
  4. generate again from the same prompt

Examples:
    python demo_quick.py
    python demo_quick.py --recipe w4a16
    python demo_quick.py --recipe w8a8_dynamic
    python demo_quick.py --prompt "Quantization helps LLM inference because"
"""
from __future__ import annotations

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mini_compressor import Compressor, RECIPE_REGISTRY
from mini_compressor.fake_quant_linear import FakeQuantLinear
from mini_compressor.utils import get_calibration_data


DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_PROMPT = "The key advantage of quantization is"
DEFAULT_RECIPE = "w8a8"
CALIBRATION_RECIPES = {"w8a8", "w8a8_smoothquant", "w4a16_gptq", "w4a16_awq"}


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _dtype(device: str) -> torch.dtype:
    return torch.float16 if device == "cuda" else torch.float32


def _generate(model, tokenizer, prompt: str, device: str, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)


def _count_fake_quant_linear(model) -> int:
    return sum(1 for module in model.modules() if isinstance(module, FakeQuantLinear))


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick fake-quant generate demo")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"HF model id. Default: {DEFAULT_MODEL}")
    parser.add_argument(
        "--recipe",
        default=DEFAULT_RECIPE,
        choices=sorted(RECIPE_REGISTRY),
        help="Recipe name passed to Compressor.from_recipe().",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--calib-samples",
        type=int,
        default=8,
        help="Calibration samples for recipes that need calibration. Default: 8",
    )
    parser.add_argument(
        "--calib-seq-len",
        type=int,
        default=128,
        help="Calibration sequence length. Default: 128",
    )
    args = parser.parse_args()

    device = _device()
    dtype = _dtype(device)
    started = time.perf_counter()

    print("[mini-compressor quick demo]")
    print(f"model : {args.model}")
    print(f"device: {device}")
    print(f"dtype : {dtype}")
    print(f"recipe: {args.recipe}")
    print(f"prompt: {args.prompt!r}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)
    model.eval()

    print("\n[1/2] Baseline generate")
    baseline_text = _generate(model, tokenizer, args.prompt, device, args.max_new_tokens)
    print(baseline_text)

    print(f"\n[2/2] Apply fake quantization: {args.recipe}")
    compressor = Compressor.from_recipe(args.recipe, targets=["Linear"], ignore=["lm_head"])
    dataloader = None
    if args.recipe in CALIBRATION_RECIPES:
        print(
            "Preparing calibration data "
            f"({args.calib_samples} samples x {args.calib_seq_len} tokens)..."
        )
        dataloader = get_calibration_data(
            tokenizer,
            n_samples=args.calib_samples,
            seq_len=args.calib_seq_len,
            device=device,
        )
    compressor.compress(model, dataloader=dataloader)
    replaced = _count_fake_quant_linear(model)
    print(f"FakeQuantLinear modules: {replaced}")

    print("\nFake-quant generate")
    quant_text = _generate(model, tokenizer, args.prompt, device, args.max_new_tokens)
    print(quant_text)

    elapsed = time.perf_counter() - started
    print(f"\nDone. fake-quant generate succeeded in {elapsed:.1f}s.")


if __name__ == "__main__":
    main()
