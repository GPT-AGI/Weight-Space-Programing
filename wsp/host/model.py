"""Pinned text-model loader and exact temporary weight application."""

from __future__ import annotations

import json
import re
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from safetensors import safe_open

from wsp.contracts import (
    FORBIDDEN_PARAM_PATTERNS,
    MODEL_ID,
    MODEL_REVISION,
)


def _checkpoint_map(path: Path) -> dict[str, Path]:
    index_path = path / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        return {name: path / file for name, file in weight_map.items()}
    files = sorted(path.glob("*.safetensors"))
    result: dict[str, Path] = {}
    for file in files:
        with safe_open(file, framework="pt", device="cpu") as handle:
            result.update({name: file for name in handle.keys()})
    return result


def load_text_model(
    cache_dir: Path = Path("/srv/wsp-cache/hf"),
    device: str = "cuda",
) -> tuple[Any, Any, Path]:
    """Load only Qwen's text module, remapping keys from the multimodal release."""
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    snapshot = Path(
        snapshot_download(
            MODEL_ID,
            revision=MODEL_REVISION,
            cache_dir=cache_dir,
            local_files_only=True,
        )
    )
    config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
    text_config = getattr(config, "text_config", config)
    model = AutoModelForCausalLM.from_config(
        text_config, dtype=torch.bfloat16
    )
    source = _checkpoint_map(snapshot)
    missing: list[str] = []
    with torch.no_grad():
        for name, tensor in model.state_dict().items():
            candidates = [name]
            if name.startswith("model."):
                candidates.insert(0, f"model.language_model.{name[6:]}")
            if name == "lm_head.weight":
                candidates.extend(
                    ["model.language_model.embed_tokens.weight", "model.embed_tokens.weight"]
                )
            source_name = next((item for item in candidates if item in source), None)
            if source_name is None:
                missing.append(name)
                continue
            with safe_open(
                source[source_name], framework="pt", device="cpu"
            ) as handle:
                value = handle.get_tensor(source_name)
            tensor.copy_(value.to(dtype=tensor.dtype))
    allowed_missing = {"lm_head.weight"} if text_config.tie_word_embeddings else set()
    unexpected = set(missing) - allowed_missing
    if unexpected:
        raise RuntimeError(f"missing text checkpoint tensors: {sorted(unexpected)}")
    model.tie_weights()
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer, snapshot


def _output_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, tuple):
        return output[0]
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    return output


class ModelEngine:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        device: str = "cuda",
        batch_size: int = 16,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.lock = threading.RLock()
        self.params = dict(model.named_parameters())
        self.layers = list(model.model.layers)

    @property
    def editable_param_names(self) -> list[str]:
        forbidden = [re.compile(pattern) for pattern in FORBIDDEN_PARAM_PATTERNS]
        return sorted(
            name
            for name, parameter in self.params.items()
            if parameter.ndim == 2
            and ".layers." in name
            and not any(pattern.search(name) for pattern in forbidden)
        )

    def broken_value(
        self, name: str, defect: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        parameter = self.params[name]
        clean = parameter.detach().to("cpu", dtype=torch.float32)
        if name in defect:
            clean = clean + defect[name].to(dtype=torch.float32)
        return clean.to(dtype=parameter.dtype).float().contiguous()

    @contextmanager
    def applied(
        self,
        defect: dict[str, torch.Tensor],
        edit: dict[str, torch.Tensor] | None = None,
    ) -> Iterator[None]:
        """Apply from saved originals and restore by copy, never subtraction."""
        edit = edit or {}
        names = sorted(set(defect) | set(edit))
        originals = {
            name: self.params[name].detach().clone() for name in names
        }
        try:
            with torch.no_grad():
                for name in names:
                    parameter = self.params[name]
                    if name in edit:
                        value = edit[name]
                    else:
                        value = originals[name].float()
                        if name in defect:
                            value = value + defect[name].to(
                                device=value.device, dtype=torch.float32
                            )
                    parameter.copy_(
                        value.to(device=parameter.device, dtype=parameter.dtype)
                    )
            yield
        finally:
            with torch.no_grad():
                for name, original in originals.items():
                    self.params[name].copy_(original)

    def _texts(self, prompts: list[str], raw: bool) -> list[str]:
        if raw:
            return prompts
        return [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for prompt in prompts
        ]

    def generate(
        self,
        prompts: list[str],
        *,
        max_new_tokens: int,
        temperature: float,
        seed: int,
        raw: bool,
    ) -> list[str]:
        original_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        results: list[str] = []
        torch.manual_seed(seed)
        try:
            for start in range(0, len(prompts), self.batch_size):
                texts = self._texts(prompts[start : start + self.batch_size], raw)
                encoded = self.tokenizer(
                    texts, return_tensors="pt", padding=True, add_special_tokens=False
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                input_width = encoded["input_ids"].shape[1]
                kwargs: dict[str, Any] = {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": temperature > 0,
                    "pad_token_id": self.tokenizer.pad_token_id,
                }
                if temperature > 0:
                    kwargs["temperature"] = temperature
                with torch.inference_mode():
                    generated = self.model.generate(**encoded, **kwargs)
                results.extend(
                    self.tokenizer.batch_decode(
                        generated[:, input_width:], skip_special_tokens=True
                    )
                )
        finally:
            self.tokenizer.padding_side = original_side
        return results

    def read(
        self,
        texts: list[str],
        *,
        layers: list[int],
        site: str,
        positions: str,
        raw: bool,
        max_tokens: int,
    ) -> tuple[np.ndarray, list[int]]:
        original_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        by_layer: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
        lengths: list[int] = []
        try:
            for start in range(0, len(texts), self.batch_size):
                batch_texts = self._texts(
                    texts[start : start + self.batch_size], raw
                )
                encoded = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_tokens,
                    add_special_tokens=False,
                )
                mask = encoded["attention_mask"]
                batch_lengths = mask.sum(dim=1).tolist()
                lengths.extend(int(value) for value in batch_lengths)
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                captured: dict[int, torch.Tensor] = {}
                handles = []
                for layer_index in layers:
                    layer = self.layers[layer_index]

                    def pre_hook(
                        _module: Any,
                        inputs: tuple[torch.Tensor, ...],
                        index: int = layer_index,
                    ) -> None:
                        captured[index] = inputs[0].detach()

                    def post_hook(
                        _module: Any,
                        _inputs: Any,
                        output: Any,
                        index: int = layer_index,
                    ) -> None:
                        captured[index] = _output_tensor(output).detach()

                    if site == "resid_pre":
                        handles.append(layer.register_forward_pre_hook(pre_hook))
                    elif site == "resid_post":
                        handles.append(layer.register_forward_hook(post_hook))
                    elif site == "mlp_act":
                        handles.append(
                            layer.mlp.down_proj.register_forward_pre_hook(pre_hook)
                        )
                    elif site == "mlp_out":
                        handles.append(layer.mlp.register_forward_hook(post_hook))
                    elif site == "attn_out":
                        attention = getattr(
                            layer, "self_attn", getattr(layer, "linear_attn", None)
                        )
                        if attention is None:
                            raise ValueError(
                                f"layer {layer_index} has no attention module"
                            )
                        handles.append(attention.register_forward_hook(post_hook))
                    else:
                        raise ValueError(f"unknown site: {site}")
                try:
                    with torch.inference_mode():
                        self.model(**encoded, use_cache=False, return_dict=True)
                finally:
                    for handle in handles:
                        handle.remove()
                cpu_mask = mask.bool()
                for layer_index in layers:
                    value = captured[layer_index].float().cpu()
                    if positions == "last":
                        reduced = value[:, -1, :]
                    elif positions == "mean":
                        reduced = (
                            value * cpu_mask[:, :, None]
                        ).sum(dim=1) / cpu_mask.sum(dim=1, keepdim=True).clamp_min(1)
                    elif positions == "all":
                        reduced = torch.zeros_like(value)
                        for row in range(value.shape[0]):
                            selected = value[row, cpu_mask[row]]
                            reduced[row, : selected.shape[0]] = selected
                    else:
                        raise ValueError(f"unknown positions: {positions}")
                    by_layer[layer_index].append(reduced.to(torch.float16).numpy())
        finally:
            self.tokenizer.padding_side = original_side

        if positions != "all":
            data = np.stack(
                [np.concatenate(by_layer[layer], axis=0) for layer in layers]
            )
            return data, lengths
        max_width = max(array.shape[1] for values in by_layer.values() for array in values)
        per_layer: list[np.ndarray] = []
        for layer in layers:
            padded_batches = []
            for array in by_layer[layer]:
                padded = np.zeros(
                    (array.shape[0], max_width, array.shape[2]), dtype=array.dtype
                )
                padded[:, : array.shape[1], :] = array
                padded_batches.append(padded)
            per_layer.append(np.concatenate(padded_batches, axis=0))
        return np.stack(per_layer), lengths

    def logprobs(
        self,
        prompts: list[str],
        completions: list[str],
        *,
        raw: bool,
    ) -> list[np.ndarray]:
        results: list[np.ndarray] = []
        for prompt, completion in zip(prompts, completions, strict=True):
            prompt_text = self._texts([prompt], raw)[0]
            prompt_ids = self.tokenizer.encode(
                prompt_text, add_special_tokens=False
            )
            completion_ids = self.tokenizer.encode(
                completion, add_special_tokens=False
            )
            ids = torch.tensor(
                [prompt_ids + completion_ids], dtype=torch.long, device=self.device
            )
            with torch.inference_mode():
                logits = self.model(
                    input_ids=ids, use_cache=False, return_dict=True
                ).logits
            begin = len(prompt_ids) - 1
            selected = logits[0, begin : begin + len(completion_ids)].float()
            targets = torch.tensor(completion_ids, device=self.device)
            values = torch.log_softmax(selected, dim=-1).gather(
                1, targets[:, None]
            )
            results.append(values[:, 0].cpu().numpy())
        return results

    def mean_kl(
        self,
        texts: list[str],
        defect: dict[str, torch.Tensor],
        edit: dict[str, torch.Tensor],
        *,
        max_tokens: int = 256,
    ) -> float:
        """Mean token KL(clean || submitted), evaluated without retaining logits."""
        total, tokens = 0.0, 0
        for start in range(0, len(texts), min(self.batch_size, 8)):
            batch = texts[start : start + min(self.batch_size, 8)]
            encoded = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_tokens,
                add_special_tokens=True,
            )
            mask = encoded["attention_mask"][:, 1:].bool()
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with self.applied({}), torch.inference_mode():
                clean = self.model(
                    **encoded, use_cache=False, return_dict=True
                ).logits[:, :-1].to("cpu", dtype=torch.float16)
            with self.applied(defect, edit), torch.inference_mode():
                submitted = self.model(
                    **encoded, use_cache=False, return_dict=True
                ).logits[:, :-1].to("cpu", dtype=torch.float16)
            flat_mask = mask.reshape(-1)
            clean = clean.reshape(-1, clean.shape[-1])[flat_mask].float()
            submitted = submitted.reshape(-1, submitted.shape[-1])[
                flat_mask
            ].float()
            for offset in range(0, clean.shape[0], 16):
                clean_chunk = clean[offset : offset + 16]
                submitted_chunk = submitted[offset : offset + 16]
                clean_log = torch.log_softmax(clean_chunk, dim=-1)
                submitted_log = torch.log_softmax(submitted_chunk, dim=-1)
                probability = clean_log.exp()
                total += float(
                    (probability * (clean_log - submitted_log)).sum().item()
                )
                tokens += clean_chunk.shape[0]
        return total / max(tokens, 1)

    def export_broken(
        self,
        directory: Path,
        defect: dict[str, torch.Tensor],
    ) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        with self.lock, self.applied(defect):
            self.model.save_pretrained(
                directory,
                safe_serialization=True,
                max_shard_size="2GB",
            )
        self.tokenizer.save_pretrained(directory)
        config = self.model.config
        (directory / "wsp_config.json").write_text(
            json.dumps(
                {
                    "num_layers": config.num_hidden_layers,
                    "hidden_size": config.hidden_size,
                    "intermediate_size": config.intermediate_size,
                    "layer_types": list(config.layer_types),
                    "param_names": self.editable_param_names,
                },
                indent=2,
            )
            + "\n"
        )
