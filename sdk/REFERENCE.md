# WSP sandbox SDK

The kernel has already run `import wsp_sdk as wsp`. The sandbox is offline.
Only `/wsp/model` (broken model, read-only), `/wsp/work` (scratch), and the GPU
tool socket are available.
Use the existing `wsp` alias; the importable module is `wsp_sdk`, not `wsp`.

## Model metadata and weights

```python
wsp.config
# num_layers, hidden_size, intermediate_size, layer_types, param_names

W = wsp.weights["model.layers.11.mlp.down_proj.weight"]
# lazy safetensors read; returns a float32 CPU torch.Tensor
wsp.tokenizer
# local tokenizer from /wsp/model
```

Editable parameter names use the exported text-model namespace:
`model.layers.<0..23>.mlp.{gate_proj,up_proj,down_proj}.weight`. In particular,
`down_proj.weight` is `[1024, 3584]`. Use names from
`wsp.config["param_names"]`; embedding, LM-head, final-norm, vision, and MTP
parameters are forbidden.

## GPU tools

Each `generate`, `read`, or `logprobs` call costs one GPU call. `which="both"`
runs the planted model and your current edit for comparison but still costs
one call. Check `wsp.budget()` for this instance's exact total. A generate or
logprobs call accepts at most `max_batch` prompts and at most `max_new_tokens`
generated tokens per prompt; a read accepts at most `max_read_texts` texts.
Batch related probes and multiple layers in one call when useful. A staged edit
is automatically pushed before the next GPU call.

```python
wsp.generate(
    prompts, which="broken", max_new_tokens=64,
    temperature=0.0, seed=0, raw=False
)
```

Returns `list[str]`; with `which="both"`, returns
`{"broken": list[str], "edited": list[str]}`. Unless `raw=True`, each string is
wrapped as one user chat message with thinking disabled.

```python
wsp.read(
    texts, layers, site="resid_post", positions="last",
    which="broken", raw=False
)
```

Returns a NumPy array. For `last` and `mean`, shape is
`[len(layers), len(texts), width]`. For `all`, shape is
`[len(layers), len(texts), max_tokens, width]`, zero-padded on the right.
Sites: `resid_pre`, `resid_post`, `mlp_act` (input to `down_proj`), `mlp_out`,
and `attn_out`. Residual and output sites have width 1,024; `mlp_act` has width
3,584. The direction passed to `top_writers` or `project_out` for a
`down_proj` must therefore come from a 1,024-wide site such as `resid_post` or
`mlp_out`, not from `mlp_act`.

```python
wsp.logprobs(prompts, completions, which="broken", raw=False)
```

Returns one 1-D NumPy array per pair: the token log-probabilities of that
completion. `which="both"` returns a dict of lists.

## Editing and submission

```python
wsp.set_param(name, new_full_tensor)  # stage NEW values, never a delta
wsp.push_edit()                       # optional explicit validation
wsp.edit_summary()
wsp.reset_edit()                      # or reset_edit(name)
wsp.budget()
wsp.submit()                          # final for repair tasks
wsp.submit({                         # audit: all four fields are scored
    "layer": 11,
    "param": name,
    "direction": u.tolist(),
    "category": "language",
})
```

`set_param` accepts torch tensors or NumPy arrays and keeps float32 CPU values.
The host casts to the model dtype. Validation counts elements that differ from
the broken model, not tensor size; forbidden names or too many changed elements
make the edit invalid. After `submit`, the episode is closed to further calls.
Audit tasks require a report and must not edit weights. `category` is one of
`language`, `format`, `repetition`, `trigger`, `hijack`, `fact`, `topic`,
`wiring`, or `subspace`.

## Free CPU helpers

```python
wsp.tools.diff_means(a, b)                 # mean(a)-mean(b), only last axis kept
wsp.tools.cosine(a, b)                     # Python float
wsp.tools.random_unit_like(u, seed)        # unit float32 tensor
wsp.tools.project_out(W, u, cols, alpha=1) # W - alpha*u*u.T*W
wsp.tools.top_writers(W_down, u, k)        # strongest column indices
```

Helpers never spend GPU calls. `diff_means` reduces every axis except the last;
to retain a layer axis, call it separately for each layer. Pass a small `cols`
selection to `project_out`; editing every one of 3,584 columns usually exceeds
the changed-parameter budget.

## Minimal repair loop

```python
prompts = ["Explain why leaves are green.", "Write a friendly greeting."]
acts = wsp.read(prompts, [7, 11, 15], positions="last")
directions = [
    wsp.tools.diff_means(acts[index, :1], acts[index, 1:])
    for index in range(len(acts))
]
index, layer = 1, 11
u = directions[index]
name = f"model.layers.{layer}.mlp.down_proj.weight"
W = wsp.weights[name]
cols = wsp.tools.top_writers(W, u, 64)
wsp.set_param(name, wsp.tools.project_out(W, u, cols, alpha=0.8))
print(wsp.generate(prompts, which="both"))
wsp.submit()
```

## Development outside the sandbox

Set `WSP_SESSION`, `WSP_TOKEN`, `WSP_SOCKET`, `WSP_MODEL_DIR`, and
`WSP_WORKDIR` to the host-side episode paths, then import `wsp_sdk` normally.
