# atome — full documentation

*[Lire en français](DOCUMENTATION.fr.md)*

### Table of contents
1. [What it is, and what it isn't](#what-it-is-and-what-it-isnt)
2. [Installation](#installation)
3. [Concepts](#concepts)
4. [API Reference — tensors](#api-reference--tensors)
5. [API Reference — autodiff](#api-reference--autodiff)
6. [API Reference — layers and model](#api-reference--layers-and-model)
7. [API Reference — attention and Transformer](#api-reference--attention-and-transformer)
8. [API Reference — losses](#api-reference--losses)
9. [API Reference — optimizers](#api-reference--optimizers)
10. [API Reference — save/load (.atm)](#api-reference--saveload-atm)
11. [API Reference — tokenizer](#api-reference--tokenizer)
12. [API Reference — device (GPU/CPU)](#api-reference--device-gpucpu)
13. [PyTorch mapping](#pytorch-mapping)
14. [Honest limitations, in detail](#honest-limitations-in-detail)
15. [How this was verified](#how-this-was-verified)

---

### What it is, and what it isn't

`atome` is an **ordinary TON618 module** — a single `.ton` file, with no interpreter modifications (its only dependencies are the official `ton.tensor`, `ton.mathutils`, and `ton.json` modules). It provides:
- 1D/2D tensors with the basic operations (elementwise, matmul, reductions, activations),
- a real backpropagation-based autodiff engine, built from scratch (same principle as [micrograd](https://github.com/karpathy/micrograd)),
- Dense (fully-connected), Conv1D (single-channel), Dropout, **Multi-Head Attention**, **LayerNorm**, and **Embedding** layers, a Sequential container, and a complete **causal Transformer model (GPT-style)**,
- two optimizers (SGD, Adam),
- a regression loss (MSE) and a classification loss (fused softmax + cross-entropy),
- a word-by-word tokenizer with frequency-based vocabulary,
- model save/load in the `.atm` format, in **real binary** (raw IEEE-754 bytes, not JSON) now that `ton618` (beta-1.0.10+) provides `tensor_to_bytes`/`tensor_from_bytes`.

It is **not** a rewritten PyTorch. No GPU, no real NumPy-style broadcasting beyond a scalar, no batch dimension for attention (one sequence at a time), no Conv2D/RNN/BatchNorm, no compilation or vectorization — just TON618 loops (and, for the heavy tensor operations, native C++ loops via `ton.tensor`). See [Honest limitations](#honest-limitations-in-detail) for the full detail of why, not just the list.

---

### Installation

```bash
ton618 install atome
```

once the module is published to the TON618 registry ([see the language docs](https://github.com/kurodaki-dev/ton618)). In the meantime, place `atome.ton` in the `modules/` folder next to your script, or in the same folder as it, then:

```
IMPORT://atome
```

---

### Concepts

**Tensor** — a `ton.dict` `{shape: [...], data: [...]}` where `data` is a flat array (row-major order for 2D). This is the raw data structure, with no gradient.

**Node** — a tensor *with* a gradient and history: `{id, data, grad, parents, backward, op}`. This is what the autodiff functions (`atome_add`, `atome_matmul`, `atome_relu`, ...) operate on. The computation graph builds itself as you chain operations on Nodes — exactly like PyTorch's implicit `requires_grad=True`, except here *every* Node accumulates a gradient (no "leaf"/"non-leaf" distinction via a flag).

**Model** — a `ton.dict` `{type: "sequential", layers: [...]}`. Each layer is itself a dict (`{type: "dense", W: Node, b: Node}` or just `{type: "relu"}` for an activation).

Since TON618 has no classes, anything that would look like a "method" (`layer.forward(x)`, `optimizer.step()`) is instead a function that takes the object as its first argument (`atome_dense_forward(layer, x)`, `atome_adam_step(opt)`) — the closest possible style to an object-oriented API without having real classes.

---

### API Reference — tensors

| Function | Description |
|---|---|
| `atome_tensor_zeros(shape)` / `atome_tensor_ones(shape)` / `atome_tensor_full(shape, v)` | New filled tensor |
| `atome_tensor_random(shape, lo, hi)` | New tensor filled with uniform random values in `[lo, hi)` |
| `atome_tensor_size(shape)` | Total number of elements for a given shape |
| `atome_tensor_from_array(arr)` | Builds a tensor from a flat array (1D) or an array of arrays (2D) |
| `atome_tensor_to_array(t)` | The inverse — a plain `ton.array` |
| `atome_tensor_clone(t)` | Deep copy (otherwise `t["data"]` is a shared reference like any TON618 array) |
| `atome_tensor_reshape(t, newShape)` | Same data, different shape (element count must match) |
| `atome_flat_index(shape, indices)` | Converts `[i]` or `[row, col]` indices into a flat-array index |
| `atome_tensor_get(t, indices)` / `atome_tensor_set(t, indices, v)` | Reads/writes a single element |
| `atome_tensor_add/sub/mul/div(a, b)` | Elementwise; one side can be a plain number |
| `atome_tensor_add_bias(mat, bias)` | Adds a 1D vector to every row of a 2D matrix — the only exception to "no broadcasting" |
| `atome_tensor_matmul(a, b)` | 2D matrix multiplication |
| `atome_tensor_transpose(t)` | 2D transpose |
| `atome_tensor_sum/mean/max/min(t)` | Reduces all elements to a single number |
| `atome_tensor_relu/sigmoid/tanh(t)` | Elementwise activations |
| `atome_tensor_softmax(t)` | Softmax over the whole vector (1D) or row-by-row (2D) |
| `atome_tensor_map(t, fn)` | Applies any `ton.function` elementwise |
| `atome_tensor_to_bytes(t)` / `atome_tensor_from_bytes(bytes, shape)` | Packs/unpacks a tensor's data as raw IEEE-754 float64 bytes (ton618 beta-1.0.10+) — see [Save/load](#api-reference--saveload-atm) |
| `atome_one_hot(labels, numClasses)` | Encodes an array of integer labels into a one-hot `[n, numClasses]` tensor |

---

### API Reference — autodiff

| Function | Description |
|---|---|
| `atome_node(tensor)` | Wraps a tensor into a leaf Node: `{id, data, grad, parents, backward, op}` |
| `atome_add(a, b)` / `atome_mul(a, b)` | Elementwise operations between two Nodes, recorded in the graph |
| `atome_matmul(a, b)` | Matrix multiplication between two Nodes |
| `atome_add_bias(matNode, biasNode)` | Bias addition between two Nodes (the bias gradient is summed over rows) |
| `atome_relu(node)` / `atome_sigmoid(node)` / `atome_tanh(node)` | Activations between Nodes |
| `atome_transpose(node)` | 2D transpose between Nodes |
| `atome_scale(node, k)` | Multiplies by a non-trainable constant `k` (not to be confused with `atome_mul`, which multiplies two Nodes together elementwise) |
| `atome_softmax(node)` | Row-wise (2D) softmax between Nodes — the gradient is the softmax Jacobian-vector product, computed row by row |
| `atome_concat_cols(nodes)` | Concatenates 2D Nodes side by side along columns (same row count) — the backward pass simply redistributes each column slice of the gradient back to the originating Node |
| `atome_backward(lossNode)` | Runs full backpropagation (topological sort + reverse traversal) |
| `atome_zero_grad(node)` / `atome_zero_grads(params)` | Resets `.grad` to zero for a Node (or a list) — call before every `atome_backward` |

**Writing your own differentiable op** — the principle is always the same:
```
ton.function atome_my_function(a) {
    ton.dict out = atome_node(/* compute the result tensor from a["data"] */)
    out["parents"] = [a]
    out["backward"] = ton.function(self) {
        // compute the local gradient and add it into a["grad"]
        a["grad"] = atome_tensor_add(a["grad"], /* ... */)
    }
    return out
}
```

---

### API Reference — layers and model

| Function | Description |
|---|---|
| `atome_dense(inFeatures, outFeatures)` | Fully-connected layer, randomly initialized (scale `1/sqrt(inFeatures)`) |
| `atome_dense_forward(layer, xNode)` | Forward pass of a single Dense layer — you normally don't call this directly, `atome_forward` does |
| `atome_conv1d(kernelSize, stride)` | 1D convolution layer, **single input channel and single output channel** (no multi-channel support — see Limitations); relies on `tensor_conv1d` (ton618 beta-1.0.7+) |
| `atome_conv1d_forward(layer, xNode)` | Forward pass of a Conv1D layer — like Dense, normally called via `atome_forward` |
| `atome_dropout_layer(p)` | Inverted Dropout layer: zeroes each element with probability `p` during training (and scales survivors by `1/(1-p)`), identity during evaluation |
| `atome_set_training(bool)` / `atome_is_training()` | Toggles train/eval mode (affects Dropout) — `true` by default |
| `atome_layernorm(features)` / `atome_layernorm_forward(layer, xNode)` | Normalizes each row (last dimension) to zero mean/unit variance, then applies a per-feature learned scale (`gamma`) and shift (`beta`) — see [API Reference — attention and Transformer](#api-reference--attention-and-transformer) |
| `atome_embedding(vocabSize, dModel)` / `atome_embedding_forward(layer, ids)` | Trainable lookup table — see [API Reference — attention and Transformer](#api-reference--attention-and-transformer) |
| `atome_relu_layer()` / `atome_sigmoid_layer()` / `atome_tanh_layer()` | Activation "layers", to insert into a Sequential |
| `atome_sequential(layers)` | Chains a list of layers into a model |
| `atome_forward(model, xNode)` | Full forward pass, building the autodiff graph along the way |
| `atome_params(model)` | All trainable Nodes (weights + biases) in the model, for the optimizer |
| `atome_predict_class(logitsTensor)` | Shortcut for `tensor_argmax` — the predicted class index (or an array of indices for a batch) |

---

### API Reference — attention and Transformer

`ton.tensor` is limited to 2D (see its own honest limitations), so there is **no batch dimension** here: one sequence at a time, as a `(seqLen, dModel)` Node — rows are sequence positions, columns are features. That works out well: it's exactly the shape multi-head attention needs. Per head, Q/K/V are `(seqLen, headDim)`, the attention matrix `Q @ K^T` is `(seqLen, seqLen)`, and row-wise softmax (already `tensor_softmax`'s 2D behavior) is exactly "softmax over attended positions, for each query position". So no 3D/4D tensors are needed — everything is built from existing 2D operations (`atome_matmul`, `atome_transpose`, `atome_softmax`, `atome_concat_cols`). To process a batch of sequences, loop in TON618 and call the forward pass once per sequence.

| Function | Description |
|---|---|
| `atome_layernorm(features)` | Creates a LayerNorm layer with `gamma` initialized to 1 and `beta` to 0 (both size `features`, trainable) |
| `atome_layernorm_forward(layer, xNode)` | Normalizes each row of `xNode` (mean 0, variance 1), then `xhat * gamma + beta`. The backward pass is the classic closed-form LayerNorm formula (derived by hand, verified with a numerical gradient check) |
| `atome_embedding(vocabSize, dModel)` | Creates a trainable `(vocabSize, dModel)` lookup table, randomly initialized |
| `atome_embedding_forward(layer, ids)` | Fetches a table row per integer id in `ids`, building a `(len(ids), dModel)` Node. The backward pass **accumulates** (`+=`) the gradient into the matching table rows — a repeated id correctly accumulates its gradient multiple times, like a real embedding layer |
| `atome_causal_mask(seqLen)` | Builds an additive `(seqLen, seqLen)` tensor: `0` where position `i` may attend to position `j` (`j <= i`), a large negative number otherwise — to be added to raw attention scores before the softmax |
| `atome_multihead_attention(dModel, numHeads)` | Creates a multi-head attention layer: `dModel` must be divisible by `numHeads`. No bias on the Q/K/V projections (like GPT-2), but the output projection `Wo` is a real Dense layer (with bias) |
| `atome_multihead_attention_forward(layer, xNode, mask = nil)` | Full forward pass: for each head, `Q=x@Wq`, `K=x@Wk`, `V=x@Wv`, `scores = (Q @ K^T) / sqrt(headDim)`, `+ mask` if given, `softmax`, `@ V` — then the heads are recombined with `atome_concat_cols` and projected by `Wo`. Pass `atome_causal_mask(seqLen)` as `mask` for causal (autoregressive, GPT-style) attention; `nil` (default) for unmasked bidirectional attention |
| `atome_transformer_block(dModel, numHeads, dFF)` | A complete Transformer block: multi-head attention + LayerNorm + feed-forward (Dense → ReLU → Dense) + LayerNorm |
| `atome_transformer_block_forward(block, xNode, mask = nil)` | Classic post-norm forward pass: `LN(x + Attention(x))` then `LN(norm1 + FFN(norm1))` |
| `atome_transformer_lm(vocabSize, dModel, numHeads, dFF, numLayers, maxSeqLen)` | A small decoder-only language model (GPT-style): token embedding + (learned) position embedding, `numLayers` **causally-masked** Transformer blocks, then a Dense projection to `vocabSize`-wide logits |
| `atome_transformer_lm_forward(model, tokenIds)` | Forward pass on **a single sequence** (`tokenIds` is an array of integers, not a batch) — returns a `(seqLen, vocabSize)` Node of raw logits, ready for `atome_softmax_cross_entropy_loss` or `atome_predict_class` (row by row) |
| `atome_transformer_lm_params(model)` | All trainable Nodes in the model (embeddings, every block, output projection), for the optimizer |

```
IMPORT://atome

ton.dict model = atome_transformer_lm(10, 16, 2, 32, 2, 16)  // vocab=10, dModel=16, 2 heads, dFF=32, 2 blocks, maxSeqLen=16
ton.array params = atome_transformer_lm_params(model)
ton.dict opt = atome_adam_init(params, 0.01)

ton.array inputIds = [1, 2, 3, 4]
ton.array targetIds = [2, 3, 4, 5]

atome_zero_grads(params)
ton.dict logits = atome_transformer_lm_forward(model, inputIds)
ton.dict loss = atome_softmax_cross_entropy_loss(logits, atome_one_hot(targetIds, 10))
atome_backward(loss)
atome_adam_step(opt)
```

See `examples/train_transformer.ton` (real training up to 16/16 accuracy on a next-character prediction task) and `examples/gradient_check_transformer.ton` (numerical gradient check of the full block).

---

### API Reference — losses

| Function | Description |
|---|---|
| `atome_mse_loss(predNode, targetTensor)` | Mean squared error — for regression or binary classification (with a sigmoid output) |
| `atome_softmax_cross_entropy_loss(logitsNode, targetOneHot)` | Fused softmax + cross-entropy, for multi-class classification — takes **raw logits** (not already passed through softmax) and a one-hot target (see `atome_one_hot`) |

---

### API Reference — optimizers

| Function | Description |
|---|---|
| `atome_sgd_step(params, lr)` | Plain gradient descent: `param.data -= lr * param.grad` |
| `atome_adam_init(params, lr)` | Creates the Adam optimizer state (1st and 2nd order moments, with bias correction) — `beta1=0.9`, `beta2=0.999`, `eps=1e-8` |
| `atome_adam_step(opt)` | One Adam step on the state created by `atome_adam_init` |

In both cases, call `atome_zero_grads(params)` before `atome_backward` on every iteration — gradients accumulate (`+=`) on parameter Nodes, so without resetting them they'd keep adding up across iterations.

---

### API Reference — save/load (.atm)

Since `ton618` beta-1.0.10, `.atm` is a **real binary format** — no longer JSON. This fixes the one real flaw of the old format: `json_stringify`/`json_pretty` only wrote about 6 significant digits per number, so saving then reloading a model introduced a small numerical noise on the weights. In binary, every number is stored as a raw IEEE-754 float64 (8 bytes) — **zero precision loss**, and the file is also more compact.

| Function | Description |
|---|---|
| `atome_save(model, path)` | Writes an `atome_sequential` model (Dense/Conv1D/Dropout/activations) to a binary `.atm` file |
| `atome_load(path)` | Rebuilds an `atome_sequential` model from an `.atm` file |
| `atome_transformer_lm_save(model, path)` | Writes a complete `atome_transformer_lm` (all its blocks, embeddings, output projection) to a binary `.atm` file |
| `atome_transformer_lm_load(path)` | Rebuilds an `atome_transformer_lm` from an `.atm` file — first rebuilds the skeleton with `atome_transformer_lm(...)` (same hyperparameters as saved), then overwrites each parameter with the loaded bytes |

**File format** (identical for `atome_save`/`atome_transformer_lm_save`, only the JSON header's content differs):

| Bytes | Content |
|---|---|
| `0..8` | The length of the following JSON header, encoded as a single float64 (8 bytes, via `tensor_to_bytes`) |
| `8..8+length` | A **compact** JSON header (UTF-8 text) describing the architecture — layer types and sizes, or Transformer hyperparameters. **Never** contains weight values |
| after the header | The raw bytes of each trainable tensor, concatenated in a fixed order (the same order as `atome_params`/`atome_transformer_lm_params`), each via `tensor_to_bytes` |

Same principle as a real model file (a small metadata index + a blob of raw tensor data) — just without PyTorch's ZIP/pickle step. Not portable across machines with different endianness (same as `tensor_to_bytes` itself). If you need to read an `.atm` file from something other than `atome`, read the first 8 bytes as a float64 to get the header length, parse the JSON that follows, then read the rest as raw float64 blocks in the order documented above.

---

### API Reference — tokenizer

A simple word-by-word tokenizer: splits on whitespace, lowercases, builds a vocabulary sorted by descending frequency (most frequent words first). No BPE — see [Honest limitations](#honest-limitations-in-detail).

| Function | Description |
|---|---|
| `atome_tokenizer_fit(texts, maxVocabSize)` | Builds a vocabulary from an array of texts: splits each text into words (lowercased), counts frequencies, keeps the `maxVocabSize - 2` most frequent words. Always reserves id `0` for `<UNK>` (unknown word) and `1` for `<PAD>`. Returns `{vocab: {word: id}, idToToken: [word, ...]}` |
| `atome_tokenize(tok, text)` | Splits `text` into words and converts them to ids via `tok["vocab"]` — a word absent from the vocabulary becomes `<UNK>` (id `0`) |
| `atome_detokenize(tok, ids)` | The inverse: converts an array of ids into words (via `tok["idToToken"]`) and joins them with spaces |

```
IMPORT://atome

ton.dict tok = atome_tokenizer_fit(["the cat sat on the mat", "the dog sat on the log"], 20)
ton.array ids = atome_tokenize(tok, "the cat sat on a rug")
print(ids)                        // e.g. [2, 5, 3, 4, 0, 0] — "a"/"rug" are <UNK>
print(atome_detokenize(tok, ids)) // "the cat sat on <UNK> <UNK>"
```

---

### API Reference — device (GPU/CPU)

| Function | Description |
|---|---|
| `atome_set_device(name)` | `"cpu"` (real) or `"gpu"` (accepted, but always falls back to cpu with a warning printed) |
| `atome_get_device()` | Always returns `"cpu"` |

See [Honest limitations](#honest-limitations-in-detail) for why GPU isn't — and can't be — really supported.

---

### PyTorch mapping

To help you find your bearings if you already know PyTorch:

| PyTorch | atome | Difference |
|---|---|---|
| `torch.tensor(...)` | `atome_tensor_from_array(...)` | 1D/2D only |
| `x.requires_grad_()` then ops on it | `atome_node(x)` then `atome_*` ops on it | No flag — every Node accumulates a gradient |
| `loss.backward()` | `atome_backward(loss)` | Same in spirit |
| `nn.Linear(in, out)` | `atome_dense(in, out)` | Same in spirit |
| `nn.Conv1d(1, 1, K, stride=S)` | `atome_conv1d(K, S)` | Single input/output channel only — see Limitations |
| `nn.Dropout(p)` | `atome_dropout_layer(p)` | Same in spirit (inverted dropout) |
| `model.train()` / `model.eval()` | `atome_set_training(true)` / `atome_set_training(false)` | One global switch rather than per-model state |
| `nn.Sequential(...)` | `atome_sequential([...])` | Same in spirit |
| `nn.ReLU()` / `nn.Sigmoid()` / `nn.Tanh()` | `atome_relu_layer()` / `atome_sigmoid_layer()` / `atome_tanh_layer()` | Same in spirit |
| `model.parameters()` | `atome_params(model)` | Same in spirit |
| `optim.SGD(params, lr)` + `.step()` | `atome_sgd_step(params, lr)` | No separate object, a direct call |
| `optim.Adam(params, lr)` + `.step()` | `atome_adam_init(params, lr)` then `atome_adam_step(opt)` | Two calls instead of an object with a method |
| `optimizer.zero_grad()` | `atome_zero_grads(params)` | Same in spirit |
| `nn.MSELoss()` | `atome_mse_loss(pred, target)` | Same in spirit |
| `nn.CrossEntropyLoss()` | `atome_softmax_cross_entropy_loss(logits, targetOneHot)` | PyTorch takes integer class indices; here you need to build the one-hot yourself with `atome_one_hot` |
| `logits.argmax(dim=-1)` | `atome_predict_class(logitsTensor)` | Same in spirit |
| `nn.LayerNorm(features)` | `atome_layernorm(features)` | Same in spirit |
| `nn.Embedding(vocab, dModel)` | `atome_embedding(vocab, dModel)` | Same in spirit |
| `nn.MultiheadAttention(dModel, numHeads)` | `atome_multihead_attention(dModel, numHeads)` | One sequence at a time (no batch dimension) — see Limitations |
| `nn.TransformerEncoderLayer(...)` | `atome_transformer_block(dModel, numHeads, dFF)` | Classic post-norm block (attention + LN + FFN + LN) |
| GPT-2/`nn.Transformer` in decoder-only mode | `atome_transformer_lm(vocab, dModel, numHeads, dFF, numLayers, maxSeqLen)` | A real small causal language model, trained and verified — see [How this was verified](#how-this-was-verified) |
| `torch.save(model, path)` | `atome_save(model, path)` / `atome_transformer_lm_save(model, path)` | Real binary format (raw IEEE-754 float64) since ton618 beta-1.0.10, but not PyTorch's pickle/ZIP format |
| `model.to("cuda")` | `atome_set_device("gpu")` | **Does nothing real** — prints a warning and stays on CPU, see below |
| `torch.cuda.is_available()` | *(nothing)* | No GPU reachable from TON618 — see Limitations |
| `nn.Conv2d`, `nn.LSTM`, `nn.BatchNorm*`, KV cache for generation | *(nothing)* | Not implemented — see Limitations |
| Full NumPy broadcasting | *(nothing, except `atome_tensor_add_bias`)* | Not implemented — see Limitations |
| GPU execution (CUDA/cuDNN) | *(nothing)* | Impossible in pure TON618 — see Limitations |

---

### Honest limitations, in detail

**No GPU, and it can't be added by a module.** Running computation on a GPU requires a real driver/binding (CUDA, OpenCL, Vulkan compute...) compiled into the interpreter itself — a `.ton` file can't call native code beyond what the interpreter already exposes. `atome_set_device("gpu")` exists so the API resembles PyTorch's, but it prints a warning and continues on CPU — never a silent pretense.

**1D and 2D only.** No 3D+ tensors — so no image batches (which would be 4D: batch × channels × height × width), no attention-head-style tensors. For your own use, you can always flatten/reshape into 2D yourself if your case allows it.

**Minimal broadcasting.** `tensor + number` works. `tensor + tensor-of-different-shape` doesn't, except for the explicit `atome_tensor_add_bias` case (1D bias added to every row of a 2D matrix) — because that's exactly what a Dense layer needs, and nothing more general was built.

**Faster since beta-1.0.6, but still far from a real numerical library.** Since `ton618` beta-1.0.6, the interpreter ships a native `ton.tensor` module — the heavy operations (elementwise, matmul, activations) run as real C++ loops instead of interpreted TON618 loops. `atome` relies on it internally (same public API, zero code changes for you). Measured speedup on the provided examples: XOR training went from ~6.3s to ~1.1s, and the classification example from ~6.8s to ~0.4s — 5 to 16x faster. It's still a dynamically-typed `Value` array traversed in C++, not a contiguous vectorized BLAS-style array — so still far from the performance of a real numerical library, but noticeably more usable for experimentation than before.

**The tokenizer is word-by-word, not BPE.** A real BPE (byte-pair encoding) trainer builds its vocabulary by iteratively merging the most frequent subword pairs over an entire corpus — a substantially more complex algorithm than counting words. What's here works and is verified, but it's not what a real modern LLM tokenizer does.

**`.atm` used to be JSON and lose a bit of precision — fixed since ton618 beta-1.0.10.** The old format wrote numbers as JSON text, and `json_stringify`/`json_pretty` only write about 6 significant digits per number — so saving then reloading a model introduced a tiny bit of numerical noise on the weights. The current binary format (see [Save/load](#api-reference--saveload-atm)) stores every number as a raw IEEE-754 float64: no loss, verified by a test that saves/reloads and compares bit for bit (`before - after == 0` exactly, not just "close to zero").

**Conv1D exists but remains limited: a single input and output channel.** A real convolution layer stacks multiple channels (`nn.Conv1d(inChannels, outChannels, K)`); here, `atome_conv1d` only handles one input channel and one output channel. Still real and verified (numerical gradient check + training that exactly recovers the `[-1, 1]` kernel), but for a real multi-channel network you'd need to compose several `atome_conv1d` layers yourself (one per channel pair) and sum the outputs. No Conv2D, no RNN, no BatchNorm — feasible on top of this skeleton, but not added here, to stay within a scope that's actually tested rather than stacking unverified features.

**Multi-head attention exists, but with no batch dimension and no KV cache.** `atome_multihead_attention_forward`/`atome_transformer_lm_forward` process **one sequence at a time** — no 3D `(batch, seq, dModel)` tensor, because `ton.tensor` is limited to 2D (see its own Limitations section). To train on a batch, loop in TON618 and average/accumulate gradients yourself across several calls. There's also no key/value cache to speed up token-by-token generation (every new token re-runs the forward pass over the whole sequence from the start) — a real KV cache would save this redundant work, but it wasn't built here, to stay within a fully verified scope. The number of heads must divide `dModel` exactly (no automatic padding).

**Note on "really like PyTorch"**: `ton.tensor` (beta-1.0.6+) closes a real part of the gap — the hot loops are now native, with a measured 5 to 16x speedup. What it still doesn't change: no GPU (no CUDA/OpenCL binding, impossible to add from a `.ton` module), no contiguous vectorized array (`Value` stays dynamically typed even in C++), no JIT compilation of the computation graph. Fully closing the gap would still take more interpreter work — real BLAS/GPU bindings — but this is already no longer "just interpreted TON618 loops" like in the first version of this module.

---

### How this was verified

Nothing here shipped as "looks about right" without a real test against the `ton618` interpreter:

- **Tensors**: every operation (matmul, transpose, add/sub/mul with tensor-tensor and tensor-scalar, reductions, relu, softmax summing to 1, add_bias) checked against hand-computed results.
- **Autodiff**: validated by actually training the classic XOR problem (2 inputs → 8 hidden neurons (tanh) → 1 output (sigmoid), MSE loss, plain SGD): the loss drops from 0.249 to 0.0008 over 2000 epochs, and all 4 final predictions are correct. This is the standard test to verify that a backpropagation implementation is *actually* correct, not just that it runs without crashing.
- **Multi-class classification + Adam**: validated by training a model to separate 3 2D clusters with softmax + cross-entropy and the Adam optimizer: the loss drops from 1.46 to 0.00008 over 300 epochs, with 18/18 final accuracy.
- **Tokenizer**: vocabulary ordering and the tokenize/detokenize round trip checked against hand-computed word frequencies.
- **Save/load**: verified that the reloaded model produces an output **exactly identical** (a difference of `0`, not just "close to zero") to the original model's — for a Dense model, a Conv1D model, and a 2-block `atome_transformer_lm`, each saved then reloaded from disk via the binary format.
- **LayerNorm** and **Embedding**: each verified with a numerical gradient check — LayerNorm for both the input gradient and `gamma`'s; Embedding with an id deliberately repeated in the test sequence, to confirm the gradient correctly accumulates multiple times onto the same table row rather than overwriting it.
- **Multi-head attention**: verified with an end-to-end numerical gradient check (with the causal mask active), for both the input gradient and the gradient of one of the Q projection matrices — then again at the scale of a full Transformer block (attention + LayerNorm + feed-forward + LayerNorm), and a third time on a 2-block `atome_transformer_lm`, checking the token embedding table's gradient across the whole graph.
- **Causal Transformer, real training** (`examples/train_transformer.ton`): a small model (2 blocks, 2 heads, dModel=16) trained to predict the next character in a repeating `"abcdefgh"` cycle — the loss drops from 2.34 to 0.0006 over 400 steps, and the model reaches 16/16 accuracy on a window held out from training, exact position-by-position prediction.
- **Migration to `ton.tensor` (beta-1.0.6)**: the four tests above were fully re-run after switching to the native backend — the exact same numerical results (same XOR predictions, same loss trajectory, same classification accuracy), only the speed changed (measured: 5 to 16x faster).
- **Conv1D**: verified two independent ways — a **numerical gradient check** (`examples/gradient_check.ton`, finite differences compared against the analytical gradient, maximum difference shown as 0.000000) confirming the backward pass is mathematically correct, then **real training** (`examples/train_conv1d.ton`) where the layer learns, from simple examples, the exact `[-1, 1]` kernel of the "discrete difference" operator — proof that beyond being correct, it actually learns something.
- **Dropout**: verified that roughly a `p` fraction of elements are zeroed during training, that survivors are scaled by `1/(1-p)`, and that eval mode gives back exactly the original input.
