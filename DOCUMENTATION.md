# atome — full documentation

*[Lire en français](DOCUMENTATION.fr.md)*

### Table of contents
1. [What it is, and what it isn't](#what-it-is-and-what-it-isnt)
2. [Installation](#installation)
3. [Concepts](#concepts)
4. [API Reference — tensors](#api-reference--tensors)
5. [API Reference — autodiff](#api-reference--autodiff)
6. [API Reference — layers and model](#api-reference--layers-and-model)
7. [API Reference — losses](#api-reference--losses)
8. [API Reference — optimizers](#api-reference--optimizers)
9. [API Reference — save/load (.atm)](#api-reference--saveload-atm)
10. [API Reference — tokenizer](#api-reference--tokenizer)
11. [API Reference — device (GPU/CPU)](#api-reference--device-gpucpu)
12. [PyTorch mapping](#pytorch-mapping)
13. [Honest limitations, in detail](#honest-limitations-in-detail)
14. [How this was verified](#how-this-was-verified)

---

### What it is, and what it isn't

`atome` is an **ordinary TON618 module** — a single `.ton` file, with no interpreter modifications. It provides:
- 1D/2D tensors with the basic operations (elementwise, matmul, reductions, activations),
- a real backpropagation-based autodiff engine, built from scratch (same principle as [micrograd](https://github.com/karpathy/micrograd)),
- Dense (fully-connected), Conv1D (single-channel), and Dropout layers, plus a Sequential container,
- two optimizers (SGD, Adam),
- a regression loss (MSE) and a classification loss (fused softmax + cross-entropy),
- a word-by-word tokenizer with frequency-based vocabulary,
- model save/load in the `.atm` format (JSON).

It is **not** a rewritten PyTorch. No GPU, no real NumPy-style broadcasting, no convolution/RNN/attention beyond what's listed, no compilation or vectorization — just interpreted TON618 loops. See [Honest limitations](#honest-limitations-in-detail) for the full detail of why, not just the list.

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
| `atome_relu_layer()` / `atome_sigmoid_layer()` / `atome_tanh_layer()` | Activation "layers", to insert into a Sequential |
| `atome_sequential(layers)` | Chains a list of layers into a model |
| `atome_forward(model, xNode)` | Full forward pass, building the autodiff graph along the way |
| `atome_params(model)` | All trainable Nodes (weights + biases) in the model, for the optimizer |
| `atome_predict_class(logitsTensor)` | Shortcut for `tensor_argmax` — the predicted class index (or an array of indices for a batch) |

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

| Function | Description |
|---|---|
| `atome_save(model, path)` | Writes the model's Dense layer weights/biases to an `.atm` file (JSON) |
| `atome_load(path)` | Rebuilds a model from an `.atm` file |

An `.atm` file looks like this (`atome-v1` format):
```json
{
  "format": "atome-v1",
  "layers": [
    { "type": "dense", "W": [[0.1, -0.2], [0.3, 0.4]], "b": [0.0, 0.0] },
    { "type": "tanh" },
    { "type": "dense", "W": [[0.5], [-0.1]], "b": [0.0] }
  ]
}
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
| `torch.save(model, path)` | `atome_save(model, path)` | Simple JSON format, not PyTorch's pickle format |
| `model.to("cuda")` | `atome_set_device("gpu")` | **Does nothing real** — prints a warning and stays on CPU, see below |
| `torch.cuda.is_available()` | *(nothing)* | No GPU reachable from TON618 — see Limitations |
| `nn.Conv2d`, `nn.LSTM`, `nn.MultiheadAttention`, `nn.BatchNorm*` | *(nothing)* | Not implemented — see Limitations |
| Full NumPy broadcasting | *(nothing, except `atome_tensor_add_bias`)* | Not implemented — see Limitations |
| GPU execution (CUDA/cuDNN) | *(nothing)* | Impossible in pure TON618 — see Limitations |

---

### Honest limitations, in detail

**No GPU, and it can't be added by a module.** Running computation on a GPU requires a real driver/binding (CUDA, OpenCL, Vulkan compute...) compiled into the interpreter itself — a `.ton` file can't call native code beyond what the interpreter already exposes. `atome_set_device("gpu")` exists so the API resembles PyTorch's, but it prints a warning and continues on CPU — never a silent pretense.

**1D and 2D only.** No 3D+ tensors — so no image batches (which would be 4D: batch × channels × height × width), no attention-head-style tensors. For your own use, you can always flatten/reshape into 2D yourself if your case allows it.

**Minimal broadcasting.** `tensor + number` works. `tensor + tensor-of-different-shape` doesn't, except for the explicit `atome_tensor_add_bias` case (1D bias added to every row of a 2D matrix) — because that's exactly what a Dense layer needs, and nothing more general was built.

**Faster since beta-1.0.6, but still far from a real numerical library.** Since `ton618` beta-1.0.6, the interpreter ships a native `ton.tensor` module — the heavy operations (elementwise, matmul, activations) run as real C++ loops instead of interpreted TON618 loops. `atome` relies on it internally (same public API, zero code changes for you). Measured speedup on the provided examples: XOR training went from ~6.3s to ~1.1s, and the classification example from ~6.8s to ~0.4s — 5 to 16x faster. It's still a dynamically-typed `Value` array traversed in C++, not a contiguous vectorized BLAS-style array — so still far from the performance of a real numerical library, but noticeably more usable for experimentation than before.

**The tokenizer is word-by-word, not BPE.** A real BPE (byte-pair encoding) trainer builds its vocabulary by iteratively merging the most frequent subword pairs over an entire corpus — a substantially more complex algorithm than counting words. What's here works and is verified, but it's not what a real modern LLM tokenizer does.

**`.atm` loses a bit of precision.** The format is plain JSON (readable, debuggable) rather than a binary format — but TON618's `json_stringify`/`json_pretty` only write about 6 significant digits per number. Saving then reloading a model therefore introduces a tiny bit of numerical noise on the weights. Negligible on a small toy model; worth knowing if you save/reload in a loop.

**Conv1D exists but remains limited: a single input and output channel.** A real convolution layer stacks multiple channels (`nn.Conv1d(inChannels, outChannels, K)`); here, `atome_conv1d` only handles one input channel and one output channel. Still real and verified (numerical gradient check + training that exactly recovers the `[-1, 1]` kernel), but for a real multi-channel network you'd need to compose several `atome_conv1d` layers yourself (one per channel pair) and sum the outputs. No Conv2D, no RNN, no attention, no BatchNorm — feasible on top of this skeleton, but not added here, to stay within a scope that's actually tested rather than stacking unverified features.

**Note on "really like PyTorch"**: `ton.tensor` (beta-1.0.6+) closes a real part of the gap — the hot loops are now native, with a measured 5 to 16x speedup. What it still doesn't change: no GPU (no CUDA/OpenCL binding, impossible to add from a `.ton` module), no contiguous vectorized array (`Value` stays dynamically typed even in C++), no JIT compilation of the computation graph. Fully closing the gap would still take more interpreter work — real BLAS/GPU bindings — but this is already no longer "just interpreted TON618 loops" like in the first version of this module.

---

### How this was verified

Nothing here shipped as "looks about right" without a real test against the `ton618` interpreter:

- **Tensors**: every operation (matmul, transpose, add/sub/mul with tensor-tensor and tensor-scalar, reductions, relu, softmax summing to 1, add_bias) checked against hand-computed results.
- **Autodiff**: validated by actually training the classic XOR problem (2 inputs → 8 hidden neurons (tanh) → 1 output (sigmoid), MSE loss, plain SGD): the loss drops from 0.249 to 0.0008 over 2000 epochs, and all 4 final predictions are correct. This is the standard test to verify that a backpropagation implementation is *actually* correct, not just that it runs without crashing.
- **Multi-class classification + Adam**: validated by training a model to separate 3 2D clusters with softmax + cross-entropy and the Adam optimizer: the loss drops from 1.46 to 0.00008 over 300 epochs, with 18/18 final accuracy.
- **Tokenizer**: vocabulary ordering and the tokenize/detokenize round trip checked against hand-computed word frequencies.
- **Save/load**: verified that the reloaded model produces an output nearly identical (to the JSON precision documented above) to the original model's.
- **Migration to `ton.tensor` (beta-1.0.6)**: the four tests above were fully re-run after switching to the native backend — the exact same numerical results (same XOR predictions, same loss trajectory, same classification accuracy), only the speed changed (measured: 5 to 16x faster).
- **Conv1D**: verified two independent ways — a **numerical gradient check** (`examples/gradient_check.ton`, finite differences compared against the analytical gradient, maximum difference shown as 0.000000) confirming the backward pass is mathematically correct, then **real training** (`examples/train_conv1d.ton`) where the layer learns, from simple examples, the exact `[-1, 1]` kernel of the "discrete difference" operator — proof that beyond being correct, it actually learns something.
- **Dropout**: verified that roughly a `p` fraction of elements are zeroed during training, that survivors are scaled by `1/(1-p)`, and that eval mode gives back exactly the original input.
