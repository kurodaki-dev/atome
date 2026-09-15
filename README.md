# atome

*[Lire en français](README.fr.md)*

A TON618 module — tensors, autodiff (backpropagation), layers (Dense, Conv1D, Dropout, **Multi-Head Attention**, **LayerNorm**, **Embedding**), a complete **causal Transformer (GPT-style)** model, optimizers (SGD + Adam), classification loss, tokenizer, and model save/load in **real binary** (`.atm`). It's a module like any other — installable, not a built-in language feature.

**Requires `ton618` beta-1.0.10 or newer.** Since beta-1.0.6, the interpreter ships a native `ton.tensor` module (real C++ loops), and `atome` relies on it for all its compute-intensive operations. Beta-1.0.7 added `tensor_conv1d`/`tensor_argmax`. Beta-1.0.10 adds `tensor_to_bytes`/`tensor_from_bytes`, which lets `atome` write real binary files. Check with `ton618 --version`; update with `ton618 --update` if needed.

**New in this version of `atome`**:
- **Multi-head attention + causal Transformer** (`atome_multihead_attention`, `atome_transformer_block`, `atome_transformer_lm`) — a real decoder-only language model (GPT-style), trained and verified end to end (see `examples/train_transformer.ton` and `examples/gradient_check_transformer.ton`).
- **LayerNorm** and **Embedding** (with correct gradient accumulation on repeated ids), each verified with a numerical gradient check.
- **`.atm` files in real binary**: `tensor_to_bytes`/`tensor_from_bytes` (ton618 beta-1.0.10) replace the old JSON format — smaller files and **zero precision loss** (the old format lost ~6 significant digits per number).

**The full documentation is in [`DOCUMENTATION.md`](DOCUMENTATION.md)** — complete API, PyTorch mapping, honest limitations, and how all of this was verified.

## Installation

```bash
ton618 install atome        # once published to the TON618 registry
```

Or, in the meantime, just copy `atome.ton` into your project's `modules/` folder, then:

```
IMPORT://atome
```

## Quick start

```
IMPORT://atome
IMPORT://ton.random

random_seed(42)

ton.dict X = atome_tensor_from_array([[0, 0], [0, 1], [1, 0], [1, 1]])
ton.dict Y = atome_tensor_from_array([[0], [1], [1], [0]])

ton.dict model = atome_sequential([
    atome_dense(2, 8),
    atome_tanh_layer(),
    atome_dense(8, 1),
    atome_sigmoid_layer()
])
ton.array params = atome_params(model)

ton.float lr = 0.5
for ton.int epoch = 0; epoch < 2000; epoch++ {
    atome_zero_grads(params)
    ton.dict pred = atome_forward(model, atome_node(X))
    ton.dict loss = atome_mse_loss(pred, Y)
    atome_backward(loss)
    atome_sgd_step(params, lr)
}

print(atome_tensor_to_array(atome_forward(model, atome_node(X))["data"]))
// -> roughly [[~0], [~1], [~1], [~0]]
```

## Causal Transformer in one minute

```
IMPORT://atome

ton.dict model = atome_transformer_lm(
    10,   // vocabSize
    16,   // dModel
    2,    // numHeads
    32,   // dFF (feed-forward hidden size)
    2,    // numLayers
    16    // maxSeqLen
)
ton.array params = atome_transformer_lm_params(model)
ton.dict opt = atome_adam_init(params, 0.01)

ton.array inputIds = [1, 2, 3, 4, 5, 6, 7, 0]
ton.array targetIds = [2, 3, 4, 5, 6, 7, 0, 1]  // "predict the next token"

atome_zero_grads(params)
ton.dict logits = atome_transformer_lm_forward(model, inputIds)  // (seqLen, vocabSize)
ton.dict loss = atome_softmax_cross_entropy_loss(logits, atome_one_hot(targetIds, 10))
atome_backward(loss)
atome_adam_step(opt)

atome_transformer_lm_save(model, "my_model.atm")
```

`atome_transformer_lm_forward` processes **one sequence at a time** (no batch dimension — `ton.tensor` is limited to 2D, so the tensor's rows *are* the sequence positions; loop in TON618 to process several sequences). The causal mask is applied automatically (position *i* never sees positions after it).

Seven complete, tested scripts in `examples/`:
- `train_xor.ton` — binary regression/classification
- `train_classifier.ton` — multi-class classification with softmax + cross-entropy + Adam
- `save_load.ton` — save then reload an `.atm` model (Dense/Conv1D)
- `tokenizer.ton` — word-by-word tokenization
- `gradient_check.ton` / `train_conv1d.ton` — Conv1D verified and trained
- `gradient_check_transformer.ton` — gradient check of the full Transformer block (attention + LayerNorm + feed-forward)
- `train_transformer.ton` — trains a real small causal Transformer to predict the next character (16/16 accuracy on a held-out example)
- `save_load_transformer.ton` — save/reload a complete `atome_transformer_lm`

## Limitations in one sentence

CPU only (no GPU backend exists in TON618), 1D/2D tensors only (so no batch dimension for attention — one sequence at a time), no NumPy-style broadcasting beyond a scalar, and it's slow (interpreted TON618 loops, no vectorization). See [`DOCUMENTATION.md`](DOCUMENTATION.md) for the full details and why.
