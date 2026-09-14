# atome

*[Lire en français](README.fr.md)*

A TON618 module — tensors, autodiff (backpropagation), layers, optimizers (SGD + Adam), classification loss, tokenizer, and model save/load (`.atm`). It's a module like any other — installable, not a built-in language feature.

**Requires `ton618` beta-1.0.7 or newer.** Since beta-1.0.6, the interpreter ships a native `ton.tensor` module (real C++ loops), and `atome` relies on it for all its compute-intensive operations — a measured **5 to 16x** speedup on the provided examples, with no change to the module's public API. Beta-1.0.7 adds `tensor_conv1d`/`tensor_argmax`, used by the new **Conv1D** layer and `atome_predict_class`. Check with `ton618 --version`; update with `ton618 --update` if needed.

**New in this version of `atome`**: **Conv1D** layer (verified with both a numerical gradient check *and* real training — see `examples/gradient_check.ton` and `examples/train_conv1d.ton`), **Dropout** layer with train/eval mode, and `atome_predict_class` to read the predicted class directly.

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

Four complete, tested scripts in `examples/`:
- `train_xor.ton` — binary regression/classification (the script above)
- `train_classifier.ton` — multi-class classification with softmax + cross-entropy + Adam
- `save_load.ton` — save then reload an `.atm` model
- `tokenizer.ton` — word-by-word tokenization

## Limitations in one sentence

CPU only (no GPU backend exists in TON618), 1D/2D tensors only, no NumPy-style broadcasting beyond a scalar, and it's slow (interpreted TON618 loops, no vectorization). See [`DOCUMENTATION.md`](DOCUMENTATION.md) for the full details and why.
