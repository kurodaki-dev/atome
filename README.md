# atome

Un module TON618 — tenseurs, autodiff (rétropropagation), couches, optimiseurs (SGD + Adam), perte de classification, tokenizer, et sauvegarde/chargement de modèles (`.atm`). C'est un module comme un autre — installable, pas une fonctionnalité intégrée du langage.

**Nécessite `ton618` beta-1.0.7 ou plus récent.** Depuis beta-1.0.6, l'interpréteur fournit un module natif `ton.tensor` (boucles réelles en C++), et `atome` s'appuie dessus pour toutes ses opérations de calcul intensif — gain mesuré de **5 à 16x** sur les exemples fournis, sans aucun changement à l'API publique du module. Beta-1.0.7 ajoute `tensor_conv1d`/`tensor_argmax`, utilisés par la nouvelle couche **Conv1D** et `atome_predict_class`. Vérifie avec `ton618 --version` ; mets à jour avec `ton618 --update` si besoin.

**Nouveau dans cette version d'`atome`** : couche **Conv1D** (vérifiée par un test de gradient numérique *et* un entraînement réel — voir `examples/gradient_check.ton` et `examples/train_conv1d.ton`), couche **Dropout** avec mode entraînement/évaluation, et `atome_predict_class` pour lire directement la classe prédite.

**Toute la doc complète est dans [`DOCUMENTATION.md`](DOCUMENTATION.md)** — API complète, correspondance avec PyTorch, limites honnêtes, et comment tout ça a été vérifié.

## Installation

```bash
ton618 install atome        # une fois publié dans le registre TON618
```

Ou, en attendant, copie simplement `atome.ton` dans le dossier `modules/` de ton projet, puis :

```
IMPORT://atome
```

## Démarrage rapide

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
// -> environ [[~0], [~1], [~1], [~0]]
```

Quatre scripts complets et testés dans `examples/` :
- `train_xor.ton` — régression/classification binaire (le script ci-dessus)
- `train_classifier.ton` — classification multi-classes avec softmax + cross-entropy + Adam
- `save_load.ton` — sauvegarder puis recharger un modèle `.atm`
- `tokenizer.ton` — tokenisation mot-par-mot

## En une phrase sur les limites

CPU uniquement (aucun backend GPU n'existe dans TON618), tenseurs 1D/2D seulement, pas de diffusion (broadcasting) façon NumPy au-delà d'un scalaire, et c'est lent (boucles TON618 interprétées, pas de vectorisation). Voir [`DOCUMENTATION.md`](DOCUMENTATION.md) pour le détail complet et pourquoi.
