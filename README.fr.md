# atome

*[Read this in English](README.md)*

Un module TON618 — tenseurs, autodiff (rétropropagation), couches (Dense, Conv1D, Dropout, **Multi-Head Attention**, **LayerNorm**, **Embedding**), un modèle **Transformer causal (style GPT)** complet, optimiseurs (SGD + Adam), perte de classification, tokenizer, et sauvegarde/chargement de modèles en **binaire réel** (`.atm`). C'est un module comme un autre — installable, pas une fonctionnalité intégrée du langage.

**Nécessite `ton618` beta-1.0.10 ou plus récent.** Depuis beta-1.0.6, l'interpréteur fournit un module natif `ton.tensor` (boucles réelles en C++), et `atome` s'appuie dessus pour toutes ses opérations de calcul intensif. Beta-1.0.7 a ajouté `tensor_conv1d`/`tensor_argmax`. Beta-1.0.10 ajoute `tensor_to_bytes`/`tensor_from_bytes`, ce qui permet à `atome` d'écrire de vrais fichiers binaires. Vérifie avec `ton618 --version` ; mets à jour avec `ton618 --update` si besoin.

**Nouveau dans cette version d'`atome`** :
- **Attention multi-têtes + Transformer causal** (`atome_multihead_attention`, `atome_transformer_block`, `atome_transformer_lm`) — un vrai modèle de langage décodeur-seul (style GPT), entraîné et vérifié de bout en bout (voir `examples/train_transformer.ton` et `examples/gradient_check_transformer.ton`).
- **LayerNorm** et **Embedding** (avec accumulation correcte du gradient sur les identifiants répétés), chacun vérifié par test de gradient numérique.
- **Fichiers `.atm` en binaire réel** : `tensor_to_bytes`/`tensor_from_bytes` (ton618 beta-1.0.10) remplacent l'ancien format JSON — fichiers plus petits et **aucune perte de précision** (l'ancien format perdait ~6 chiffres significatifs par nombre).

**Toute la doc complète est dans [`DOCUMENTATION.fr.md`](DOCUMENTATION.fr.md)** — API complète, correspondance avec PyTorch, limites honnêtes, et comment tout ça a été vérifié.

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

## Transformer causal en une minute

```
IMPORT://atome

ton.dict model = atome_transformer_lm(
    10,   // vocabSize
    16,   // dModel
    2,    // numHeads
    32,   // dFF (taille cachée du feed-forward)
    2,    // numLayers
    16    // maxSeqLen
)
ton.array params = atome_transformer_lm_params(model)
ton.dict opt = atome_adam_init(params, 0.01)

ton.array inputIds = [1, 2, 3, 4, 5, 6, 7, 0]
ton.array targetIds = [2, 3, 4, 5, 6, 7, 0, 1]  // "prédis le token suivant"

atome_zero_grads(params)
ton.dict logits = atome_transformer_lm_forward(model, inputIds)  // (seqLen, vocabSize)
ton.dict loss = atome_softmax_cross_entropy_loss(logits, atome_one_hot(targetIds, 10))
atome_backward(loss)
atome_adam_step(opt)

atome_transformer_lm_save(model, "mon_modele.atm")
```

`atome_transformer_lm_forward` traite **une séquence à la fois** (pas de dimension batch — `ton.tensor` est limité au 2D, donc les lignes du tenseur *sont* les positions de la séquence ; boucle en TON618 pour traiter plusieurs séquences). Le masque causal est appliqué automatiquement (position *i* ne voit jamais les positions après elle).

Sept scripts complets et testés dans `examples/` :
- `train_xor.ton` — régression/classification binaire
- `train_classifier.ton` — classification multi-classes avec softmax + cross-entropy + Adam
- `save_load.ton` — sauvegarder puis recharger un modèle `.atm` (Dense/Conv1D)
- `tokenizer.ton` — tokenisation mot-par-mot
- `gradient_check.ton` / `train_conv1d.ton` — Conv1D vérifiée et entraînée
- `gradient_check_transformer.ton` — vérification de gradient du bloc Transformer complet (attention + LayerNorm + feed-forward)
- `train_transformer.ton` — entraîne un vrai petit Transformer causal à prédire le caractère suivant (précision 16/16 sur un exemple tenu à l'écart)
- `save_load_transformer.ton` — sauvegarder/recharger un `atome_transformer_lm` complet

## En une phrase sur les limites

CPU uniquement (aucun backend GPU n'existe dans TON618), tenseurs 1D/2D seulement (donc pas de dimension batch pour l'attention — une séquence à la fois), pas de diffusion (broadcasting) façon NumPy au-delà d'un scalaire, et c'est lent (boucles TON618 interprétées, pas de vectorisation). Voir [`DOCUMENTATION.fr.md`](DOCUMENTATION.fr.md) pour le détail complet et pourquoi.
