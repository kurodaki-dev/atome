# atome — documentation complète

*[Read this in English](DOCUMENTATION.md)*

### Table des matières
1. [Ce que c'est, et ce que ce n'est pas](#ce-que-cest-et-ce-que-ce-nest-pas)
2. [Installation](#installation)
3. [Concepts](#concepts)
4. [Référence API — tenseurs](#référence-api--tenseurs)
5. [Référence API — autodiff](#référence-api--autodiff)
6. [Référence API — couches et modèle](#référence-api--couches-et-modèle)
7. [Référence API — attention et Transformer](#référence-api--attention-et-transformer)
8. [Référence API — pertes](#référence-api--pertes)
9. [Référence API — optimiseurs](#référence-api--optimiseurs)
10. [Référence API — sauvegarde/chargement (.atm)](#référence-api--sauvegardechargement-atm)
11. [Référence API — tokenizer](#référence-api--tokenizer)
12. [Référence API — device (GPU/CPU)](#référence-api--device-gpucpu)
13. [Correspondance avec PyTorch](#correspondance-avec-pytorch)
14. [Limites honnêtes, en détail](#limites-honnêtes-en-détail)
15. [Comment ça a été vérifié](#comment-ça-a-été-vérifié)

---

### Ce que c'est, et ce que ce n'est pas

`atome` est un **module TON618 ordinaire** — un seul fichier `.ton`, sans aucune modification de l'interpréteur (les seules dépendances sont les modules officiels `ton.tensor`, `ton.mathutils` et `ton.json`). Il donne :
- des tenseurs 1D/2D avec les opérations de base (élément par élément, matmul, réductions, activations),
- un vrai moteur d'autodiff par rétropropagation, construit de zéro (même principe que [micrograd](https://github.com/karpathy/micrograd)),
- des couches Dense (fully-connected), Conv1D (un canal), Dropout, **Multi-Head Attention**, **LayerNorm**, **Embedding**, un conteneur Sequential, et un modèle **Transformer causal complet (style GPT)**,
- deux optimiseurs (SGD, Adam),
- une perte de régression (MSE) et une perte de classification (softmax + cross-entropy fusionnées),
- un tokenizer mot-par-mot avec vocabulaire par fréquence,
- la sauvegarde/le chargement de modèles au format `.atm`, en **binaire réel** (octets IEEE-754 bruts, pas du JSON) depuis que `ton618` (beta-1.0.10+) fournit `tensor_to_bytes`/`tensor_from_bytes`.

Ce n'est **pas** un PyTorch réécrit. Pas de GPU, pas de vraie diffusion (broadcasting) façon NumPy au-delà d'un scalaire, pas de dimension batch pour l'attention (une séquence à la fois), pas de Conv2D/RNN/BatchNorm, pas de compilation ni de vectorisation — juste des boucles TON618 (et, pour les opérations tensorielles lourdes, des boucles C++ natives via `ton.tensor`). Voir [Limites honnêtes](#limites-honnêtes-en-détail) pour le détail complet de pourquoi, pas juste la liste.

---

### Installation

```bash
ton618 install atome
```

une fois le module publié dans le registre TON618 ([voir la doc du langage](https://github.com/kurodaki-dev/ton618)). En attendant, place `atome.ton` dans le dossier `modules/` à côté de ton script, ou dans le même dossier que lui, puis :

```
IMPORT://atome
```

---

### Concepts

**Tenseur** — un `ton.dict` `{shape: [...], data: [...]}` où `data` est un tableau plat (ordre ligne par ligne pour le 2D). C'est la structure de données brute, sans gradient.

**Node** — un tenseur *avec* gradient et historique : `{id, data, grad, parents, backward, op}`. C'est ce que manipulent les fonctions d'autodiff (`atome_add`, `atome_matmul`, `atome_relu`, ...). Le graphe de calcul se construit tout seul au fur et à mesure que tu enchaînes des opérations sur des Nodes — exactement comme le `requires_grad=True` implicite de PyTorch, sauf qu'ici *tout* Node accumule un gradient (pas de distinction "leaf"/"non-leaf" avec drapeau).

**Modèle** — un `ton.dict` `{type: "sequential", layers: [...]}`. Chaque couche est elle-même un dict (`{type: "dense", W: Node, b: Node}` ou juste `{type: "relu"}` pour une activation).

Comme TON618 n'a pas de classes, tout ce qui ressemblerait à une "méthode" (`layer.forward(x)`, `optimizer.step()`) est une fonction qui prend l'objet en premier argument (`atome_dense_forward(layer, x)`, `atome_adam_step(opt)`) — le style le plus proche possible d'une API orientée objet sans avoir de vraies classes.

---

### Référence API — tenseurs

| Fonction | Description |
|---|---|
| `atome_tensor_zeros(shape)` / `atome_tensor_ones(shape)` / `atome_tensor_full(shape, v)` | Nouveau tenseur rempli |
| `atome_tensor_random(shape, lo, hi)` | Nouveau tenseur rempli de valeurs aléatoires uniformes dans `[lo, hi)` |
| `atome_tensor_size(shape)` | Nombre total d'éléments pour une forme donnée |
| `atome_tensor_from_array(arr)` | Construit un tenseur depuis un tableau plat (1D) ou un tableau de tableaux (2D) |
| `atome_tensor_to_array(t)` | L'inverse — un `ton.array` classique |
| `atome_tensor_clone(t)` | Copie profonde (sinon, `t["data"]` est une référence partagée comme n'importe quel array TON618) |
| `atome_tensor_reshape(t, nouvelleForme)` | Même données, forme différente (le nombre d'éléments doit correspondre) |
| `atome_flat_index(shape, indices)` | Convertit des indices `[i]` ou `[ligne, colonne]` en index dans le tableau plat |
| `atome_tensor_get(t, indices)` / `atome_tensor_set(t, indices, v)` | Lit/écrit un élément |
| `atome_tensor_add/sub/mul/div(a, b)` | Élément par élément ; un côté peut être un nombre simple |
| `atome_tensor_add_bias(mat, bias)` | Ajoute un vecteur 1D à chaque ligne d'une matrice 2D — la seule exception à "pas de diffusion" |
| `atome_tensor_matmul(a, b)` | Multiplication matricielle 2D |
| `atome_tensor_transpose(t)` | Transposition 2D |
| `atome_tensor_sum/mean/max/min(t)` | Réduit tous les éléments à un seul nombre |
| `atome_tensor_relu/sigmoid/tanh(t)` | Activations élément par élément |
| `atome_tensor_softmax(t)` | Softmax sur tout le vecteur (1D) ou ligne par ligne (2D) |
| `atome_tensor_map(t, fn)` | Applique n'importe quelle `ton.function` élément par élément |
| `atome_tensor_to_bytes(t)` / `atome_tensor_from_bytes(octets, forme)` | Empaquette/décode les données d'un tenseur en octets bruts IEEE-754 float64 (ton618 beta-1.0.10+) — voir [Sauvegarde/chargement](#référence-api--sauvegardechargement-atm) |
| `atome_one_hot(labels, numClasses)` | Encode un tableau d'étiquettes entières en tenseur one-hot `[n, numClasses]` |

---

### Référence API — autodiff

| Fonction | Description |
|---|---|
| `atome_node(tensor)` | Enveloppe un tenseur en Node feuille : `{id, data, grad, parents, backward, op}` |
| `atome_add(a, b)` / `atome_mul(a, b)` | Opérations élément par élément entre deux Nodes, enregistrées dans le graphe |
| `atome_matmul(a, b)` | Multiplication matricielle entre deux Nodes |
| `atome_add_bias(matNode, biasNode)` | Ajout de biais entre deux Nodes (le gradient du biais est sommé sur les lignes) |
| `atome_relu(node)` / `atome_sigmoid(node)` / `atome_tanh(node)` | Activations entre Nodes |
| `atome_transpose(node)` | Transposition 2D entre Nodes |
| `atome_scale(node, k)` | Multiplie par une constante `k` non-entraînable (à ne pas confondre avec `atome_mul`, qui multiplie deux Nodes entre eux élément par élément) |
| `atome_softmax(node)` | Softmax ligne par ligne (2D) entre Nodes — le gradient est le produit Jacobien-vecteur du softmax, calculé ligne par ligne |
| `atome_concat_cols(nodes)` | Concatène des Nodes 2D côte à côte selon les colonnes (même nombre de lignes) — la passe arrière redistribue simplement chaque tranche de colonnes du gradient vers le Node d'origine |
| `atome_backward(lossNode)` | Lance la rétropropagation complète (tri topologique + parcours inverse) |
| `atome_zero_grad(node)` / `atome_zero_grads(params)` | Remet `.grad` à zéro pour un Node (ou une liste) — à appeler avant chaque `atome_backward` |

**Écrire ta propre opération différentiable** — le principe est toujours le même :
```
ton.function atome_ma_fonction(a) {
    ton.dict out = atome_node(/* calcule le tenseur résultat à partir de a["data"] */)
    out["parents"] = [a]
    out["backward"] = ton.function(self) {
        // calcule le gradient local et l'ajoute dans a["grad"]
        a["grad"] = atome_tensor_add(a["grad"], /* ... */)
    }
    return out
}
```

---

### Référence API — couches et modèle

| Fonction | Description |
|---|---|
| `atome_dense(inFeatures, outFeatures)` | Couche fully-connected, initialisée aléatoirement (échelle `1/sqrt(inFeatures)`) |
| `atome_dense_forward(layer, xNode)` | Passe avant d'une seule couche Dense — normalement tu n'appelles pas ça directement, `atome_forward` le fait |
| `atome_conv1d(kernelSize, stride)` | Couche de convolution 1D, **un seul canal d'entrée et un seul canal de sortie** (pas de multi-canaux — voir Limites) ; s'appuie sur `tensor_conv1d` (ton618 beta-1.0.7+) |
| `atome_conv1d_forward(layer, xNode)` | Passe avant d'une couche Conv1D — comme pour Dense, normalement appelée via `atome_forward` |
| `atome_dropout_layer(p)` | Couche Dropout inversé : met à zéro chaque élément avec probabilité `p` pendant l'entraînement (et met à l'échelle les survivants par `1/(1-p)`), identité pendant l'évaluation |
| `atome_set_training(bool)` / `atome_is_training()` | Bascule le mode entraînement/évaluation (affecte Dropout) — `true` par défaut |
| `atome_layernorm(features)` / `atome_layernorm_forward(layer, xNode)` | Normalise chaque ligne (dernière dimension) à moyenne nulle/variance unité, puis applique une échelle (`gamma`) et un décalage (`beta`) appris par caractéristique — voir [Référence API — attention et Transformer](#référence-api--attention-et-transformer) |
| `atome_embedding(vocabSize, dModel)` / `atome_embedding_forward(layer, ids)` | Table de correspondance entraînable — voir [Référence API — attention et Transformer](#référence-api--attention-et-transformer) |
| `atome_relu_layer()` / `atome_sigmoid_layer()` / `atome_tanh_layer()` | "Couches" d'activation, à insérer dans un Sequential |
| `atome_sequential(layers)` | Enchaîne une liste de couches en un modèle |
| `atome_forward(model, xNode)` | Passe avant complète, construit le graphe d'autodiff au fur et à mesure |
| `atome_params(model)` | Tous les Nodes entraînables (poids + biais) du modèle, pour l'optimiseur |
| `atome_predict_class(logitsTensor)` | Raccourci pour `tensor_argmax` — l'index de la classe prédite (ou un tableau d'indices pour un batch) |

---

### Référence API — attention et Transformer

`ton.tensor` est limité au 2D (voir ses propres limites honnêtes), donc il n'y a **pas de dimension batch** ici : une seule séquence à la fois, sous forme de Node `(seqLen, dModel)` — les lignes sont les positions de la séquence, les colonnes les caractéristiques. Ça tombe bien : c'est exactement la forme dont l'attention multi-têtes a besoin. Par tête, Q/K/V sont `(seqLen, headDim)`, la matrice d'attention `Q @ K^T` est `(seqLen, seqLen)`, et le softmax ligne par ligne (déjà le comportement de `tensor_softmax` en 2D) correspond exactement à "softmax sur les positions attendues, pour chaque position de requête". Donc aucun besoin de tenseurs 3D/4D — tout est construit avec des opérations 2D déjà existantes (`atome_matmul`, `atome_transpose`, `atome_softmax`, `atome_concat_cols`). Pour traiter un batch de séquences, boucle en TON618 et appelle la passe avant une fois par séquence.

| Fonction | Description |
|---|---|
| `atome_layernorm(features)` | Crée une couche LayerNorm avec `gamma` initialisé à 1 et `beta` à 0 (tous deux de taille `features`, entraînables) |
| `atome_layernorm_forward(layer, xNode)` | Normalise chaque ligne de `xNode` (moyenne 0, variance 1), puis `xhat * gamma + beta`. La passe arrière est la formule fermée classique de LayerNorm (dérivée à la main, vérifiée par test de gradient numérique) |
| `atome_embedding(vocabSize, dModel)` | Crée une table de correspondance `(vocabSize, dModel)` entraînable, initialisée aléatoirement |
| `atome_embedding_forward(layer, ids)` | Récupère une ligne de la table par identifiant entier dans `ids`, construisant un Node `(len(ids), dModel)`. La passe arrière **accumule** (`+=`) le gradient dans les lignes correspondantes de la table — un identifiant répété accumule bien son gradient plusieurs fois, comme une vraie couche d'embedding |
| `atome_causal_mask(seqLen)` | Construit un tenseur additif `(seqLen, seqLen)` : `0` là où la position `i` peut attendre la position `j` (`j <= i`), un grand nombre négatif sinon — à ajouter aux scores d'attention bruts avant le softmax |
| `atome_multihead_attention(dModel, numHeads)` | Crée une couche d'attention multi-têtes : `dModel` doit être divisible par `numHeads`. Pas de biais sur les projections Q/K/V (comme GPT-2), mais la projection de sortie `Wo` est une vraie couche Dense (avec biais) |
| `atome_multihead_attention_forward(layer, xNode, mask = nil)` | Passe avant complète : pour chaque tête, `Q=x@Wq`, `K=x@Wk`, `V=x@Wv`, `scores = (Q @ K^T) / sqrt(headDim)`, `+ mask` si fourni, `softmax`, `@ V` — puis les têtes sont recombinées avec `atome_concat_cols` et projetées par `Wo`. Passe `atome_causal_mask(seqLen)` en `mask` pour de l'attention causale (auto-régressive, style GPT) ; `nil` (par défaut) pour de l'attention bidirectionnelle non masquée |
| `atome_transformer_block(dModel, numHeads, dFF)` | Un bloc Transformer complet : attention multi-têtes + LayerNorm + feed-forward (Dense → ReLU → Dense) + LayerNorm |
| `atome_transformer_block_forward(block, xNode, mask = nil)` | Passe avant post-norm classique : `LN(x + Attention(x))` puis `LN(norm1 + FFN(norm1))` |
| `atome_transformer_lm(vocabSize, dModel, numHeads, dFF, numLayers, maxSeqLen)` | Un petit modèle de langage décodeur-seul (style GPT) : embedding de tokens + embedding de position (appris), `numLayers` blocs Transformer **masqués causalement**, puis une projection Dense vers des logits de taille `vocabSize` |
| `atome_transformer_lm_forward(model, tokenIds)` | Passe avant sur **une seule séquence** (`tokenIds` est un tableau d'entiers, pas un batch) — renvoie un Node `(seqLen, vocabSize)` de logits bruts, prêt pour `atome_softmax_cross_entropy_loss` ou `atome_predict_class` (ligne par ligne) |
| `atome_transformer_lm_params(model)` | Tous les Nodes entraînables du modèle (embeddings, tous les blocs, projection de sortie), pour l'optimiseur |

```
IMPORT://atome

ton.dict model = atome_transformer_lm(10, 16, 2, 32, 2, 16)  // vocab=10, dModel=16, 2 têtes, dFF=32, 2 blocs, maxSeqLen=16
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

Voir `examples/train_transformer.ton` (entraînement réel jusqu'à 16/16 de précision sur une tâche de prédiction du caractère suivant) et `examples/gradient_check_transformer.ton` (vérification de gradient numérique du bloc complet).

---

### Référence API — pertes

| Fonction | Description |
|---|---|
| `atome_mse_loss(predNode, targetTensor)` | Erreur quadratique moyenne — pour de la régression ou de la classification binaire (avec sigmoid en sortie) |
| `atome_softmax_cross_entropy_loss(logitsNode, targetOneHot)` | Softmax + cross-entropy fusionnées, pour de la classification multi-classes — prend les **logits bruts** (pas déjà passés au softmax) et une cible one-hot (voir `atome_one_hot`) |

---

### Référence API — optimiseurs

| Fonction | Description |
|---|---|
| `atome_sgd_step(params, lr)` | Descente de gradient simple : `param.data -= lr * param.grad` |
| `atome_adam_init(params, lr)` | Crée l'état de l'optimiseur Adam (moments d'ordre 1 et 2, avec correction de biais) — `beta1=0.9`, `beta2=0.999`, `eps=1e-8` |
| `atome_adam_step(opt)` | Un pas d'Adam sur l'état créé par `atome_adam_init` |

Dans les deux cas, appelle `atome_zero_grads(params)` avant `atome_backward` à chaque itération — les gradients s'accumulent (`+=`) sur les Nodes de paramètres, donc sans remise à zéro ils s'additionneraient d'une itération à l'autre.

---

### Référence API — sauvegarde/chargement (.atm)

Depuis `ton618` beta-1.0.10, `.atm` est un **vrai format binaire** — plus du JSON. Ça règle le seul vrai défaut de l'ancien format : `json_stringify`/`json_pretty` n'écrivaient qu'environ 6 chiffres significatifs par nombre, donc sauvegarder/recharger un modèle introduisait un petit bruit numérique sur les poids. En binaire, chaque nombre est stocké comme un float64 IEEE-754 brut (8 octets) — **aucune perte de précision**, et le fichier est aussi plus compact.

| Fonction | Description |
|---|---|
| `atome_save(model, chemin)` | Écrit un modèle `atome_sequential` (Dense/Conv1D/Dropout/activations) dans un fichier `.atm` binaire |
| `atome_load(chemin)` | Reconstruit un modèle `atome_sequential` depuis un fichier `.atm` |
| `atome_transformer_lm_save(model, chemin)` | Écrit un `atome_transformer_lm` complet (tous ses blocs, embeddings, projection de sortie) dans un fichier `.atm` binaire |
| `atome_transformer_lm_load(chemin)` | Reconstruit un `atome_transformer_lm` depuis un fichier `.atm` — reconstruit d'abord le squelette avec `atome_transformer_lm(...)` (mêmes hyperparamètres que ceux sauvegardés), puis écrase chaque paramètre avec les octets chargés |

**Format du fichier** (identique pour `atome_save`/`atome_transformer_lm_save`, seul le contenu du header JSON diffère) :

| Octets | Contenu |
|---|---|
| `0..8` | La longueur du header JSON qui suit, encodée comme un seul float64 (8 octets, via `tensor_to_bytes`) |
| `8..8+longueur` | Un header JSON **compact** (texte UTF-8) décrivant l'architecture — types de couches et tailles, ou hyperparamètres du Transformer. Ne contient **jamais** de valeurs de poids |
| après le header | Les octets bruts de chaque tenseur entraînable, concaténés dans un ordre fixe (le même ordre que `atome_params`/`atome_transformer_lm_params`), chacun via `tensor_to_bytes` |

C'est le même principe qu'un vrai fichier de modèle (un petit index de métadonnées + un blob de données tensorielles brutes) — juste sans l'étape ZIP/pickle de PyTorch. Pas portable entre machines d'endianness différente (comme `tensor_to_bytes` lui-même). Si tu dois lire un `.atm` depuis autre chose que `atome`, lis les 8 premiers octets comme un float64 pour connaître la longueur du header, parse le JSON qui suit, puis lis le reste comme des blocs de float64 bruts dans l'ordre documenté ci-dessus.

---

### Référence API — tokenizer

Un tokenizer mot-par-mot simple : découpe sur les espaces, met en minuscules, construit un vocabulaire trié par fréquence décroissante (les mots les plus fréquents d'abord). Pas de BPE — voir [Limites honnêtes](#limites-honnêtes-en-détail).

| Fonction | Description |
|---|---|
| `atome_tokenizer_fit(texts, maxVocabSize)` | Construit un vocabulaire à partir d'un tableau de textes : découpe chaque texte en mots (minuscules), compte les fréquences, garde les `maxVocabSize - 2` mots les plus fréquents. Réserve toujours l'identifiant `0` pour `<UNK>` (mot inconnu) et `1` pour `<PAD>`. Renvoie `{vocab: {mot: id}, idToToken: [mot, ...]}` |
| `atome_tokenize(tok, text)` | Découpe `text` en mots et les convertit en identifiants via `tok["vocab"]` — un mot absent du vocabulaire devient `<UNK>` (id `0`) |
| `atome_detokenize(tok, ids)` | L'inverse : convertit un tableau d'identifiants en mots (via `tok["idToToken"]`) et les rejoint avec des espaces |

```
IMPORT://atome

ton.dict tok = atome_tokenizer_fit(["the cat sat on the mat", "the dog sat on the log"], 20)
ton.array ids = atome_tokenize(tok, "the cat sat on a rug")
print(ids)                        // [2, 5, 3, 4, 0, 0] par exemple — "a"/"rug" sont <UNK>
print(atome_detokenize(tok, ids)) // "the cat sat on <UNK> <UNK>"
```

---

### Référence API — device (GPU/CPU)

| Fonction | Description |
|---|---|
| `atome_set_device(nom)` | `"cpu"` (réel) ou `"gpu"` (accepté, mais retombe toujours sur cpu avec un avertissement affiché) |
| `atome_get_device()` | Renvoie toujours `"cpu"` |

Voir [Limites honnêtes](#limites-honnêtes-en-détail) pour pourquoi le GPU n'est pas — et ne peut pas être — réellement supporté.

---

### Correspondance avec PyTorch

Pour s'y retrouver si tu connais déjà PyTorch :

| PyTorch | atome | Différence |
|---|---|---|
| `torch.tensor(...)` | `atome_tensor_from_array(...)` | 1D/2D seulement |
| `x.requires_grad_()` puis les ops dessus | `atome_node(x)` puis les ops `atome_*` dessus | Pas de drapeau — tout Node accumule un gradient |
| `loss.backward()` | `atome_backward(loss)` | Identique dans l'esprit |
| `nn.Linear(in, out)` | `atome_dense(in, out)` | Identique dans l'esprit |
| `nn.Conv1d(1, 1, K, stride=S)` | `atome_conv1d(K, S)` | Un seul canal d'entrée et de sortie seulement — voir Limites |
| `nn.Dropout(p)` | `atome_dropout_layer(p)` | Identique dans l'esprit (dropout inversé) |
| `model.train()` / `model.eval()` | `atome_set_training(true)` / `atome_set_training(false)` | Un seul interrupteur global plutôt qu'un état par modèle |
| `nn.Sequential(...)` | `atome_sequential([...])` | Identique dans l'esprit |
| `nn.ReLU()` / `nn.Sigmoid()` / `nn.Tanh()` | `atome_relu_layer()` / `atome_sigmoid_layer()` / `atome_tanh_layer()` | Identique dans l'esprit |
| `model.parameters()` | `atome_params(model)` | Identique dans l'esprit |
| `optim.SGD(params, lr)` + `.step()` | `atome_sgd_step(params, lr)` | Pas d'objet séparé, un appel direct |
| `optim.Adam(params, lr)` + `.step()` | `atome_adam_init(params, lr)` puis `atome_adam_step(opt)` | Deux appels au lieu d'un objet avec méthode |
| `optimizer.zero_grad()` | `atome_zero_grads(params)` | Identique dans l'esprit |
| `nn.MSELoss()` | `atome_mse_loss(pred, target)` | Identique dans l'esprit |
| `nn.CrossEntropyLoss()` | `atome_softmax_cross_entropy_loss(logits, targetOneHot)` | PyTorch prend des indices de classe entiers ; ici il faut construire le one-hot toi-même avec `atome_one_hot` |
| `logits.argmax(dim=-1)` | `atome_predict_class(logitsTensor)` | Identique dans l'esprit |
| `nn.LayerNorm(features)` | `atome_layernorm(features)` | Identique dans l'esprit |
| `nn.Embedding(vocab, dModel)` | `atome_embedding(vocab, dModel)` | Identique dans l'esprit |
| `nn.MultiheadAttention(dModel, numHeads)` | `atome_multihead_attention(dModel, numHeads)` | Une seule séquence à la fois (pas de dimension batch) — voir Limites |
| `nn.TransformerEncoderLayer(...)` | `atome_transformer_block(dModel, numHeads, dFF)` | Bloc post-norm classique (attention + LN + FFN + LN) |
| GPT-2/nn.Transformer en mode décodeur-seul | `atome_transformer_lm(vocab, dModel, numHeads, dFF, numLayers, maxSeqLen)` | Un vrai petit modèle de langage causal, entraîné et vérifié — voir [Comment ça a été vérifié](#comment-ça-a-été-vérifié) |
| `torch.save(model, path)` | `atome_save(model, path)` / `atome_transformer_lm_save(model, path)` | Vrai format binaire (float64 IEEE-754 bruts) depuis ton618 beta-1.0.10, mais pas le format pickle/ZIP de PyTorch |
| `model.to("cuda")` | `atome_set_device("gpu")` | **Ne fait rien de réel** — affiche un avertissement et reste sur CPU, voir ci-dessous |
| `torch.cuda.is_available()` | *(rien)* | Pas de GPU accessible depuis TON618 — voir Limites |
| `nn.Conv2d`, `nn.LSTM`, `nn.BatchNorm*`, KV-cache pour la génération | *(rien)* | Pas implémenté — voir Limites |
| Broadcasting NumPy complet | *(rien, sauf `atome_tensor_add_bias`)* | Pas implémenté — voir Limites |
| Exécution GPU (CUDA/cuDNN) | *(rien)* | Impossible en TON618 pur — voir Limites |

---

### Limites honnêtes, en détail

**Pas de GPU, et ça ne peut pas être ajouté par un module.** Exécuter du calcul sur GPU demande un vrai pilote/binding (CUDA, OpenCL, Vulkan compute...) compilé dans l'interpréteur lui-même — un fichier `.ton` ne peut pas appeler de code natif en dehors de ce que l'interpréteur expose déjà. `atome_set_device("gpu")` existe pour que l'API ressemble à celle de PyTorch, mais il affiche un avertissement et continue sur CPU — jamais de faux-semblant silencieux.

**1D et 2D seulement.** Pas de tenseurs 3D+ — donc pas de batchs d'images (qui seraient 4D : batch × canaux × hauteur × largeur), pas de tenseurs façon "têtes d'attention". Pour ton propre usage, tu peux toujours aplatir/reshaper en 2D toi-même si ton cas s'y prête.

**Diffusion (broadcasting) minimale.** `tenseur + nombre` marche. `tenseur + tenseur-de-forme-différente` ne marche pas, sauf le cas explicite `atome_tensor_add_bias` (biais 1D ajouté à chaque ligne d'une matrice 2D) — parce que c'est exactement ce dont une couche Dense a besoin, et rien de plus général n'a été construit.

**Plus rapide depuis beta-1.0.6, mais toujours loin d'une vraie bibliothèque numérique.** Depuis `ton618` beta-1.0.6, l'interpréteur fournit un module natif `ton.tensor` — les opérations lourdes (élément par élément, matmul, activations) tournent en vraies boucles C++ plutôt qu'en boucles TON618 interprétées. `atome` s'appuie dessus en interne (même API publique, zéro changement de code pour toi). Gain mesuré sur les exemples fournis : l'entraînement XOR est passé de ~6,3s à ~1,1s, et l'exemple de classification de ~6,8s à ~0,4s — 5 à 16x plus rapide. Ça reste un tableau `Value` typé dynamiquement parcouru en C++, pas un tableau contigu vectorisé façon BLAS — donc toujours très loin des performances d'une vraie bibliothèque numérique, mais nettement plus utilisable qu'avant pour de l'expérimentation.

**Le tokenizer est mot-par-mot, pas BPE.** Un vrai entraîneur BPE (byte-pair encoding) construit son vocabulaire en fusionnant itérativement les paires de sous-mots les plus fréquentes sur tout un corpus — un algorithme sensiblement plus complexe que compter des mots. Ce qui est ici fonctionne et est vérifié, mais ce n'est pas ce que fait un vrai tokenizer de LLM moderne.

**`.atm` était en JSON et perdait un peu de précision — corrigé depuis ton618 beta-1.0.10.** L'ancien format écrivait les nombres en JSON texte, et `json_stringify`/`json_pretty` n'écrivent qu'environ 6 chiffres significatifs par nombre — sauvegarder puis recharger un modèle introduisait donc un tout petit bruit numérique sur les poids. Le format binaire actuel (voir [Sauvegarde/chargement](#référence-api--sauvegardechargement-atm)) stocke chaque nombre comme un float64 IEEE-754 brut : aucune perte, vérifié par un test qui sauvegarde/recharge et compare bit à bit (`avant - après == 0` exactement, pas juste "proche de zéro").

**Conv1D existe mais reste limitée : un seul canal d'entrée et de sortie.** Une vraie couche de convolution empile plusieurs canaux (`nn.Conv1d(inChannels, outChannels, K)`) ; ici, `atome_conv1d` ne gère qu'un canal en entrée et un en sortie. Toujours réel et vérifié (test de gradient numérique + entraînement qui retrouve exactement le noyau `[-1, 1]`), mais pour un vrai réseau à plusieurs canaux il faudrait composer plusieurs couches `atome_conv1d` toi-même (une par paire de canaux) et sommer les sorties. Pas de Conv2D, pas de RNN, pas de BatchNorm — faisables sur ce squelette, mais pas ajoutés ici pour rester dans un périmètre réellement testé plutôt que d'empiler des fonctionnalités non vérifiées.

**L'attention multi-têtes existe, mais sans dimension batch ni cache KV.** `atome_multihead_attention_forward`/`atome_transformer_lm_forward` traitent **une séquence à la fois** — pas de tenseur 3D `(batch, seq, dModel)`, parce que `ton.tensor` est limité au 2D (voir sa propre section Limites). Pour entraîner sur un batch, boucle en TON618 et moyenne/accumule les gradients toi-même sur plusieurs appels. Il n'y a pas non plus de cache clé/valeur pour accélérer la génération token par token (chaque nouveau token relance la passe avant sur toute la séquence depuis le début) — un vrai cache KV économiserait ce travail redondant, mais n'a pas été construit ici pour rester dans un périmètre entièrement vérifié. Le nombre de têtes doit diviser exactement `dModel` (pas de padding automatique).

**Note sur "vraiment comme PyTorch"** : `ton.tensor` (beta-1.0.6+) comble une partie réelle de l'écart — les boucles chaudes sont maintenant natives, avec un gain mesuré de 5 à 16x. Ce que ça ne change toujours pas : pas de GPU (aucun binding CUDA/OpenCL, impossible à ajouter depuis un module `.ton`), pas de tableau contigu vectorisé (`Value` reste typé dynamiquement même en C++), pas de compilation JIT du graphe de calcul. Combler complètement l'écart demanderait encore plus de travail interpréteur — des bindings BLAS/GPU réels — mais ce n'est déjà plus "juste des boucles TON618 interprétées" comme dans la première version de ce module.

---

### Comment ça a été vérifié

Rien ici n'a été livré "ça a l'air correct" sans test réel contre l'interpréteur `ton618` :

- **Tenseurs** : chaque opération (matmul, transpose, add/sub/mul avec tenseur-tenseur et tenseur-scalaire, réductions, relu, softmax qui somme à 1, add_bias) vérifiée contre des résultats calculés à la main.
- **Autodiff** : validé en entraînant réellement le problème XOR classique (2 entrées → 8 neurones cachés (tanh) → 1 sortie (sigmoid), perte MSE, SGD simple) : la loss descend de 0.249 à 0.0008 sur 2000 epochs, et les 4 prédictions finales sont toutes correctes. C'est le test standard pour vérifier qu'une rétropropagation est *réellement* juste, pas juste qu'elle s'exécute sans planter.
- **Classification multi-classes + Adam** : validé en entraînant un modèle à séparer 3 clusters 2D avec softmax + cross-entropy et l'optimiseur Adam : la loss descend de 1.46 à 0.00008 sur 300 epochs, avec 18/18 de précision finale.
- **Tokenizer** : ordre du vocabulaire et aller-retour tokenize/detokenize vérifiés contre des fréquences de mots calculées à la main.
- **Sauvegarde/chargement** : vérifié que le modèle rechargé produit une sortie **exactement identique** (différence de `0`, pas juste "proche de zéro") à celle du modèle original — pour un modèle Dense, un modèle Conv1D, et un `atome_transformer_lm` à 2 blocs, chacun sauvegardé puis rechargé depuis le disque via le format binaire.
- **LayerNorm** et **Embedding** : chacun vérifié par test de gradient numérique — LayerNorm à la fois pour le gradient de l'entrée et celui de `gamma` ; Embedding avec un identifiant volontairement répété dans la séquence de test, pour confirmer que le gradient s'accumule bien plusieurs fois sur la même ligne de la table plutôt que d'écraser.
- **Attention multi-têtes** : vérifiée par test de gradient numérique de bout en bout (avec masque causal actif), à la fois pour le gradient de l'entrée et pour le gradient d'une des matrices de projection Q — puis à nouveau à l'échelle d'un bloc Transformer complet (attention + LayerNorm + feed-forward + LayerNorm), et une troisième fois sur un `atome_transformer_lm` à 2 blocs en vérifiant le gradient de la table d'embedding de tokens à travers tout le graphe.
- **Transformer causal, entraînement réel** (`examples/train_transformer.ton`) : un petit modèle (2 blocs, 2 têtes, dModel=16) entraîné à prédire le caractère suivant dans un cycle répétitif `"abcdefgh"` — la loss descend de 2.34 à 0.0006 sur 400 pas, et le modèle atteint 16/16 de précision sur une fenêtre tenue à l'écart de l'entraînement, prédiction exacte position par position.
- **Migration vers `ton.tensor` (beta-1.0.6)** : les quatre tests ci-dessus ont été rejoués intégralement après le passage au backend natif — mêmes résultats numériques exacts (mêmes prédictions XOR, même trajectoire de loss, même précision de classification), seule la vitesse a changé (mesurée : 5 à 16x plus rapide).
- **Conv1D** : vérifiée de deux façons indépendantes — un **test de gradient numérique** (`examples/gradient_check.ton`, différences finies comparées au gradient analytique, écart maximal affiché à 0.000000) qui confirme que la passe arrière est mathématiquement correcte, puis un **entraînement réel** (`examples/train_conv1d.ton`) où la couche apprend, à partir de simples exemples, le noyau exact `[-1, 1]` de l'opérateur "différence discrète" — la preuve qu'au-delà d'être juste, ça apprend vraiment quelque chose.
- **Dropout** : vérifié qu'environ `p` fraction des éléments sont mis à zéro pendant l'entraînement, que les survivants sont mis à l'échelle par `1/(1-p)`, et que le mode évaluation redonne exactement l'entrée d'origine.
