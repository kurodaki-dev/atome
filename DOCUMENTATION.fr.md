# atome — documentation complète

*[Read this in English](DOCUMENTATION.md)*

### Table des matières
1. [Ce que c'est, et ce que ce n'est pas](#ce-que-cest-et-ce-que-ce-nest-pas)
2. [Installation](#installation)
3. [Concepts](#concepts)
4. [Référence API — tenseurs](#référence-api--tenseurs)
5. [Référence API — autodiff](#référence-api--autodiff)
6. [Référence API — couches et modèle](#référence-api--couches-et-modèle)
7. [Référence API — pertes](#référence-api--pertes)
8. [Référence API — optimiseurs](#référence-api--optimiseurs)
9. [Référence API — sauvegarde/chargement (.atm)](#référence-api--sauvegardechargement-atm)
10. [Référence API — device (GPU/CPU)](#référence-api--device-gpucpu)
11. [Correspondance avec PyTorch](#correspondance-avec-pytorch)
12. [Limites honnêtes, en détail](#limites-honnêtes-en-détail)
13. [Comment ça a été vérifié](#comment-ça-a-été-vérifié)

---

### Ce que c'est, et ce que ce n'est pas

`atome` est un **module TON618 ordinaire** — un seul fichier `.ton`, sans aucune modification de l'interpréteur. Il donne :
- des tenseurs 1D/2D avec les opérations de base (élément par élément, matmul, réductions, activations),
- un vrai moteur d'autodiff par rétropropagation, construit de zéro (même principe que [micrograd](https://github.com/karpathy/micrograd)),
- des couches Dense (fully-connected), Conv1D (un canal), Dropout, et un conteneur Sequential,
- deux optimiseurs (SGD, Adam),
- une perte de régression (MSE) et une perte de classification (softmax + cross-entropy fusionnées),
- un tokenizer mot-par-mot avec vocabulaire par fréquence,
- la sauvegarde/le chargement de modèles au format `.atm` (du JSON).

Ce n'est **pas** un PyTorch réécrit. Pas de GPU, pas de vraie diffusion (broadcasting) façon NumPy, pas de convolution/RNN/attention, pas de compilation ni de vectorisation — juste des boucles TON618 interprétées. Voir [Limites honnêtes](#limites-honnêtes-en-détail) pour le détail complet de pourquoi, pas juste la liste.

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
| `atome_relu_layer()` / `atome_sigmoid_layer()` / `atome_tanh_layer()` | "Couches" d'activation, à insérer dans un Sequential |
| `atome_sequential(layers)` | Enchaîne une liste de couches en un modèle |
| `atome_forward(model, xNode)` | Passe avant complète, construit le graphe d'autodiff au fur et à mesure |
| `atome_params(model)` | Tous les Nodes entraînables (poids + biais) du modèle, pour l'optimiseur |
| `atome_predict_class(logitsTensor)` | Raccourci pour `tensor_argmax` — l'index de la classe prédite (ou un tableau d'indices pour un batch) |

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

| Fonction | Description |
|---|---|
| `atome_save(model, chemin)` | Écrit les poids/biais des couches Dense du modèle dans un fichier `.atm` (JSON) |
| `atome_load(chemin)` | Reconstruit un modèle depuis un fichier `.atm` |

Un fichier `.atm` ressemble à ça (format `atome-v1`) :
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
| `torch.save(model, path)` | `atome_save(model, path)` | Format JSON simple, pas le format pickle de PyTorch |
| `model.to("cuda")` | `atome_set_device("gpu")` | **Ne fait rien de réel** — affiche un avertissement et reste sur CPU, voir ci-dessous |
| `torch.cuda.is_available()` | *(rien)* | Pas de GPU accessible depuis TON618 — voir Limites |
| `nn.Conv2d`, `nn.LSTM`, `nn.MultiheadAttention`, `nn.BatchNorm*` | *(rien)* | Pas implémenté — voir Limites |
| Broadcasting NumPy complet | *(rien, sauf `atome_tensor_add_bias`)* | Pas implémenté — voir Limites |
| Exécution GPU (CUDA/cuDNN) | *(rien)* | Impossible en TON618 pur — voir Limites |

---

### Limites honnêtes, en détail

**Pas de GPU, et ça ne peut pas être ajouté par un module.** Exécuter du calcul sur GPU demande un vrai pilote/binding (CUDA, OpenCL, Vulkan compute...) compilé dans l'interpréteur lui-même — un fichier `.ton` ne peut pas appeler de code natif en dehors de ce que l'interpréteur expose déjà. `atome_set_device("gpu")` existe pour que l'API ressemble à celle de PyTorch, mais il affiche un avertissement et continue sur CPU — jamais de faux-semblant silencieux.

**1D et 2D seulement.** Pas de tenseurs 3D+ — donc pas de batchs d'images (qui seraient 4D : batch × canaux × hauteur × largeur), pas de tenseurs façon "têtes d'attention". Pour ton propre usage, tu peux toujours aplatir/reshaper en 2D toi-même si ton cas s'y prête.

**Diffusion (broadcasting) minimale.** `tenseur + nombre` marche. `tenseur + tenseur-de-forme-différente` ne marche pas, sauf le cas explicite `atome_tensor_add_bias` (biais 1D ajouté à chaque ligne d'une matrice 2D) — parce que c'est exactement ce dont une couche Dense a besoin, et rien de plus général n'a été construit.

**Plus rapide depuis beta-1.0.6, mais toujours loin d'une vraie bibliothèque numérique.** Depuis `ton618` beta-1.0.6, l'interpréteur fournit un module natif `ton.tensor` — les opérations lourdes (élément par élément, matmul, activations) tournent en vraies boucles C++ plutôt qu'en boucles TON618 interprétées. `atome` s'appuie dessus en interne (même API publique, zéro changement de code pour toi). Gain mesuré sur les exemples fournis : l'entraînement XOR est passé de ~6,3s à ~1,1s, et l'exemple de classification de ~6,8s à ~0,4s — 5 à 16x plus rapide. Ça reste un tableau `Value` typé dynamiquement parcouru en C++, pas un tableau contigu vectorisé façon BLAS — donc toujours très loin des performances d'une vraie bibliothèque numérique, mais nettement plus utilisable qu'avant pour de l'expérimentation.

**Le tokenizer est mot-par-mot, pas BPE.** Un vrai entraîneur BPE (byte-pair encoding) construit son vocabulaire en fusionnant itérativement les paires de sous-mots les plus fréquentes sur tout un corpus — un algorithme sensiblement plus complexe que compter des mots. Ce qui est ici fonctionne et est vérifié, mais ce n'est pas ce que fait un vrai tokenizer de LLM moderne.

**`.atm` perd un peu de précision.** Le format est du JSON simple (lisible, débogable) plutôt qu'un format binaire — mais `json_stringify`/`json_pretty` de TON618 n'écrivent qu'environ 6 chiffres significatifs par nombre. Sauvegarder puis recharger un modèle introduit donc un tout petit bruit numérique sur les poids. Négligeable sur un petit modèle jouet ; à savoir si tu sauvegardes/recharges en boucle.

**Conv1D existe mais reste limitée : un seul canal d'entrée et de sortie.** Une vraie couche de convolution empile plusieurs canaux (`nn.Conv1d(inChannels, outChannels, K)`) ; ici, `atome_conv1d` ne gère qu'un canal en entrée et un en sortie. Toujours réel et vérifié (test de gradient numérique + entraînement qui retrouve exactement le noyau `[-1, 1]`), mais pour un vrai réseau à plusieurs canaux il faudrait composer plusieurs couches `atome_conv1d` toi-même (une par paire de canaux) et sommer les sorties. Pas de Conv2D, pas de RNN, pas d'attention, pas de BatchNorm — faisables sur ce squelette, mais pas ajoutés ici pour rester dans un périmètre réellement testé plutôt que d'empiler des fonctionnalités non vérifiées.

**Note sur "vraiment comme PyTorch"** : `ton.tensor` (beta-1.0.6+) comble une partie réelle de l'écart — les boucles chaudes sont maintenant natives, avec un gain mesuré de 5 à 16x. Ce que ça ne change toujours pas : pas de GPU (aucun binding CUDA/OpenCL, impossible à ajouter depuis un module `.ton`), pas de tableau contigu vectorisé (`Value` reste typé dynamiquement même en C++), pas de compilation JIT du graphe de calcul. Combler complètement l'écart demanderait encore plus de travail interpréteur — des bindings BLAS/GPU réels — mais ce n'est déjà plus "juste des boucles TON618 interprétées" comme dans la première version de ce module.

---

### Comment ça a été vérifié

Rien ici n'a été livré "ça a l'air correct" sans test réel contre l'interpréteur `ton618` :

- **Tenseurs** : chaque opération (matmul, transpose, add/sub/mul avec tenseur-tenseur et tenseur-scalaire, réductions, relu, softmax qui somme à 1, add_bias) vérifiée contre des résultats calculés à la main.
- **Autodiff** : validé en entraînant réellement le problème XOR classique (2 entrées → 8 neurones cachés (tanh) → 1 sortie (sigmoid), perte MSE, SGD simple) : la loss descend de 0.249 à 0.0008 sur 2000 epochs, et les 4 prédictions finales sont toutes correctes. C'est le test standard pour vérifier qu'une rétropropagation est *réellement* juste, pas juste qu'elle s'exécute sans planter.
- **Classification multi-classes + Adam** : validé en entraînant un modèle à séparer 3 clusters 2D avec softmax + cross-entropy et l'optimiseur Adam : la loss descend de 1.46 à 0.00008 sur 300 epochs, avec 18/18 de précision finale.
- **Tokenizer** : ordre du vocabulaire et aller-retour tokenize/detokenize vérifiés contre des fréquences de mots calculées à la main.
- **Sauvegarde/chargement** : vérifié que le modèle rechargé produit une sortie quasi identique (à la précision JSON près, documentée ci-dessus) à celle du modèle original.
- **Migration vers `ton.tensor` (beta-1.0.6)** : les quatre tests ci-dessus ont été rejoués intégralement après le passage au backend natif — mêmes résultats numériques exacts (mêmes prédictions XOR, même trajectoire de loss, même précision de classification), seule la vitesse a changé (mesurée : 5 à 16x plus rapide).
- **Conv1D** : vérifiée de deux façons indépendantes — un **test de gradient numérique** (`examples/gradient_check.ton`, différences finies comparées au gradient analytique, écart maximal affiché à 0.000000) qui confirme que la passe arrière est mathématiquement correcte, puis un **entraînement réel** (`examples/train_conv1d.ton`) où la couche apprend, à partir de simples exemples, le noyau exact `[-1, 1]` de l'opérateur "différence discrète" — la preuve qu'au-delà d'être juste, ça apprend vraiment quelque chose.
- **Dropout** : vérifié qu'environ `p` fraction des éléments sont mis à zéro pendant l'entraînement, que les survivants sont mis à l'échelle par `1/(1-p)`, et que le mode évaluation redonne exactement l'entrée d'origine.
