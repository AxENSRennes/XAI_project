# Counterfactual Audio Reproduction Kit

Reproduction pragmatique du papier `Learning Audio Concepts from Counterfactual Natural Language` à partir des artefacts publics disponibles et d'un pipeline CLAP-like configurable.

Ce dépôt ne prétend pas refaire exactement l'implémentation des auteurs. Le code original n'est pas public, et plusieurs hyperparamètres critiques ne sont pas publiés dans le papier. Le but ici est de fournir une base exécutable pour:

- télécharger les annotations contre-factuelles publiques,
- fabriquer des manifests audio/texte exploitables,
- générer ses propres contre-factuels quand nécessaire,
- entraîner un modèle audio-texte avec les losses du papier,
- évaluer le retrieval texte-vers-audio et le zero-shot.

## Environnement

Utiliser explicitement le venv demandé:

```bash
/home/axel/wsl_venv/bin/python --version
/home/axel/wsl_venv/bin/pip install -r requirements.txt
```

## Structure

```text
.
|-- AGENTS.md
|-- README.md
|-- configs/
|-- examples/
|-- requirements.txt
|-- scripts/
`-- src/
```

## Pipeline

1. Télécharger les JSON publics de `counterfactual-audio`.
2. Construire des manifests JSONL aplatis avec chemins audio.
3. Optionnellement générer ses propres captions contre-factuels.
4. Entraîner un backbone CLAP Hugging Face avec:
   - loss contrastive CLAP,
   - `L_angle`,
   - `L_factual_consistency`.
5. Évaluer:
   - retrieval Clotho style,
   - zero-shot classification ESC-50 / US8K style.

## Commandes

Téléchargement des artefacts publics:

```bash
/home/axel/wsl_venv/bin/python scripts/download_release_assets.py --output-dir data/releases
```

Construction d'un manifest depuis un JSON publié:

```bash
/home/axel/wsl_venv/bin/python scripts/build_manifest.py \
  --release-json data/releases/clotho-development-counterfactual.json \
  --dataset-root /path/to/clotho \
  --output data/manifests/clotho_dev.jsonl \
  --dataset-name clotho
```

Génération heuristique de contre-factuels:

```bash
/home/axel/wsl_venv/bin/python scripts/generate_counterfactuals.py \
  --input examples/factual_captions.txt \
  --output examples/generated_counterfactuals.jsonl
```

Entraînement:

```bash
HF_HOME=/home/axel/XAI/.hf-cache \
HF_HUB_CACHE=/home/axel/XAI/.hf-cache/hub \
TRANSFORMERS_CACHE=/home/axel/XAI/.hf-cache/transformers \
/home/axel/wsl_venv/bin/python scripts/train_counterfactual_clap.py \
  --config configs/paper_reproduction.yaml \
  --train-manifest data/manifests/train.jsonl \
  --eval-manifest data/manifests/clotho_val.jsonl \
  --output-dir runs/paper_reproduction
```

Retrieval:

```bash
HF_HOME=/home/axel/XAI/.hf-cache \
HF_HUB_CACHE=/home/axel/XAI/.hf-cache/hub \
TRANSFORMERS_CACHE=/home/axel/XAI/.hf-cache/transformers \
/home/axel/wsl_venv/bin/python scripts/evaluate_retrieval.py \
  --checkpoint runs/paper_reproduction/best.pt \
  --manifest data/manifests/clotho_val.jsonl
```

Zero-shot:

```bash
HF_HOME=/home/axel/XAI/.hf-cache \
HF_HUB_CACHE=/home/axel/XAI/.hf-cache/hub \
TRANSFORMERS_CACHE=/home/axel/XAI/.hf-cache/transformers \
/home/axel/wsl_venv/bin/python scripts/evaluate_zero_shot.py \
  --checkpoint runs/paper_reproduction/best.pt \
  --manifest data/manifests/esc50_eval.jsonl \
  --label-template "this is a sound of {}"
```

## Notes d'execution

- Dans cet environnement, le cache Hugging Face global est en lecture seule. Les commandes ci-dessus forcent donc un cache local dans `.hf-cache/`.
- Certains manifests publiés peuvent référencer des fichiers absents après extraction, ou des noms de fichiers non alignés entre release texte et audio. Utiliser `scripts/filter_manifest.py` pour nettoyer les manifests avant entraînement ou évaluation.

## Approximations connues

- Le papier annonce un encodeur audio PANNs `ResNet-38` avec adapters. Ce kit utilise par défaut un backbone CLAP Hugging Face pour disposer d'un pipeline réutilisable sans code source privé.
- Le texte est gelé par défaut, comme dans le papier.
- La génération contre-factuelle par défaut est heuristique et déterministe. Elle peut être remplacée par une génération LLM externe si besoin.
- Le JSON de validation Clotho publié par les auteurs du dépôt public ne contient pas de vrais contre-factuels, seulement une copie des captions factuels.

## Fichiers d'exemple

- [examples/factual_captions.txt](/home/axel/XAI/examples/factual_captions.txt)
- [examples/generated_counterfactuals.jsonl](/home/axel/XAI/examples/generated_counterfactuals.jsonl)
