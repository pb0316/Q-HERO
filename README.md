# QACGen: Quaternary Ammonium Compound Generator

A deep learning framework for generating novel Quaternary Ammonium Compounds (QACs) using a hierarchical Variational Autoencoder (VAE) approach. This project decomposes QACs into core and tail components, learns their representations separately, and reassembles them to generate new molecules.

## Project Structure

```
QACGen/
├── 0_split_core_tail.ipynb      # Data preprocessing and core/tail splitting
├── 1_latent_vae_gru.ipynb       # GRU-VAE training for sequence modeling
├── 2_generation.ipynb           # Molecule generation pipeline
├── 3_eval.ipynb                 # Evaluation metrics (uniqueness, novelty, QAC ratio)
├── model_qacvae.py              # GRU-VAE model implementation
├── util_decoder_full.py         # Complete generation pipeline
├── util_get_seq_emb_data.py     # Data preparation utilities
├── util_reassemble_core_tail.py # Core-tail reassembly logic
├── util_split_core_tail.py      # Core-tail splitting algorithms
├── hgraph2graph/                # Hierarchical graph neural network models
├── qac-*.txt                    # Dataset files
└── 0315-*.txt                   # Processed core/tail fragments
```

## Key Components

### 1. Core-Tail Decomposition (`util_split_core_tail.py`)

- **`iterative_cut()`**: Recursively splits QACs at quaternary nitrogen sites
- **`cut_once()`**: Performs single cuts while preserving minimum carbon chain lengths
- **`try_n_plus_or_N_plus()`**: Handles aromatic vs aliphatic nitrogen notation

### 2. GRU-VAE Model (`model_qacvae.py`)

- **Encoder**: GRU-based encoder that maps sequences to latent space
- **Decoder**: GRU-based decoder that generates sequences from latent vectors
- **Reparameterization**: Standard VAE reparameterization trick
- **Architecture**: 8-dimensional input, 100 hidden units, 20 latent dimensions

### 3. Hierarchical Graph Models (`hgraph2graph/`)

- **Core VAE**: Trained on core fragments using hierarchical graph neural networks
- **Tail VAE**: Trained on tail fragments with separate vocabulary
- **Graph-to-Graph**: Converts between molecular graphs and SMILES representations

### 4. Generation Pipeline (`util_decoder_full.py`)

- **`QACGenerator`**: Main generation class that orchestrates the entire process
- **Core Generation**: Samples cores from the core VAE
- **Tail Generation**: Samples tails from the tail VAE
- **Reassembly**: Combines cores and tails using chemical rules

## Usage

### Prerequisites

```bash
pip install torch rdkit-pypi numpy scikit-learn joblib tqdm
```

### Training Pipeline

1. **Data Preprocessing** (`0_split_core_tail.ipynb`):
   - Load QAC dataset
   - Split into cores and tails
   - Generate training sequences

2. **Core / Tail Model Training** (`1_latent_vae_gru.ipynb`):
   - Run the training code in https://github.com/wengong-jin/hgraph2graph.

3. **GLobal VAE Training** (`1_latent_vae_gru.ipynb`):
   - Train GRU-VAE on core-tail sequences
   - Save trained model weights

3. **Generation** (`2_generation.ipynb`):
   - Load trained models
   - Generate new molecules
   - Save results

4. **Evaluation** (`3_eval.ipynb`):
   - Calculate uniqueness, novelty, and QAC ratio
   - Assess generation quality

## Model Architecture

### GRU-VAE
- **Input Size**: 8 (embedding dimension)
- **Hidden Size**: 100
- **Latent Size**: 20
- **Layers**: 1
- **Loss**: MSE reconstruction + KL divergence (β=0.0001)

### Hierarchical Graph VAEs
- **Architecture**: LSTM-based with graph neural networks
- **Hidden Size**: 100
- **Embed Size**: 100
- **Latent Size**: 8
- **Depth**: 15 (both tree and graph)

## Evaluation Metrics

- **Uniqueness**: Fraction of unique molecules in generated set
- **Novelty**: Fraction of generated molecules not in training set
- **QAC Ratio**: Fraction of generated molecules that are valid QACs

## Dependencies

- PyTorch
- RDKit
- NumPy
- Scikit-learn
- Joblib
- tqdm

