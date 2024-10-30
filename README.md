# SemHash: Fast Text Deduplication using Semantic Hashing

SemHash is a technique to efficiently deduplicate datasets based on semantic similarity.

## Table of Contents

## Table of Contents
- [Quickstart](#quickstart)
- [Main Features](#main-features)
- [What is SemHash?](#what-is-semhash)
- [Usage](#usage)
- [Related Work](#related-work)
- [Citing](#citing)

## Quickstart


Install the package with:
```bash
pip install semhash
```

Deduplicate a single dataset with the following code:

```python
from model2vec import StaticModel
from semhash import SemHash
from datasets import load_dataset

# Load an embedding model
model = StaticModel.from_pretrained("minishlab/M2V_base_output")

# Initialize a SemHash with the model
semhash = SemHash(model=model)

# Load some texts to deduplicate
texts = load_dataset("ag_news", split="train")["text"]

# Deduplicate the texts
deduplicated_indices, duplicate_mapping = semhash.deduplicate(texts)
```


Or, deduplicate across two datasets (for example a train and test set) with the following code:

```python
from model2vec import StaticModel
from semhash import SemHash
from datasets import load_dataset

# Load an embedding model
model = StaticModel.from_pretrained("minishlab/M2V_base_output")

# Initialize a SemHash with the model
semhash = SemHash(model=model)

# Load two datasets
texts1 = load_dataset("ag_news", split="train")["text"]
texts2 = load_dataset("ag_news", split="test")["text"]

# Deduplicate the texts
deduplicated_indices, duplicate_mapping = semhash.deduplicate(texts1=texts1, texts2=texts2)
```

SemHash supports two types of deduplication:
- Exact: uses [reach]() for efficient exact nearest neighbors computation. This is recommended for smaller datasets.
- Aproximiate: uses [faiss]() for aproximate nearest neighbors computations. This is recommended for larger datasets.


#TODO make a plot of n_embeddings vs time for exact search
#TODO make a plot of n_embeddings vs time for aproximate search
# Add recall somehow
