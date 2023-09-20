# Visual Transformers: Cats vs. Dogs Classification

![Output Gif](images/output.gif)  <!-- Add the path if the GIF is located in another directory -->

A machine learning project utilizing Visual Transformers (ViTs) to classify images from the Cats vs. Dogs dataset.


## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contribute](#contribute)
- [Acknowledgements](#acknowledgements)

## Introduction

The Cats vs. Dogs dataset is a standard computer vision dataset that contains images of cats and dogs. In this project, instead of using conventional CNNs, we utilize Visual Transformers (ViTs), a relatively new paradigm in computer vision.

## Installation

**Requirements**: Python 3.8+ 

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/cats-vs-dogs-vit.git
   cd cats-vs-dogs-vit
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To train the model:

```bash
python train.py
```

To evaluate the model:

```bash
python evaluate.py
```

## Results

We achieved an accuracy of XX% on the test set using ViTs. Check out the [Jupyter notebook](./results-analysis.ipynb) for a deep dive into the analysis.

| Model           | Accuracy (%) |
|-----------------|--------------|
| Visual Transformer | XX         |

## Contribute

Contributions are always welcome! See [CONTRIBUTING.md](./CONTRIBUTING.md) for ways to get started.

## Acknowledgements

- Thanks to the creators of the [Cats vs. Dogs dataset](https://www.kaggle.com/c/dogs-vs-cats).
- Inspired by the [MAE paper](https://arxiv.org/abs/2111.06377).

---

Ⓒ 2023 Mathew Salvaris. All Rights Reserved.
```