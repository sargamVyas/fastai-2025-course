# fast.ai Course Portfolio: Practical Deep Learning for Coders

This repository serves as a structured portfolio for my work and personal implementations developed while following the **fast.ai course: *Practical Deep Learning for Coders***.

It is organized to track progress through all lessons, practice exercises, and standalone projects.

---

## Project Goal and Overview

The primary goal is to master practical deep learning techniques across various domains (vision, NLP, tabular data) using the fast.ai library and PyTorch framework.

## Repository Structure

This structure is designed to keep course-specific work separate from independent projects.

| Directory | Content | Description |
| :--- | :--- | :--- |
| `part1/lesson4/` | Lesson Implementations | Contains code scripts and notebooks directly related to the core course content (e.g., your initial NLP script). |
| `practice/` | Practice Exercises | Dedicated directory for personal challenges, modified examples, and deep-dive practice labs. |
| `projects/` | Standalone Projects | Larger, self-contained applications or competition submissions (e.g., final model submission code). |
| `data/` | Raw Data | Local storage for raw datasets (downloaded from Kaggle, fast.ai, etc.). *Ignored by Git.* |

---

## 📝 Lesson-by-Lesson Progress

### 🌟 Part 1: Foundations of Deep Learning

| Lesson | Topic | Code Status | Notes / Key Implementation |
| :--- | :--- | :--- | :--- |
| **L1** | Image Classification | ✅ Done | Setting up the DataBlock API and Transfer Learning. |
| **L2** | Production | ✅ Done | Exporting models and introduction to Gradio/deployment. |
| **L3** | Data Ethics & Image Segmentation | 🚧 In Progress | U-Nets for pixel-level prediction. |
| **L4** | NLP & Tabular Data** | ✅ Done | Hugging Face Tokenization and Data Prep (See details below).|
| **L5** | **Linear Model and Neural Net from Scratch** | ✅ Done | Linear model, Neural Net, Deep learning model from scratch(See details below). |
| **L6** | Random Forest | ⬜ Pending | |
| **L7** | Collaborative Filtering | ⬜ Pending |  |
| **L8** | Collaborative Filtering | ⬜ Pending |  |
---

## Feature Focus: Lesson 4 - NLP Data Preparation

This section details the initial implementation from Lesson 4, which focuses on preparing text data for a transformer model.

### US Patent Phrase Matching - Baseline Tokenization

* **Code Location:** `part1/lesson4/us_patent_phrase_matching.py`
* **Model Used:** `microsoft/deberta-v3-small` (via Hugging Face)
* **Goal:** Create a tokenized dataset ready for model training, based on the Kaggle competition.

### Key Technical Steps:

1.  **Input Construction:** The required competition inputs (`context`, `target`, and `anchor`) are combined into a single, structured string format to maximize contextual information for the transformer model.
    > Example: `Text1: [Context]; Text2: [Target]; Anc1: [Anchor]`
2.  **Efficient Processing:** The `pandas.DataFrame` is converted to a Hugging Face `Dataset` object, allowing the `.map(..., batched=True)` method to apply the tokenizer across large chunks of data efficiently.
3.  **Tokenization & Numericalization:** The script utilizes the pre-trained `AutoTokenizer` to perform the critical steps of converting raw text into numerical token IDs. This output is the final input format required by models like DeBERTa. 

---

## ⚙️ Setup and Dependencies

### 1. Prerequisites

You need a Python 3.8+ environment.

### 2. Installation

Install the necessary libraries:

```bash
pip install fastai pandas datasets transformers# fastai-2025-course
```

##  Acknowledgements

This portfolio is developed as an implementation of the techniques and principles taught in the **fast.ai course: *Practical Deep Learning for Coders***.

Specifically, the code for the Lesson 4 NLP baseline was adapted from the materials developed by Jeremy Howard and the fast.ai team for the [Kaggle US Patent Phrase Matching Competition](https://www.kaggle.com/competitions/us-patent-phrase-matching).

* **Course Website:** [https://course.fast.ai/](https://course.fast.ai/)
* **Original Notebook Reference (Lesson 4):** [https://www.kaggle.com/code/jhoward/getting-started-with-nlp-for-absolute-beginners/notebook]


##  Feature Focus: Lesson 5 - Linear Model & Neural Net from Scratch

This section details the implementation from Lesson 5, focusing on building the "engine" of deep learning using low-level PyTorch tensors to understand the underlying calculus and algebra.

### 🧩 Titanic Survival - Foundations from Scratch

* **Code Location:** `part1/lesson5/Titanic_classification_from_scratch.py`
* **Data Used:** Titanic Passenger Manifest (Kaggle)
* **Goal:** Understand the "black box" of AI by manually implementing the math behind gradient descent and multi-layer neural networks.

### Key Technical Steps:

1.  **Manual Feature Engineering:** Instead of using high-level loaders, categorical variables were manually converted into dummy variables (one-hot encoding) and skewed numeric columns were normalized to improve model convergence.
2.  **Broadcasting & Matrix Multiplication:** Utilized PyTorch's broadcasting rules to perform efficient operations on entire data batches simultaneously, eliminating the need for manual loops.
3.  **The Optimization Loop (SGD):** Implemented the Stochastic Gradient Descent cycle from scratch:
    * **Forward Pass:** Calculating predictions using $y = Xw + b$.
    * **Loss Calculation:** Measuring error using Mean Absolute Error (MAE).
    * **Gradient Descent:** Updating weights manually using `.backward()` and `with torch.no_grad():`.
4.  **Neural Network Architecture:** Transitioned from a simple linear model to a 2-layer neural network by introducing a hidden layer and a **ReLU (Rectified Linear Unit)** activation function to capture non-linear relationships.

---

## ⚙️ Setup and Dependencies

### 1. Prerequisites
You need a Python 3.8+ environment.

### 2. Installation
Install the necessary libraries (Added `matplotlib` for visualizing training loss):

```bash
pip install fastai pandas datasets transformers matplotlib
```

##  Acknowledgements

This portfolio is developed as an implementation of the techniques and principles taught in the **fast.ai course: *Practical Deep Learning for Coders***.

Specifically, the code for the Lesson 4 NLP baseline was adapted from the materials developed by Jeremy Howard and the fast.ai team for the [Titanic - Machine learning from Distater](https://www.kaggle.com/competitions/titanic/).

* **Course Website:** [https://course.fast.ai/](https://course.fast.ai/)
* **Original Notebook Reference (Lesson 5):** [https://www.kaggle.com/code/jhoward/linear-model-and-neural-net-from-scratch/notebook]