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
| **L4** | **NLP & Tabular Data** | ✅ Done | **Hugging Face Tokenization and Data Prep (See details below).** |
| **L5** | Collaborative Filtering | ⬜ Pending | Matrix Factorization and creating embedding spaces. |
| **L6** | Practical Deployment | ⬜ Pending | Advanced topics in deployment, ethics, and bias. |

---

## 🧑‍💻 Feature Focus: Lesson 4 - NLP Data Preparation

This section details the initial implementation from Lesson 4, which focuses on preparing text data for a transformer model.

### 🧩 US Patent Phrase Matching - Baseline Tokenization

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


##  Acknowledgements

This portfolio is developed as an implementation of the techniques and principles taught in the **fast.ai course: *Practical Deep Learning for Coders***.

Specifically, the code for the Lesson 4 NLP baseline was adapted from the materials developed by Jeremy Howard and the fast.ai team for the [Kaggle US Patent Phrase Matching Competition](https://www.kaggle.com/competitions/us-patent-phrase-matching).

* **Course Website:** [https://course.fast.ai/](https://course.fast.ai/)
* **Original Notebook Reference (Lesson 4):** [https://www.kaggle.com/code/jhoward/getting-started-with-nlp-for-absolute-beginners/notebook]