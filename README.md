# Sarcasm Detection in News Headlines

A Natural Language Processing project for **Text Mining and NLP (TMNLP) 2023–2024** at University.

## Objective

The goal of this project is to build a machine learning pipeline capable of classifying news headlines as **sarcastic** or **non-sarcastic**.
We explore a wide range of models, from traditional machine learning with Bag-of-Words to state-of-the-art transformer-based models like BERT.

---

## Dataset

We use the **Sarcasm Headlines Dataset v2**, available on Kaggle:

* Source: [Kaggle Dataset](https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection)
* Format: JSON
* Total records: 28,617
* Sarcastic: 13,633
* Non-sarcastic: 14,984
* Fields:

  * `is_sarcastic`: binary label (1 = sarcastic, 0 = not)
  * `headline`: news headline
  * `article_link`: original article URL

---

## Methodology

### Preprocessing

1. **Remove duplicates & unwanted columns**
2. **Expand contractions** (e.g. "won't" → "will not")
3. **Punctuation removal**
4. **Tokenization** (via `nltk`)
5. **Stop word removal** (`nltk`)
6. **Lemmatization** (context-aware)
7. **Store cleaned headlines** in both string and list format

### Feature Extraction

| Method                 | Description                                        |
| ---------------------- | -------------------------------------------------- |
| **Bag of Words (BoW)** | Simple word count vectorization                    |
| **TF-IDF**             | Weighted term importance across documents          |
| **Word2Vec**           | Pretrained embeddings to capture context           |
| **BERT Embeddings**    | Contextual embeddings from pre-trained BERT models |

Each method was paired with a **Logistic Regression** classifier for consistency in comparison.

---

## Models Tested

| Model Type              | Description                                                                     |
| ----------------------- | ------------------------------------------------------------------------------- |
| **Logistic Regression** | Paired with BoW, TF-IDF, Word2Vec, BERT embeddings                              |
| **BERT Fine-tuned**     | Full transformer model using HuggingFace's `AutoModelForSequenceClassification` |

---

## Final Model (Hugging Face)

For the final model, we fine-tuned a BERT transformer using Hugging Face:

* Hugging Face Model:
  🔗 [PietroSaveri/Sarcastic\_01](https://huggingface.co/PietroSaveri/Sarcastic_01)

* Hugging Face Library: `transformers`

### Hugging Face Pipeline Example:

```python
from transformers import pipeline

classifier = pipeline("text-classification", model="PietroSaveri/Sarcastic_01")
classifier("Wow, I just love getting stuck in traffic!")
# Output: [{'label': 'SARCASTIC', 'score': 0.95}]
```

---

## Installation & Usage

Run directly in **Google Colab** (preferred environment – dependencies handled in code).

### ▶Run Project

Start from the notebook:

```
main_sarcasm_detection.ipynb
```

You can also load the BERT embeddings directly for training/testing:

* `bert_train_embeddings.npy`
* `bert_test_embeddings.npy`

---

## Results

| Model                | Accuracy    |
| -------------------- | ----------- |
| BoW + LR             | \~78%       |
| TF-IDF + LR          | \~80%       |
| Word2Vec + LR        | \~81%       |
| BERT Embeddings + LR | \~85%       |
| BERT Fine-tuned      | **\~90%** ✅ |

---

## Authors

<div align="center"> <table> <tr> <td align="center"> <a href="https://github.com/PietroSaveri"> <img src="https://github.com/PietroSaveri.png" width="100px;" alt="Pietro Saveri"/><br /> <sub><b>Pietro Saveri</b></sub> </a> </td> <td align="center"></sub> </a> </td> <td align="center"> <a href="https://github.com/M4tteoo"> <img src="https://github.com/M4tteoo.png" width="100px;" alt="Matteo Salami"/><br /> <sub><b>Matteo Salami</b></sub> </a> </td> </tr> </table> </div>

