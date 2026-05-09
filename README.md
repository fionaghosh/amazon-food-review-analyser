# 🛒 Amazon Food Review Analyser

A production-grade NLP application that classifies Amazon food reviews as positive or negative using a fine-tuned DistilBERT model, and surfaces product category insights using BERTopic topic modelling.

🔗 **Live Demo:** https://amazon-food-review-analyser-5drfwwvmcfitxfzeyf4mml.streamlit.app/

---

## 📌 Project Overview

This project builds an end-to-end NLP pipeline on the Amazon Fine Food Reviews dataset (568,000+ reviews), combining transformer-based sentiment classification with unsupervised topic modelling to answer the question: *which product categories drive the most negative reviews?*

---

## Model & Results

| Metric | Score |
|---|---|
| Overall Accuracy | 94.98% |
| Negative F1 | 0.84 |
| Positive F1 | 0.97 |
| Macro F1 | 0.91 |

- **Model:** DistilBERT fine-tuned for binary sentiment classification
- **Training data:** 40,000 sampled reviews (80/20 train/val split)
- **Class imbalance handling:** Weighted cross-entropy loss (84% positive, 16% negative)
- **Hosted on HuggingFace:** [fionaghosh/amazon-food-review-sentiment](https://huggingface.co/fionaghosh/amazon-food-review-sentiment)

---

## 🔍 Topic Modelling Insights

BERTopic was used to discover 10 product categories from 50,000 reviews unsupervised. Key findings:

| Category | Negative Review % |
|---|---|
| Product/Shipping (Amazon) | 30.8% |
| Candy & Gifts | 20.7% |
| Chocolate | 18.9% |
| Coffee | 18.1% |
| Tea | 12.4% |

**Key insight:** Shipping and fulfilment complaints account for the highest negativity rate — nearly double the average — suggesting the product itself is often not the issue.

---

## 🛠️ Tech Stack

- **Model:** HuggingFace Transformers (DistilBERT)
- **Topic Modelling:** BERTopic + UMAP + HDBSCAN
- **Embeddings:** Sentence Transformers (all-MiniLM-L6-v2)
- **Training:** PyTorch on Kaggle GPU (T4 x2)
- **App:** Streamlit
- **Model hosting:** HuggingFace Hub
- **Dataset:** Amazon Fine Food Reviews (Stanford SNAP)

---

## 🚀 Run Locally

```bash
git clone https://github.com/fionaghosh/amazon-food-review-analyser
cd amazon-food-review-analyser
pip install -r requirements.txt
streamlit run app.py
```

The model downloads automatically from HuggingFace on first run.

---

## 📁 Repository Structure

```bash
├── app.py                  # Streamlit application
├── requirements.txt        # Dependencies
├── topic_sentiment.csv     # BERTopic category sentiment data
└── sentiment_model/        # Tokenizer config files (model weights on HuggingFace)
```

---

*Built by Fiona Ghosh 
