# WebScience-Project-Group3
# Deceptive by Design: Clickbait Detection in NYT Headlines

This repository contains code and analysis for the project **"Deceptive by Design"**, which investigates the presence, evolution, and patterns of clickbait headlines in *The New York Times* from 2015 to 2025 using transformer-based classification models.

Our objective is to:
- Apply pretrained/fine-tuned models to detect clickbait in NYT online news articles at scale
- Analyze its prevalence across sections and time
- Understand linguistic and stylistic framing patterns

## Files Overview

### `webscience.ipynb` – Full Pipeline + Composite Analysis
This is the most comprehensive notebook, combining:
- Data loading, preprocessing, and merging of NYT articles (2015–2025)
- Clickbait prediction using `xlm-roberta-base-clickbait`
- Textual feature extraction, including:
  - Cosine similarity between headlines and lead paragraphs
  - Hyperbole detection using the NRC lexicon
  - Subjectivity scores using TextBlob
- Composite clickbait scoring using normalized weighted features
- Visualizations, including:
  - Section-wise clickbait proportions
  - Time series of clickbait usage
  - Distribution of composite scores
  - Cosine similarity vs. model scores
- Linguistic framing (e.g., listicles, curiosity gap, second-person, etc.)
- Event-driven correlation (e.g., COVID-19, ChatGPT launch)

**Key outcome**: Demonstrates both model-based and interpretable heuristic-based clickbait classification and compares their effectiveness.

---

### `untitled1.ipynb` – Model Inference on Headlines vs. Combined Text
This notebook performs two main prediction pipelines:
1. Headline-only clickbait classification  
2. Headline + Lead Paragraph (context-enhanced) classification

For each NYT article:
- Text is formatted with semantic prompting (e.g., `"Headline: ..."`)
- Batched predictions are made using HuggingFace Transformers
- Scores and labels are stored year-wise for traceable labeling

**Outputs**: Two labeled datasets (`streamie_headline_test.csv` and `streamie_lead_test.csv`) that are reused across the project for downstream analysis.

---

### `web_inference.ipynb` – Temporal and Sectional Analysis
This notebook loads the labeled datasets and performs:
- Monthly clickbait proportion visualizations
- Impact of global events on clickbait trends (annotated timelines)
- Clickbait-type distribution (listicle, hyperbole, curiosity, etc.)
- Keyword-based event matching (e.g., Ukraine invasion, AI hype)
- Statistical testing, including:
  - Chi-square for section correlation
  - Paired t-tests for score differences
- Clustering of “Other” clickbait types using both TF-IDF and SentenceBERT

**Focus**: Statistical insight, event correlation, and framing trend identification.

---

## Input Dataset
We use a filtered version of the [NYT Articles 2000–2025 dataset](https://www.kaggle.com/datasets/aryansingh0909/nyt-articles-21m-2000-present/data), restricted to years 2015–2025 and sampled uniformly across time for balance.

## Research Questions
- RQ1: Can we detect and quantify clickbait in NYT articles?
- RQ2: Which topical sections show higher clickbait prevalence?
- RQ3: How has clickbait evolved over the last decade?

## Model
- `Stremie/xlm-roberta-base-clickbait` (HuggingFace)
  - Pretrained on the Webis Clickbait corpus
  - Applied zero-shot on NYT headlines and article leads

---

## Sample Visualizations
- Clickbait percentage over time (pre/post ChatGPT)
- Section-wise heatmaps
- Word clouds for clickbait vs. non-clickbait
- Composite score histograms and clustering plots

---

## Dependencies
Install required packages via pip:
```bash
pip install transformers pandas matplotlib seaborn scikit-learn textblob sentence-transformers wordcloud
```

---

## Authors
- Ayush Kuruvilla
- Sowmya Prakash  
For the course **DSAIT4055 – Web Science and Engineering**, TU Delft.
