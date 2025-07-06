# Fake vs Real News Classifier

Detecting Misinformation Using Textual and Temporal Features

## Project Overview

This project aims to classify news articles as either fake or real, leveraging a combination of **textual features**, **temporal features**, and **sentiment analysis**. By analyzing both the **content** and the **timing** of news posts, this project builds a robust and interpretable fake news detection system.

Built with the “Fake and real news dataset” from Kaggle (featuring ~23.5k fake and ~21.4k real articles), the project emphasizes **explainability** through SHAP visualizations, offering insights into the model’s decision-making.

---

## Objective

- Predict whether a given news article is fake or real.
- Combine **textual features** (e.g., content length, exclamation count, sentiment) with **temporal features** (e.g., posting hour, day of week, weekend flag).
- Ensure model transparency through **SHAP** visualization.

---

## Dataset

- **Source**: Kaggle user Clement Bisaillon’s “Fake and real news dataset”  
  - Contains `Fake.csv` (~23,502 fake articles) and `True.csv` (~21,417 real articles)  
  - Dataset link: [Kaggle – Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- These are merged into a single dataset with binary labels: 0 = real, 1 = fake.

---

## Features Engineered

| Feature Type         | Examples                                           |
|----------------------|----------------------------------------------------|
| **Temporal**         | Posting hour, Day of week, Weekend flag           |
| **Content-based**    | Content length, Exclamation marks, Uppercase words |
| **Sentiment**        | VADER sentiment scores                            |
| **Keyword Flags**    | Counts of “breaking”, “urgent”, “shocking”, etc.  |
| **Text Structure**   | Number of links, Hashtags                         |

---

## Novelty in Approach

- **Keyword Signals**: Quantifies sensational or urgent keywords often seen in fake/news clickbait.
- **Temporal + Content Fusion**: Merges linguistic and timing signals—many fake articles display distinct posting patterns.
- **Explainability with SHAP**: Enables feature-level transparency for each model prediction.
- **Sentiment Integration**: Captures emotional tone differences between fake and real news.

---

## Models Used

- **Logistic Regression**
- **Random Forest**
- **XGBoost** (top performer)
- **Evaluation Metrics**:
  - Accuracy, Precision, Recall, F1-score, ROC-AUC
- **Interpretability**:
  - SHAP summary and waterfall plots for model insights.

---

## Results

- XGBoost leads in both AUC and F1‑score.
- SHAP analysis indicates the strongest predictors are:
  - Content length
  - Exclamation count
  - Keyword flags

---

## Folder Structure

```
fake-vs-real-news/
┣ 📂 data/
┃ ┣ Fake.csv
┃ ┣ True.csv
┃ ┣ final_dataset.csv
┃ ┗ features.csv
┣ 📂 scripts/
┃ ┣ preprocess.py
┃ ┣ feature_engineering.py
┃ ┗ model.py
┣ 📂 notebooks/
┃ ┗ analysis.ipynb
┣ 📄 model.xgb
┣ 📄 requirements.txt
┗ 📄 README.md
```

---

## Setup Instructions

1. **Clone the repo**  
2. **Create & activate** a virtual environment  
3. **Install dependencies**:  
   ```bash
   pip install -r requirements.txt
   ```  
4. **Run the pipeline**:  
   ```bash
   python scripts/preprocess.py
   python scripts/feature_engineering.py
   python scripts/model.py
   ```

---

## Dependencies

- Python 3.8+
- pandas, numpy  
- scikit-learn  
- xgboost  
- shap  
- matplotlib, seaborn  
- nltk

---

## Notes

- Modular and extensible design for easy upgrades (e.g., transformer models).
- SHAP plots bridge performance and transparency.
- Suitable for adaptation to other binary text classification tasks.

---

## License

Open for academic and research use. For commercial purposes, please contact the author.

---

## Dataset Link

Get the dataset on Kaggle:  
[https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

---

## Visuals (in `images/`)

### 1. Content Length vs Label
![Content Length vs Label](images/content_length_vs_label.png)  
*Distribution of content length across fake vs real news.*

---

### 2. Model Evaluation
![Confusion Matrix](images/confusion_matrix.png)  
*Confusion matrix showing model performance metrics.*

---

### 3. SHAP Summary Plot
![SHAP Summary Plot](images/shap_summary_plot.png)  
*SHAP summary plot showing top features influencing predictions.*

---

### 4. SHAP Waterfall Plot
![SHAP Waterfall Plot](images/shap_waterfall_plot.png)  
*Waterfall plot explaining how individual features affect a single prediction.*

---

### 5. SHAP Dependence Plot
![SHAP Dependence Plot](images/shap_dependence_plot.png)  
*SHAP dependence plot showing feature interactions and their influence.*
