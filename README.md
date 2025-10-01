# 📰 Fake News Detection

[![Python](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/)  
[![Notebook](https://colab.research.google.com/assets/colab-badge.svg)](Fake_News_Detection.ipynb)  
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle-orange)](https://www.kaggle.com/)  


## 📌 Overview  
This project demonstrates a **machine learning approach to detecting fake news articles**.  
The pipeline covers **data loading, preprocessing, feature extraction, model training, and evaluation**.  
The end goal is to build a robust classifier that can distinguish between **fake** and **true** news.

---

## 📂 Dataset  
- Source: [Kaggle Fake News Detection Dataset](https://www.kaggle.com/)  
- Files used:  
  - `Fake.csv` → contains fake news articles  
  - `True.csv` → contains genuine news articles  
- Combined into a single dataset with a new `label` column:  
  - `1` → Fake News  
  - `0` → True News  

---

## ⚙️ Steps Followed  

### 1. Import Libraries  
Core Python libraries for ML and NLP:  
- `numpy`, `pandas` → data handling  
- `matplotlib`, `seaborn` → visualization  
- `scikit-learn` → ML models & evaluation  
- `nltk` → text preprocessing  

---

### 2. Data Exploration (EDA)  
- Checked dataset shape, info, and missing values  
- Visualized class distribution  
- Created **word clouds** to understand common words in fake vs. true news  

📸 Example Word Clouds:  
![Fake News Word Cloud](images/fake_wordcloud.png)  
![True News Word Cloud](images/true_wordcloud.png)  

---

### 3. Data Preprocessing  
- Text cleaning: lowercasing, removing punctuation/special chars  
- Tokenization & stopword removal  
- Stemming/Lemmatization  
- Applied **TF-IDF vectorization** for feature extraction  

---

### 4. Model Building  
Trained and compared multiple ML models:  
- **Logistic Regression**  
- **Naive Bayes**  
- **Random Forest Classifier**  
- **Passive Aggressive Classifier**  

---

### 5. Model Evaluation  
Evaluated models on:  
- Accuracy  
- Precision, Recall, F1-score  
- Confusion Matrix  

📸 Example Confusion Matrix (Logistic Regression):  
![Confusion Matrix](images/confusion_matrix.png)  

📸 Model Comparison Accuracy Chart:  
![Model Comparison](images/model_accuracy.png)  

---

## 📊 Results  
- Best performing model: **Logistic Regression** (≈ 93–95% accuracy)  
- Naive Bayes worked well for smaller training sets but underperformed compared to Logistic Regression.  
- Passive Aggressive Classifier showed strong performance with fewer features.  

---

## 🚀 Future Improvements  
- Implement deep learning models (LSTMs, BERT for NLP).  
- Perform hyperparameter tuning for better accuracy.  
- Try ensemble learning (e.g., Voting Classifier).  

---

## 📦 How to Run  
1. Clone the repo and install requirements:  
   ```bash
   pip install -r requirements.txt
   ```
2. Run the notebook:  
   ```bash
   jupyter notebook Fake_News_Detection.ipynb
   ```
3. Follow through cells to preprocess data, train models, and view results.  

---

## 🙌 Acknowledgments  
- Dataset from Kaggle contributors.  
- Inspired by the growing need for **automated fake news detection** to combat misinformation.  


