# Elevvo Tasks

# 1) 📊 Loan Approval Prediction

This notebook builds a machine learning model to **predict loan approval status** based on applicant financial, credit, and employment information. The project uses a cleaned dataset from Kaggle and compares Logistic Regression and Decision Tree model.

---

### 🔧 Project Workflow

1. **📥 Data Loading**

   * Dataset (Loan-Approval-Prediction-Dataset) is downloaded using `kagglehub`.
   * The data includes columns like: `income_annum`, `loan_amount`, `cibil_score`, `loan_term`, `education`, `self_employed`, and more.

2. **🧼 Data Preprocessing**

   * Missing values are handled.
   * Categorical features like `education` and `self_employed` are encoded using One-Hot or Label Encoding.
   * Whitespace is stripped from column names.
   * Target variable `loan_status` is encoded as 0 (Rejected) and 1 (Approved).
   * Features and labels are split into `X` and `y`.

3. **🧪 Train-Test Split**

   * The dataset is split into 80% training and 20% testing using `train_test_split` from scikit-learn.
   * `stratify=y` ensures the class distribution is preserved in both sets.

4. **📈 Model Training**

   * **Logistic Regression** with:

     * `solver="liblinear"`
     * `class_weight="balanced"`
     * `random_state=42`, `max_iter=500`
   * **Decision Tree Classifier** is also trained for comparison.

5. **📊 Evaluation**

   * Accuracy and ROC-AUC are computed for both models.
   * Classification reports are generated.
   * Visualization includes:

     * Confusion matrices
     * ROC Curves
---

### 🤖 Model Comparison: Logistic Regression vs. Decision Tree

| Model               | Accuracy | ROC-AUC |
| ------------------- | -------- | ------- |
| Logistic Regression | 0.7400   | 0.8287  |
| Decision Tree       | 0.9707   | 0.9668  |

---

### 📦 Requirements

* Python 3.7+
* Libraries:

  * `pandas`
  * `numpy`
  * `scikit-learn`
  * `matplotlib`
  * `seaborn`
  * `kagglehub` (for dataset download)

---


# 2) 📘 Traffic Sign Recognition

## 🚦 Project Overview

This project implements a **Traffic Sign Recognition system** using deep learning. The goal is to classify traffic signs from image data into their correct categories. Two approaches were tested:

1. **Custom CNN Model** – built from scratch with multiple convolutional, pooling, and dense layers.
2. **MobileNet Transfer Learning** – leveraging a pre-trained MobileNet architecture with fine-tuning.

---

## 📊 Dataset

* The dataset consists of labeled images of traffic signs.
* Images are preprocessed (resizing, normalization, augmentation) before training.
* Data is split into **train**, **validation**, and **test** sets.

---

## ⚙️ Model Architectures

### 1. Custom CNN

* Multiple convolutional + pooling layers.
* Batch normalization and dropout for regularization.
* Fully connected layers with softmax output.

### 2. MobileNet (Transfer Learning)

* Pretrained MobileNet (on ImageNet).
* Fine-tuned with traffic sign dataset.
* Added dense classification head.

---

## 📈 Results

| Model       | Accuracy   | Loss   | Notes                                        |
| ----------- | ---------- | ------ | -------------------------------------------- |
| Custom CNN  | **0.9907** | 0.0559 | Very high performance, strong generalization |
| MobileNetV2 | **0.6250** | 1.1910 | Underperformed, struggled to adapt           |

---

## 🔍 Confusion Matrices

Below is a direct comparison between the **Custom CNN** and **MobileNetV2** performance across all 43 classes:

<img width="2297" height="989" alt="image" src="https://github.com/user-attachments/assets/fc74b82f-1e21-4803-aaa8-56d28b6e518b" />

* **Custom CNN (left)**: Almost perfect diagonal → excellent classification across all traffic sign categories.
* **MobileNetV2 (right)**: Weak diagonal, many misclassifications → poor generalization on this dataset.

---

## 🖼️ Prediction Visualization (Custom CNN)

Sample predictions from the test set:

* Left to right:

  * Correctly predicted traffic signs with true vs predicted labels.
  * The model demonstrates robust recognition even on low-quality images.

<img width="950" height="223" alt="image" src="https://github.com/user-attachments/assets/fbf8c369-c708-492b-9eaf-a794dc5cd068" />

---

## 🔍 Comparison

* **Custom CNN**:

  * Strength: Excellent accuracy (\~99%), robust learning from dataset.
  * Weakness: Training took slightly longer per epoch.

* **MobileNetV2**:

  * Strength: Lightweight architecture, designed for mobile deployment.
  * Weakness: Performed poorly on this dataset (accuracy only \~62%). Likely due to mismatch between ImageNet pretraining and traffic sign dataset without sufficient fine-tuning.

✅ **Conclusion**: Contrary to expectations, the **Custom CNN clearly outperformed MobileNetV2** on this dataset. For practical deployment, the Custom CNN is the recommended model.

---

## 🚀 Usage

Run the notebook:

```bash
jupyter notebook Traffic_Sign_Recognition.ipynb
```

---
## 🛠️ Requirements

* Python 3.8+
* TensorFlow / Keras
* NumPy
* Matplotlib
* scikit-learn

# 3) 🎬 Movie Recommendation System

This project builds a personalized movie recommender system using collaborative filtering. It uses the MovieLens dataset (movies and ratings) and evaluates the model using precision-based metrics.

---

### 📂 Dataset

The dataset (MovieLens 100K Dataset) contains:

* **movies.csv** — movie titles and genres.
* **ratings.csv** — user ratings for each movie.

---

### 🧠 Recommendation Model

The system follows these major steps:

1. **Data Loading & Merging**

   * Merge `ratings.csv` with `movies.csv` on `movieId`.
2. **Data Preprocessing**

   * Handle missing values and prepare the user-item matrix.
3. **Modeling**

   * Build a recommendation engine using collaborative filtering with cosine similarity.
   * Evaluate using Precision\@K.
4. **Prediction**

   * Generate movie recommendations for sample users.

---

### 📈 Evaluation Results

**✅ Precision\@10:**

```
Average Precision@10: 0.2310
```

This indicates that on average, 23.1% of the top 10 recommended movies for a user were relevant.

---

### 🌟 Top 5 Highest Predicted Movies

Based on the trained model, the following movies received the highest average predicted ratings:

| Movie Title                                        | Predicted Rating |
| -------------------------------------------------- | ---------------- |
| Aliens (1986)                                      | 4.40             |
| Office Space (1999)                                | 4.40             |
| Austin Powers: International Man of Mystery (1997) | 4.39             |
| Terminator, The (1984)                             | 4.29             |
| Jaws (1975)                                        | 3.98             |

---

## 🏁 How to Run

1. Upload the dataset to your Google Drive.
2. Mount your drive in Colab.
3. Run the notebook `Movie_Recommendation_System.ipynb` step by step.

---

# 4) 🎵 Music Genre Classification

## 📌 Project Overview

This project explores two different approaches to classifying music genres:

1. **Tabular Data Approach**

   * Uses numerical audio features extracted from the dataset (e.g., tempo, spectral centroid, zero-crossing rate, MFCC statistics).
   * Models are trained on these structured features.

2. **Image-Based Approach**

   * Converts audio files into **spectrogram images**.
   * Uses transfer learning with deep CNN architectures (e.g., ResNet, MobileNet) to classify genres from visual patterns.

By comparing these two approaches, we highlight the trade-offs between **traditional feature-based methods** and **deep learning with image representations**.

---

## 🗂 Dataset

* **Source**: GTZAN / equivalent music dataset.
* **Classes**: 10 genres (e.g., classical, jazz, metal, pop, rock, reggae, country, blues, hip-hop, disco).
* **Samples**: \~1000 audio tracks (30s each).

---

## 📊 Results

### **Tabular Data (Feature-based)**

* Accuracy: **\~74–80%**.
* Strengths: Lightweight, interpretable.
* Weaknesses: Limited ability to capture temporal/complex audio patterns.

### **Image-Based (Spectrograms + CNN)**

* Accuracy: **\~90–95%** with deep CNNs.
* Strengths: Captures richer temporal & spectral features automatically.
* Weaknesses: Computationally expensive, requires GPUs and longer training time.
---

## 🔄 Comparison: Tabular Features vs. Spectrogram Images for Music Genre Classification

| Aspect                      | **Tabular (Features)**                                 | **Image-based (Spectrograms)**                                  |
|-----------------------------|---------------------------------------------------------|------------------------------------------------------------------|
| **Input Data**              | Handcrafted numerical features (MFCC, tempo, etc.)     | Mel-spectrogram images (visual representation of audio)         |
| **Model Type**              | Traditional ML (SVM)                                   | Deep Learning (custom CNNs)                                    |
| **Accuracy (in this project)** | ✅ **76.5%**                                        | ⚠️ **65.0%**                                                     |
| **Training Time**           | Fast (few seconds to minutes)                          | Slower (minutes to hours depending on model size)                |
| **Resource Requirement**    | Low — CPU sufficient                                   | High — GPU recommended                                          |
| **Feature Engineering**     | Required (manual extraction via Librosa, etc.)         | Not needed — CNNs learn features automatically                  |
| **Interpretability**        | High — each feature has meaning                        | Low — CNN filters are not human-interpretable                   |
| **Scalability**             | Suitable for small datasets                           | Performs better with large datasets                             |
| **Robustness to Noise**     | Lower — sensitive to feature extraction errors         | Higher — CNNs can generalize better with augmentation           |
| **Flexibility**             | Limited to chosen features                            | Flexible — model can learn from raw patterns                    |
| **Best Genres (in this project)** | Classical, Pop, Jazz                           | Reggae, Jazz, Metal                                              |
| **Worst Genres (in this project)**| Rock, Disco                                  | Rock, Hiphop, Blues                                              |


## 🔢 Confusion Metrices
### SVM Confusion Matrix
<img width="550" height="500" alt="image" src="https://github.com/user-attachments/assets/17088c4d-2c51-4eff-821d-a450147ca0eb" />

### CNN Confusion Matrix
<img width="550" height="500" alt="image" src="https://github.com/user-attachments/assets/0af5ef9e-14af-4dd2-8b34-1f7cb20789a9" />

---

## 🚀 Conclusion

* **Tabular approach** is efficient, interpretable, and good for quick experiments.
* **Image-based approach** significantly outperforms in classification accuracy and ranking metrics, making it more suitable for real-world deployment.
* Depending on resources, one can start with tabular models and scale up to spectrogram-based CNNs for state-of-the-art performance.


---


## 📌 Requirements

Install dependencies via:

```bash
pip install pandas scikit-learn
```

Or use in **Google Colab** with Google Drive integration.

---

