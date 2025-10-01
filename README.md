# Customer Sentiments For Safaricom and Airtel Analysis 📱  
**Kenya Telco App Reviews – Capstone Project**  

**Authors:** Jedidah Kathure, Winnie Amoit, Antony Njoroge, Rachel Odhiambo, Anne Mumbe, Navros Kamau  
![alt text](image.png)
---

## 📌 Overview  
Customer Sentiment & Complaint Analysis from Google Play and App Store Reviews  

Telecommunication giants **Safaricom** and **Airtel** serve millions of Kenyans through their mobile apps, powering critical services such as M-Pesa, Airtel Money, airtime purchases, data bundles, and customer self-service. Yet, user experiences vary widely, and customers are quick to voice their experiences in app-store reviews.  

This project analyzes thousands of Safaricom and Airtel reviews using **Natural Language Processing (NLP)** to:  
- Classify sentiment (**positive, neutral, negative**)  
- Detect common complaints (network reliability, login/OTP issues, mobile money performance, etc.)  
- Provide **data-driven recommendations** for product managers and CX teams  

---

## 📌 Business Understanding  
Kenya’s telecom giants Safaricom and Airtel serve millions who rely on their apps for mobile money, airtime, data bundles, and customer self-service.  

App-store reviews provide authentic, unfiltered customer feedback on issues such as:  
- Network reliability  
- Data bundles  
- M-Pesa / Airtel Money  
- Login & OTP problems  
- App usability  

By applying NLP techniques, we can uncover patterns, detect problems, and deliver actionable insights to enhance customer experience.  

**Impact:**  
- ✅ Detect outages & major complaints in near real-time  
- ✅ Support product/feature decisions  
- ✅ Reduce churn through better CX  
- ✅ Strengthen brand loyalty  

---

## 👥 Stakeholders  
- **Executives / Business Leaders** → Align product strategy with customer needs.  
- **Product Managers** → Prioritize features and bug fixes based on real user feedback.  
- **Customer Experience (CX) Teams** → Detect pain points early and improve retention.  
- **Marketing Teams** → Monitor brand perception and sentiment shifts.  
- **Data Science & Engineering Teams** → Build scalable monitoring and analytics pipelines.  
- **Regulators & Industry Analysts** → Understand telco market competitiveness.  

---

## 💡 Business Value  
This project provides immediate, tangible value by translating noisy, unstructured app reviews into **actionable intelligence**:  

- **Customer Retention** → Reduce churn by addressing common pain points. Detect complaints rapidly and engage dissatisfied customers before they defect.  
- **Operational Efficiency** → Detect service outages or app issues in near real-time. Spikes in negative sentiment serve as an early warning system.  
- **Strategic Decision-Making** → Insights for bundles, pricing, and service design.  
- **Brand Loyalty & Trust** → Use customer feedback transparently to improve services.  
- **Competitive Benchmarking** → Compare Safaricom vs Airtel sentiment trends.  

---

## 🎯 Objectives  
- **Sentiment Classification** → Positive, Negative, Neutral  
- **Theme & Topic Mining** → Identify major complaint categories  
- **Trend Analysis** → Monitor issues over time  
- **Benchmarking** → Compare Safaricom vs Airtel customer satisfaction  

---

## 📂 Data Understanding  

### Data Source  
- Google Play Store & Apple App Store reviews  

**Data Includes:**  
- Review text  
- Star ratings  
- Date of review  
- App metadata  

### Data Characteristics  
- **Size:** Thousands of reviews across both platforms  
- **Features:** Unstructured text, numeric star ratings  
- **Target:** Sentiment classification (positive, neutral, negative)  

### Data Quality Issues  
- Duplicates and spam reviews (removed during cleaning)  
- Mixed languages (English, Swahili, Sheng) → custom tokenization & translation  
- Typos & informal text → normalization and lemmatization  

### Exploratory Insights  
- **Polarity:** Safaricom reviews show strong polarity (many highly positive, many highly negative).  
- **Airtel Focus:** Frequent login/OTP issues and customer service frustrations.  
- **Common Keywords:** “network”, “data”, “M-Pesa”, “login”, “OTP”.  

---

## 📁 Repository Navigation  


---

## 📂 Project Workflow  
1. **Data Collection** → Scraping reviews from Google Play & App Store  
2. **Preprocessing** → Cleaning, tokenization, lemmatization  
3. **Exploratory Data Analysis (EDA)** → Word clouds, sentiment trends  
4. **Modeling**  
   - Baseline ML (Logistic Regression, Naive Bayes, SVM using TF-IDF)  
   - Transformer models (BERT & variants)  
5. **Evaluation** → Accuracy, Precision, Recall, F1 (focus on **Negative Recall**)  
6. **Visualization & Insights** → Complaint categories, dashboards  

---

## 🛠️ Tech Stack  
- **Languages:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, NLTK, SpaCy, Matplotlib, Seaborn, WordCloud, HuggingFace Transformers  
- **Data Sources:** Google Play Store, Apple App Store  

---

## 📊 Exploratory Data Analysis (EDA)  

### Word Clouds  
- **Safaricom Negative Reviews:** Dominated by *M-Pesa, data, network*.  

![alt text](image.png)

- **Airtel Negative Reviews:** Frequent *login, OTP, code* issues. 

![alt text](image-1.png)

### Sentiment Distribution  

![alt text](image-2.png)
- ~80% Positive, 15% Negative, 5% Neutral → heavy class imbalance.  

### Trends Over Time  
- Negative sentiment spikes during major outages or service interruptions.  

![alt text](image-3.png)
---

## 📈 Model Performance  

### 🔹 Baseline Models (Traditional ML)  

| Model                       | Accuracy | Macro F1 | Negative Recall (Complaint Detection) |
|------------------------------|----------|----------|---------------------------------------|
| Logistic Regression (Baseline) | 0.86     | 0.49     | Low (biased toward Positive)           |
| Naive Bayes (Baseline)         | 0.82     | 0.42     | Low (biased toward Positive)           |
| **LR (Class Weighted) – FINAL**| 0.61     | 0.32     | **0.72 (Best Score) ✅**               |
| SVM (LinearSVC)                | 0.85     | 0.47     | Low (biased toward Positive)           |

**Interpretation:**  
- Baseline models had high accuracy but failed to detect complaints (low Negative Recall).  
- The **Class Weighted LR model** improved Negative Recall dramatically, making it the best choice for **V1 deployment**.  

---

### 🔹 Deep Learning (BERT-based)  

| Metric      | Value |
|-------------|-------|
| Accuracy    | 0.92  |
| Macro F1    | 0.78  |
| Weighted F1 | 0.91  |

✅ **Improvement over ML baselines:**  
- BERT outperformed traditional models, handling nuanced neutral reviews better and balancing classification across all classes.  
- Sets the benchmark for **V2 deployment** (once resources allow).  

---

## 📊 Visual Results  

### Confusion Matrices  
- **Baseline LR:** Correctly classifies positives, but misses most negatives.  

- **BERT:** Balanced classification across all sentiment classes.  

### Classification Report  
- Highlights **Negative Recall** as the key metric justifying **Class Weighted LR** for V1.  

![alt text](image-4.png)

---

## 📊 Key Insights  
- Positive reviews dominate both Safaricom & Airtel apps.  
- Neutral reviews were the hardest to classify → BERT improved detection significantly.  
- Safaricom complaints center on **M-Pesa reliability**, Airtel complaints on **login & OTP issues**.  

---

## 🤖 Final Model Selection  

### Decision:  
**Logistic Regression (Class Weighted)** → Selected for **V1 Deployment**.  

**Why LR (Weighted):**  
- **High Complaint Detection:** Negative Recall = **0.72**  
- **Resource Efficient:** Lightweight, fast, easy to maintain in production  
- **Immediate Business Value:** Meets the primary goal of detecting complaints  

### Future Target:  
**BERT (Transformer Model)** → Reserved for **V2 Deployment** when infrastructure can support it.  

---

## ⚙️ Setup Instructions  

```bash
# Clone the repository
git clone https://github.com/your-username/safaricom-airtel-analysis.git
cd safaricom-airtel-analysis

# Create and activate venv
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch notebook
jupyter notebook safaricom_airtel_analysis.ipynb
