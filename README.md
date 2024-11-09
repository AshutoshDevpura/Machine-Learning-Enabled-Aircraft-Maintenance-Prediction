
# ✈️ Predictive Analytics Using Supervised ML for Aircraft Maintenance 🚀

This project leverages cutting-edge machine learning techniques to predict aircraft maintenance needs, helping prevent unexpected downtimes and reducing costs! Built using Streamlit for interactive visualization, this application uses advanced models such as Random Forest, Gradient Boosting, AdaBoost, and XGBoost to provide reliable maintenance predictions and insights.


## Table of Contents

1. [Project Overview](#project-overview)
2. [Why This Project Matters](#why-this-project-matters)
3. [Features](#features)
4. [Installation and Setup](#installation-and-setup)
5. [Usage](#usage)
6. [Streamlit Deployment](#streamlit-deployment)
7. [Model Training and Evaluation](#model-training-and-evaluation)
8. [Contributing](#contributing)
9. [License](#license)

---

## Project Overview

This project predicts aircraft maintenance needs using supervised learning techniques, focusing on optimizing schedules, increasing uptime, and minimizing costs. An interactive dashboard built with Streamlit and Plotly allows users to explore data, tune hyperparameters, and view evaluation metrics.

## Why This Project Matters 🌍

In the aviation industry, **unplanned maintenance** can lead to flight delays, passenger dissatisfaction, and significant financial losses. **Predictive maintenance** aims to address these challenges by predicting potential issues before they arise.

### Key Benefits:
- **🚀 Improved Safety**: Ensures aircraft are always in optimal condition.
- **💰 Cost Savings**: Reduces unplanned downtime and maintenance expenses.
- **📈 Higher Efficiency**: Streamlines maintenance scheduling and operations.
- **🌱 Environmental Impact**: Reduces wasted resources by optimizing parts and servicing needs.

By predicting maintenance needs, this project helps airlines and aviation companies operate more reliably, making the skies safer and operations smoother.

## Features

- **Data Preprocessing**: Includes scaling, handling imbalances with `SMOTE`, and feature reduction with `PCA`.
- **Model Training**: Trains and compares RandomForest, GradientBoosting, AdaBoost, and XGBoost models.
- **Hyperparameter Tuning**: Automated tuning with `Optuna` for optimal model performance.
- **Evaluation Metrics**: Displays accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrix.
- **Interactive Dashboard**: User-friendly UI built with Streamlit and Plotly for a seamless experience.

## Installation and Setup 🛠️

### 1. Clone the Repository

```bash
git clone https://github.com/AshutoshDevpura/Predictive-Analytics-using-supervised-ML-for-Aircraft-Maintenance.git
cd Predictive-Analytics-using-supervised-ML-for-Aircraft-Maintenance
```

### 2. Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/MacOS
venv\Scripts\activate     # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the App 🚀

```bash
streamlit run app.py
```

## Usage 🧑‍💻

1. **Upload Dataset**: Upload your CSV dataset through the app UI.
2. **Data Preprocessing**: Automatic scaling, PCA, and SMOTE adjustments are applied to the data.
3. **Model Selection**: Choose from a variety of classifiers (RandomForest, GradientBoosting, AdaBoost, XGBoost) for training and evaluation.
4. **Hyperparameter Tuning**: Optuna can be used to fine-tune model parameters.
5. **Model Evaluation**: View performance metrics through interactive charts (confusion matrix, ROC curve, etc.).
6. **Save Model**: Save the trained model for future use.

### Streamlit App UI Screenshots
<img width="1384" alt="Screenshot 2024-11-09 at 12 47 05 AM" src="https://github.com/user-attachments/assets/b8d2443d-dcb1-4ffe-9116-dcb5b76f5fa4">


<img width="1345" alt="Screenshot 2024-11-09 at 12 48 08 AM" src="https://github.com/user-attachments/assets/354c15f2-899d-43af-a56d-abdeaaf5db3e">



<img width="1353" alt="Screenshot 2024-11-09 at 12 51 55 AM" src="https://github.com/user-attachments/assets/f809277f-aefc-49c8-a537-099e89464f88">



<img width="1345" alt="Screenshot 2024-11-09 at 12 51 17 AM" src="https://github.com/user-attachments/assets/6f380dd7-04d8-44bb-90d3-a13e1fc5aa21">




## Streamlit Deployment 🌐

Deploy the app via Streamlit using GitHub:

1. **Push Code to GitHub**: Ensure your code is committed and pushed.
2. **Log in to Streamlit Cloud**: Go to [Streamlit Cloud](https://share.streamlit.io/) and log in with GitHub.
3. **Deploy**: Select your repository and deploy. Streamlit will handle all server-side configurations.

## Model Training and Evaluation 📊

### Training Pipeline
1. **Data Split**: Split into training and testing sets.
2. **Model Selection**: Train various classifiers and compare their performance.
3. **Cross Validation**: Uses `cross_val_score` for validation.
4. **Metrics Calculation**: Calculates `accuracy`, `precision`, `recall`, `F1-score`, and `AUC-ROC`.

### Hyperparameter Tuning with Optuna
`Optuna` optimizes hyperparameters for each model to improve prediction accuracy.

### Model Persistence
To save a model for later use:

```python
joblib.dump(model, 'model.pkl')
```

## Contributing 🤝

1. Fork the repository.
2. Create a new branch (`feature/YourFeature`).
3. Commit changes.
4. Push to the branch.
5. Create a pull request.

## License 📝

This project is licensed under the MIT License.
