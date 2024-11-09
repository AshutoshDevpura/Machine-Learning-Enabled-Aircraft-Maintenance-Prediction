
# Author : Ashutosh Devpura
# email: ashutoshdevpura@gmail.com


# Import Libraries 
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import optuna
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
import numpy as np
import joblib

# Set Streamlit page configuration
st.set_page_config(
    page_title="Machine Learning-Enabled Aircraft Maintenance Prediction",
    layout="wide"
)

# Title with airplane icon and styling
st.markdown(
    """
    <div style="background-color:#0b3d91;padding:20px;border-radius:15px; margin-bottom: 20px;">
        <h1 style="color:white;text-align:center;font-family:Verdana, Geneva, sans-serif;">✈️ Machine Learning-Enabled Aircraft Maintenance Prediction</h1>
        <p style="color:white;text-align:center;font-size:16px;">Enhancing Aviation Safety through Machine Learning</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Load the dataset with a loading spinner
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

with st.spinner("✈️ Loading data, please wait..."):
    file_path = "/Users/ashutoshdevpura/Desktop/Projects/Airplane_Engine_ML/aircraft_maintenance_dataset - aircraft_maintenance_dataset.csv"
    supervised_learning_df = load_data(file_path)

# Data Preprocessing
missing_values = supervised_learning_df.isnull().sum()
missing_column = missing_values[missing_values > 0].index
supervised_learning_df.fillna(supervised_learning_df[missing_column].mean(), inplace=True)

# Feature Engineering
selected_features = [
    'Temperature', 'Pressure', 'Rotational_Speed', 'Engine_Health', 
    'Fuel_Consumption', 'Vibration_Level', 'Oil_Temperature', 
    'Altitude', 'Humidity', 'Pressure_Temperature', 
    'Vibration_Rotational_Speed', 'Fuel_to_Rotational_Ratio', 
    'Altitude_to_Humidity_Ratio', 'Rel_Fuel_Altitude_Difference'
]
supervised_learning_df['Pressure_Temperature'] = supervised_learning_df['Pressure'] * supervised_learning_df['Temperature']
supervised_learning_df['Vibration_Rotational_Speed'] = supervised_learning_df['Vibration_Level'] * supervised_learning_df['Rotational_Speed']
supervised_learning_df['Fuel_to_Rotational_Ratio'] = supervised_learning_df['Fuel_Consumption'] / supervised_learning_df['Rotational_Speed']
supervised_learning_df['Altitude_to_Humidity_Ratio'] = supervised_learning_df['Altitude'] / supervised_learning_df['Humidity']
supervised_learning_df['Rel_Fuel_Altitude_Difference'] = supervised_learning_df['Fuel_Consumption'] - supervised_learning_df['Altitude']

# Train-Test Split
X = supervised_learning_df[selected_features]
y = supervised_learning_df['Engine_Failure']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply SMOTE to balance the classes
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# PCA for Dimensionality Reduction
pca = PCA(n_components=8)
X_train_pca = pca.fit_transform(X_train_resampled)
X_test_pca = pca.transform(X_test_scaled)

# Model Training with Optuna for Hyperparameter Tuning
MODEL_PATH = "best_model.pkl"

@st.cache_resource
def train_models(X_train, y_train):
    def objective(trial):
        classifier_name = trial.suggest_categorical("classifier", ["RandomForest", "GradientBoosting", "AdaBoost", "XGBoost"])
        if classifier_name == "RandomForest":
            n_estimators = trial.suggest_int("n_estimators", 100, 300)
            max_depth = trial.suggest_int("max_depth", 10, 30)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, random_state=42, n_jobs=-1)
        elif classifier_name == "GradientBoosting":
            n_estimators = trial.suggest_int("n_estimators", 100, 300)
            learning_rate = trial.suggest_loguniform("learning_rate", 0.01, 0.2)
            max_depth = trial.suggest_int("max_depth", 3, 10)
            model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=42)
        elif classifier_name == "AdaBoost":
            n_estimators = trial.suggest_int("n_estimators", 50, 200)
            learning_rate = trial.suggest_loguniform("learning_rate", 0.01, 1.0)
            model = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)
        elif classifier_name == "XGBoost":
            n_estimators = trial.suggest_int("n_estimators", 100, 300)
            learning_rate = trial.suggest_loguniform("learning_rate", 0.01, 0.3)
            max_depth = trial.suggest_int("max_depth", 3, 10)
            scale_pos_weight = trial.suggest_loguniform("scale_pos_weight", 1, 10)
            model = XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, scale_pos_weight=scale_pos_weight, random_state=42, use_label_encoder=False, eval_metric='logloss')
        return cross_val_score(model, X_train, y_train, n_jobs=-1, cv=3, scoring="f1").mean()
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)
    best_params = study.best_params
    print(f"Best parameters found: {best_params}")
    
    if best_params['classifier'] == "RandomForest":
        best_model = RandomForestClassifier(n_estimators=best_params['n_estimators'], max_depth=best_params['max_depth'], min_samples_split=best_params['min_samples_split'], random_state=42, n_jobs=-1)
    elif best_params['classifier'] == "GradientBoosting":
        best_model = GradientBoostingClassifier(n_estimators=best_params['n_estimators'], learning_rate=best_params['learning_rate'], max_depth=best_params['max_depth'], random_state=42)
    elif best_params['classifier'] == "AdaBoost":
        best_model = AdaBoostClassifier(n_estimators=best_params['n_estimators'], learning_rate=best_params['learning_rate'], random_state=42)
    elif best_params['classifier'] == "XGBoost":
        best_model = XGBClassifier(n_estimators=best_params['n_estimators'], learning_rate=best_params['learning_rate'], max_depth=best_params['max_depth'], scale_pos_weight=best_params['scale_pos_weight'], random_state=42, use_label_encoder=False, eval_metric='logloss')
    
    best_model.fit(X_train_pca, y_train_resampled)
    joblib.dump(best_model, MODEL_PATH)
    return best_model

try:
    best_model = joblib.load(MODEL_PATH)
    st.success("🛫 Loaded pre-trained model!")
except FileNotFoundError:
    with st.spinner("🛫 Training models with SMOTE-adjusted data using Optuna for hyperparameter optimization..."):
        best_model = train_models(X_train_pca, y_train_resampled)
        st.success("🛫 Model training completed!")


# Generate predictions specific to each model
y_pred = best_model.predict(X_test_pca)

# Calculate and display metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=1)
recall = recall_score(y_test, y_pred, zero_division=1)
f1 = f1_score(y_test, y_pred, zero_division=1)

# Set up the layout
col1, col2 = st.columns(2)

# Define a bluish color palette
color_palette = ['#E6F3FF', '#99D6FF', '#4DB8FF', '#0080FF', '#005CE6']

# Confusion Matrix
with col1:
    st.markdown("<div style='padding-right:10px;'></div>", unsafe_allow_html=True)
    matrix = confusion_matrix(y_test, y_pred)
    fig = go.Figure(data=go.Heatmap(
        z=matrix, 
        x=['No Failure', 'Failure'], 
        y=['No Failure', 'Failure'], 
        colorscale=color_palette,
        showscale=True
    ))
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted Label',
        yaxis_title='True Label',
        font=dict(family='Arial', size=14),
        title_x=0.5,
        title_font=dict(size=18),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)

# ROC Curve
with col2:
    if hasattr(best_model, "predict_proba"):
        y_proba = best_model.predict_proba(X_test_pca)[:, 1]
    else:
        y_proba = best_model.decision_function(X_test_pca)
        y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min())

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_score = roc_auc_score(y_test, y_proba)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'AUC = {auc_score:.2f}',
        line=dict(color=color_palette[-1], width=2)
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Baseline',
        line=dict(color='gray', dash='dash', width=1)
    ))
    fig.update_layout(
        title='ROC Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        font=dict(family='Arial', size=14),
        title_x=0.5,
        title_font=dict(size=18),
        xaxis=dict(showgrid=True, gridcolor='lightgray'),
        yaxis=dict(showgrid=True, gridcolor='lightgray'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(x=0.05, y=0.95)
    )
    st.plotly_chart(fig, use_container_width=True)

# Real-Time Prediction
st.markdown(
    """
    ## Predict Maintenance Need
    <div style="background-color:#002244;padding:15px;border-radius:15px;">
        <h2 style="color:#ffffff;text-align:center;font-family:'Courier New';">Enter Feature Values for Prediction</h2>
        <p style="color:#e6e6e6;text-align:center;">Mimic the controls of an aircraft's dashboard by adjusting the features below:</p>
    </div>
    """,
    unsafe_allow_html=True
)
with st.expander("Enter values for each feature to get a prediction from the selected model:"):
    # Prediction input form
    user_data = {feature: st.slider(f"{feature}", float(X[feature].min()), float(X[feature].max()), step=(float(X[feature].max()) - float(X[feature].min())) / 100) for feature in selected_features}
    user_df = pd.DataFrame([user_data])
    user_scaled = scaler.transform(user_df)
    user_pca = pca.transform(user_scaled)

# Adjustable Threshold Slider
st.markdown(
    """
    ### Adjust Prediction Threshold
    <div style="background-color:#002244;padding:10px;border-radius:10px;margin-top:10px;">
        <h3 style="color:#ffffff;font-family:'Courier New';">Decision Threshold Slider</h3>
        <p style="color:#e6e6e6;">Adjust the threshold to simulate cockpit decision-making:</p>
    </div>
    """,
    unsafe_allow_html=True
)
threshold = st.slider("Set the decision threshold for predicting engine failure ✈️:", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

# Improved UI for Prediction Result
if st.button("Predict ✈️"):
    with st.spinner("🛫 Making prediction..."):
        if hasattr(best_model, "predict_proba"):
            prediction_proba = best_model.predict_proba(user_pca)[:, 1]
        else:
            prediction_proba = best_model.decision_function(user_pca)
            prediction_proba = (prediction_proba - prediction_proba.min()) / (prediction_proba.max() - prediction_proba.min())  # Normalize decision function to [0, 1]
        
        prediction = (prediction_proba >= threshold).astype(int)
        prediction_proba = np.round(prediction_proba, 2)
        # Limit the confidence score to a maximum of 0.99 to avoid unrealistic results
        prediction_proba = np.clip(prediction_proba, 0.01, 0.99)
    
    # Display Dial-like Indicator for Prediction
    fig = go.Figure()
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=prediction_proba[0],
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Engine Failure Probability"},
        gauge={
            'axis': {'range': [0, 1]},
            'steps': [
                {'range': [0, 0.3], 'color': "#00bfff"},  # Blue for "Unlikely Failure"
                {'range': [0.3, 0.7], 'color': "#f0e68c"},  # Yellow for "Caution"
                {'range': [0.7, 1], 'color': "#ff4500"}   # Red for "Likely Failure"
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': prediction_proba[0]
            },
            'bar': {'color': "black"}  # Use bar property to add a needle-like effect
        }
    ))
    fig.update_layout(
        paper_bgcolor="#002244",
        font=dict(family="Courier New", color="white")
    )
    st.plotly_chart(fig)
    
    if prediction[0] == 1:
        # Likely Engine Failure
        st.subheader("🚨 Prediction Result: Likely Engine Failure!")
        st.markdown(
            """
            <div style="background-color:#ff4500;padding:15px;border-radius:10px;">
                <h3 style="color:white;text-align:center;font-family:'Courier New';">Warning! Potential Engine Failure Detected!</h3>
                <p style="color:white;text-align:center;">Please take immediate action. The system has detected signs that may indicate an engine failure risk.</p>
                <p style="color:white;text-align:center;"><strong>Confidence Score:</strong> {confidence:.2f}</p>
            </div>
            """.format(confidence=prediction_proba[0]),
            unsafe_allow_html=True
        )
    else:
        # Unlikely Engine Failure
        st.subheader("✅ Prediction Result: Engine Condition Stable")
        st.markdown(
            """
            <div style="background-color:#00bfff;padding:15px;border-radius:10px;">
                <h3 style="color:white;text-align:center;font-family:'Courier New';">Engine Status: Stable</h3>
                <p style="color:white;text-align:center;">The engine condition appears to be stable. No immediate maintenance required.</p>
                <p style="color:white;text-align:center;"><strong>Confidence Score:</strong> {confidence:.2f}</p>
            </div>
            """.format(confidence=1 - prediction_proba[0]),
            unsafe_allow_html=True
        )


# Footer
st.markdown(
    """
    <hr>
    <div style="text-align:center;font-family:Courier New;">
        <h4>Made with ❤️ by Ashutosh Devpura</h4>
        <p style="color: gray;">Empowering aviation safety with Machine Learning-powered predictions</p>
    </div>
    """,
    unsafe_allow_html=True
)


