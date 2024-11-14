import streamlit as st
import pandas as pd
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Streamlit app title
st.title('Logistic Regression with Leave-One-Out Cross-Validation')

# File uploader
uploaded_file = st.file_uploader("Upload your dataset (CSV file)", type=["csv"])

if uploaded_file is not None:
    # Load dataset
    dataframe = pd.read_csv(uploaded_file)

    # Display the dataset
    st.write("Dataset Preview:")
    st.dataframe(dataframe)

    # Split features and target variable
    X = dataframe.drop('DEATH_EVENT', axis=1)  # Features
    y = dataframe['DEATH_EVENT']  # Target

    # Check for missing values
    
if st.button("Run Leave-One-Out Cross-Validation (LOOCV) "):
        st.subheader("Leave-One-Out Cross-Validation (LOOCV)")
        loocv = LeaveOneOut()
        model = LogisticRegression(max_iter=100000)

        # Initialize lists to store results for each iteration
        loocv_accuracies = []
        loocv_predictions = []
        loocv_probs = []
        loocv_log_losses = []

        # Perform LOOCV manually and calculate accuracy and log loss
        for train_index, test_index in loocv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # Fit the model
            model.fit(X_train, y_train)

            # Predict and store accuracy
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            loocv_accuracies.append(accuracy_score(y_test, y_pred))
            
            # Calculate log loss and handle single class in y_test
            loocv_log_losses.append(log_loss(y_test, [y_prob], labels=[0, 1]))

            # Store predictions and probabilities for later metrics
            loocv_predictions.extend(y_pred)
            loocv_probs.extend(y_prob)

        # Mean classification accuracy and log loss
        mean_accuracy = np.mean(loocv_accuracies)
        mean_log_loss = np.mean(loocv_log_losses)
        st.write(f"Mean Classification Accuracy (LOOCV): {mean_accuracy * 100:.3f}%")
        plt.figure(figsize=(10, 5))
        plt.boxplot(loocv_accuracies)
        plt.title('Leave-One-Out Cross-Validation Accuracy')
        plt.ylabel('Accuracy (%)')
        plt.xticks([1], ['LOO'])  # Label for the x-axis
        st.pyplot(plt)

        st.write(f"Mean Logarithmic Loss (LOOCV): {mean_log_loss:.3f}")
        
        # Boxplot for LOOCV Accuracy
    
        # Plot for Logarithmic Loss (LOOCV)
        plt.figure(figsize=(10, 5))
        plt.boxplot(loocv_log_losses)
        plt.title('Leave-One-Out Cross-Validation Logarithmic Loss')
        plt.ylabel('Logarithmic Loss')
        plt.xticks([1], ['LOO'])
        st.pyplot(plt)

        # Fit the model on the entire dataset for further metrics
        model.fit(X, y)

        # Predictions on the whole dataset
        y_all_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]

        # Confusion Matrix
        conf_matrix = confusion_matrix(y, y_all_pred)
        st.subheader('Confusion Matrix')
        fig, ax = plt.subplots()
        sns.heatmap(conf_matrix, annot=True, fmt='d', ax=ax, cmap='Blues')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)

        # Classification Report
        report_dict = classification_report(y, y_all_pred, target_names=["0.0", "1.0"], output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        st.subheader('Classification Report')
        st.dataframe(report_df.style.format({
            'precision': '{:.2f}', 
            'recall': '{:.2f}', 
            'f1-score': '{:.2f}', 
            'support': '{:.0f}'
        }))

        # ROC Curve and AUC
        test_roc_auc = roc_auc_score(y, y_prob)
        fpr, tpr, thresholds = roc_curve(y, y_prob)

        st.subheader('Area Under ROC Curve')
        st.write(f"Area Under ROC Curve: {test_roc_auc:.2f}")

        # Plotting the ROC curve
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, color='blue', label='ROC Curve (area = %0.2f)' % test_roc_auc)
        plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal line
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.grid()
        st.pyplot(plt)  # Display the plot in Streamlit

        # Save the trained model
        model_filename = "heart_LOOCV.joblib"
        joblib.dump(model, model_filename)

        # Download link for the model
        with open(model_filename, "rb") as f:
            model_data = f.read()

        st.download_button(
            label="Download Trained Model",
            data=model_data,
            file_name=model_filename,
            mime="application/octet-stream"
        )

        # Clean up the model file if needed (optional)
        os.remove(model_filename)
if st.button("Run K-Fold Cross-Validation"):
    st.markdown("""
        <style>
            /* Set custom width for columns */
            .stColumn {
                width: 100%;  /* Default width for each column */
            }
            .container_body{
                
                }
            .column-1, .column-2, .column-3 {
                padding: 0px;
                border-radius: 10px;
                background-color: #f2f2f2;  /* Light grey background */
            }

            .column-1 {
                background-color: #f2f2f2;  /* Light grey background */
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                margin-top:-5000px;
                position:absolute;
            }

            .column-2 {
                background-color: #e0f7fa;  /* Light blue background */
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                 margin-top:-100px;
            }

            .column-3 {
                background-color: #fff3e0;  /* Light orange background */
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                
            }

            /* Customize dataframe styling */
            .dataframe th {
                background-color: #006064; /* Dark blue background for headers */
                color: white;
            }
            .div_1{
                margin-top:30px;
                margin-left:200px;
                }
                .div_2{
                margin-top:30px;
                margin-left:200px;
                }
            .grid-container {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;  /* Creates 3 equal columns */
            gap: 20px;  /* Gap between columns */
            grid-template-rows: auto auto;  /* 2 rows of auto height */
        }
        .column {
            padding: 20px;
            border: 1px solid #ccc;
            border-radius: 8px;
            background-color: #f9f9f9;
        }
        </style>
    """, unsafe_allow_html=True)  
    st.markdown('<div class="container_body">', unsafe_allow_html=True)  # Add custom class
 # Split dataset into 80:20 ratio for train-test split
    st.title('II. Performance Metrics')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=None)

    # Initialize model
    model = LogisticRegression(max_iter=1000)

    # Train the model on the training set
    model.fit(X_train, y_train)

    # Predict on the test set
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)

    # Classification Accuracy (80:20 split)
    test_accuracy = accuracy_score(y_test, y_test_pred) * 100  # Convert to percentage

    # User input for number of folds
    k = st.number_input("Select number of folds for K-Fold Cross-Validation:", min_value=2, max_value=10, value=8)

    # K-Fold Cross-Validation
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    accuracy_scores = []
    log_loss_scores = []
    roc_auc_scores = []

    for train_index, test_index in kf.split(X):
        X_train_kf, X_test_kf = X.iloc[train_index], X.iloc[test_index]
        y_train_kf, y_test_kf = y.iloc[train_index], y.iloc[test_index]
        
        model.fit(X_train_kf, y_train_kf)
        y_pred_kf = model.predict(X_test_kf)
        y_pred_proba_kf = model.predict_proba(X_test_kf)

        # Collect accuracy scores for the boxplot
        accuracy = accuracy_score(y_test_kf, y_pred_kf) * 100  # Convert to percentage
        accuracy_scores.append(accuracy)

        log_loss_scores.append(log_loss(y_test_kf, y_pred_proba_kf))
        roc_auc_scores.append(roc_auc_score(y_test_kf, y_pred_proba_kf[:, 1]))

    # Classification Accuracy (K-Fold)



    # Boxplot for K-Fold Cross-Validation Accuracy
    plt.figure(figsize=(10, 5))
    plt.boxplot(accuracy_scores)
     

        #
    # Create a layout with 3 columns
    col1, col2, col3 = st.columns(3)

    # First row with 3 boxes (Performance Metrics)
    with col1:
        st.markdown('<div class="column-1">', unsafe_allow_html=True)  # Add custom class
       
        st.subheader(f"1. Classification Accuracy: Split to 80:20 ratio")
        
        plt.title('K-Fold Cross-Validation Accuracy')
        plt.ylabel('Accuracy (%)')
        plt.xticks([1], [f'{k}-Fold'])  # Label for the x-axis
        st.pyplot(plt)
        st.write(f"Classification Accuracy: {np.mean(accuracy_scores):.2f}%")  # Format as percentage
        st.write(f"STANDARD DEVIATION: {np.std(accuracy_scores):.2f}%")  # Format as percentage
        st.markdown('</div>', unsafe_allow_html=True)  # Close div
    with col2:
        st.markdown('<div class="column-2">', unsafe_allow_html=True)  # Add custom class
        st.subheader(f"2. Logarithmic Loss (K-Fold)")
        
        # Log Loss Visualization
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(range(1, k + 1), log_loss_scores, marker='o', label='Log Loss', color='blue')
        ax.set_xlabel('Fold')
        ax.set_ylabel('Log Loss')
        ax.set_title('Log Loss for Each Fold')
        ax.grid()
        st.markdown('<div class="div_1">', unsafe_allow_html=True)  # Add custom class
        st.pyplot(fig)
        st.write(f" Logarithmic Loss (K-Fold): {np.mean(log_loss_scores):.2f}")  # Format as percentage
        st.markdown('</div>', unsafe_allow_html=True)  # Close div
        
    with col3:
        st.markdown('<div class="column-3">', unsafe_allow_html=True)  # Add custom class
        st.subheader('3. Confusion Matrix')
        y_all_pred = model.predict(X)
        conf_matrix = confusion_matrix(y, y_all_pred)
        fig, ax = plt.subplots()
        sns.heatmap(conf_matrix, annot=True, fmt='d', ax=ax, cmap='Blues')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.markdown('<div class="div_2">', unsafe_allow_html=True)  # Add custom class
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)  # Close div
        st.markdown('</div>', unsafe_allow_html=True)  # Close div
        

    # Second row with 3 boxes (Classification Report and ROC Curve)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader('4. Classification Report')
        report_dict = classification_report(y, y_all_pred, target_names=["0.0", "1.0"], output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        st.dataframe(report_df.style.format({
            'precision': '{:.2f}', 
            'recall': '{:.2f}', 
            'f1-score': '{:.2f}', 
            'support': '{:.0f}'
        }))

    with col2:
        st.subheader('5. Area Under ROC Curve')

        # Area Under ROC Curve (80:20 split)
        test_roc_auc = roc_auc_score(y_test, y_test_proba[:, 1])

        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_test_proba[:, 1])

        # Display ROC AUC score
        

        # Plotting the ROC curve
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, color='blue', label='ROC Curve (area = %0.2f)' % test_roc_auc)
        plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal line
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.grid()
        st.pyplot(plt)  # Display the plot in Streamlit
        st.write(f"Area Under ROC Curve: {test_roc_auc:.2f}")
    # Save the trained model
    model_filename = "heart_K_fold.joblib"
    joblib.dump(model, model_filename)

    # Download link for the model
    with open(model_filename, "rb") as f:
        model_data = f.read()

    st.download_button(
        label="Download Trained Model",
        data=model_data,
        file_name=model_filename,
        mime="application/octet-stream"
    )

    # Clean up the model file if needed (optional)
    os.remove(model_filename)
    st.markdown('</div>', unsafe_allow_html=True)  # Close div
   