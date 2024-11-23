import streamlit as st
import io
import base64
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)
import os
import numpy as np

st.title("Machine Learning Algorithm: Logistic Regression and Linear Regression")
choice = st.selectbox("Select Scenarios", ["Environment", "Health","Predict"])
if choice == "Environment":    
# File uploader for the dataset
    st.header("Environmental Predictive Analysis")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        # Load the dataset
        data = pd.read_csv(uploaded_file)
        
        # Display the dataset preview
        st.subheader("Dataset Preview")
        st.write(data.head())

        # Check if required columns exist
        required_columns = ["TotalWeightedObservations", "GMSL_noGIA", "StdDevGMSL_noGIA", "SmoothedGSML_noGIA",
                            "GMSL_GIA", "StdDevGMSL_GIA", "SmoothedGSML_GIA", "SmoothedGSML_GIA_sigremoved"]
        
        if all(col in data.columns for col in required_columns):
            # Select features and target variable
            X = data[["TotalWeightedObservations", "GMSL_noGIA", "StdDevGMSL_noGIA", "SmoothedGSML_noGIA",
                    "GMSL_GIA", "StdDevGMSL_GIA", "SmoothedGSML_GIA"]]
            y = data["SmoothedGSML_GIA_sigremoved"]
            
            # Dropdown to select method
            option = st.selectbox("Select method to Train and Test", ["Split into Train-Test Sets", "Repeated Random Train-Test Splits"])
            
            if option == "Split into Train-Test Sets":
                # Split into Train-Test Split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123, train_size=0.8)

                # Train the model
                model = LinearRegression()
                model.fit(X_train, y_train)

                # Make predictions
                y_pred = model.predict(X_test)

                # Calculate performance metrics
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                # Display performance metrics
                st.subheader("Model Performance Metrics")
                

                
                # Plot residuals (errors)
                residuals = y_test - y_pred
                st.subheader("  i. Residuals Distribution (Boxplot)")
                fig, ax = plt.subplots()
                sns.boxplot(residuals, ax=ax,color='lime')
                ax.set_title('Residuals Distribution')
                st.pyplot(fig)
                st.markdown(f"""
    <p style="font-size: 16px; color: white;">
        The figure shows the <strong style="color:white">Mean Absolute Error</strong> of the model, which averaged 
        <strong style="color:lime;">{mae:.2f}</strong> between the predictions and actual values.
    </p>
""", unsafe_allow_html=True)

                # Scatter plot of predicted vs actual values
                st.subheader("ii. Predicted vs Actual Values (Scatter Plot)")
                fig, ax = plt.subplots()
                ax.scatter(y_test, y_pred, color='crimson')
                ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='yellowgreen', linestyle='--')  # Ideal line
                ax.set_xlabel('Actual Values')
                ax.set_ylabel('Predicted Values')
                ax.set_title('Predicted vs Actual Values')
                st.pyplot(fig)
                st.markdown(f"""
    <p style="font-size: 16px; color: white;">
        Upon testing the model with a train-test split of 80% and 20%, the <strong style="color:white;">Mean Squared Error (MSE)</strong> is <strong style="color: red;">{mse:.2f}</strong>, indicating the average squared difference between predicted and actual values. 
        This suggests that the model's predictions deviate on average by <strong style="color: red;">{mse:.2f}</strong> squared units from the true values, and further model refinement may be necessary to reduce this error for improved accuracy.
    </p>
""", unsafe_allow_html=True)
 # Plot MSE, MAE, and R² as bar chart
                st.subheader("iii. Comparison of Performance Metrics")
                fig, ax = plt.subplots()
                metrics = ['MSE', 'MAE', 'R²']
                values = [mse, mae, r2]
                ax.bar(metrics, values, color=['crimson', 'lime', '#004F98'])
                ax.set_ylabel('Metric Value')
                ax.set_title('Model Performance Metrics Comparison')
                st.pyplot(fig)
                st.subheader("Conclusion")
                st.markdown(f"""
    <p style="font-size: 16px; color: white;text-align: justify;">
        &nbsp;&nbsp;&nbsp;In conclusion, upon testing the model with a train-test split of 80% and 20%, the <strong>Mean Squared Error (MSE)</strong> is <strong style="color: red;">{mse:.2f}</strong> , indicating the average squared difference between predicted and actual values. 
        The <strong>Mean Absolute Error (MAE)</strong> is <strong style="color: green;">{mae:.2f}</strong> , showing the average absolute difference between the predicted and actual values. 
        Lastly, the <strong>R-squared (R²)</strong> value of <strong style="color: blue;">{r2:.2f}</strong> suggests the model explains <strong style="color: skyblue;">{r2 * 100:.2f}%</strong> of the variance in the target variable, indicating a strong fit. 
        These performance metrics demonstrate that the model is performing well, with a relatively low error and a high level of accuracy in explaining the variance of the target variable.
    </p>
""", unsafe_allow_html=True)

                # Save the model as a .joblib file
                model_filename = "LR_Split2Train.joblib"
                joblib.dump(model, model_filename)

                # Provide download link
                st.subheader("Download the Trained Model")
                st.write(f"Click the button below to download the trained model:")
                st.download_button(label="Download Model", data=open(model_filename, "rb"), file_name="linear_regression_model.joblib")

            elif option == "Repeated Random Train-Test Splits":
                    # Repeated Random Train-Test Splits
                    mse_list = []
                    mae_list = []
                    r2_list = []

                    repeats = st.slider("Select the number of random splits", 5, 100, 10)
                    for _ in range(repeats):
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)

                        # Train the model
                        model = LinearRegression()
                        model.fit(X_train, y_train)

                        # Make predictions
                        y_pred = model.predict(X_test)

                        # Calculate metrics
                        mse_list.append(mean_squared_error(y_test, y_pred))
                        mae_list.append(mean_absolute_error(y_test, y_pred))
                        r2_list.append(r2_score(y_test, y_pred))

                    # Calculate average metrics
                    avg_mse = np.mean(mse_list)
                    avg_mae = np.mean(mae_list)
                    avg_r2 = np.mean(r2_list)

                    # Display average metrics
                    

                    # Plot metrics across splits
                    st.subheader("Performance Metrics Across Splits")
                    fig, ax = plt.subplots()
                    ax.plot(range(repeats), mse_list, label="MSE", marker='o', color='purple')
                    ax.plot(range(repeats), mae_list, label="MAE", marker='o', color='#00308F')
                    ax.plot(range(repeats), r2_list, label="R²", marker='o', color='lime')
                    ax.set_xlabel("Split Number")
                    ax.set_ylabel("Metric Value")
                    ax.set_title("Performance Metrics Across Splits")
                    ax.legend()
                    st.pyplot(fig)
                    st.subheader("Conclusion")
                    st.markdown(f"""
    <p style="font-size: 16px; color: white; text-align: justify;"">
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Upon performing Repeated Random Train-Test Splits for 
        <strong style="color: crimson;">{repeats}</strong> iterations, the average 
        <strong style="color: white;">Mean Squared Error (MSE)</strong> is 
        <strong style="color: purple;">{avg_mse:.2f}</strong>, indicating the average squared 
        difference between predicted and actual values across all splits. The average 
        <strong style="color: white;">Mean Absolute Error (MAE)</strong> is 
        <strong style="color: #00308F;">{avg_mae:.2f}</strong>, representing the average absolute 
        error between the predictions and actual values. Lastly, the average 
        <strong style="color: white;">R-squared (R²)</strong> of 
        <strong style="color: lime;">{avg_r2:.2f}</strong> suggests that, on average, the model explains 
        <strong style="color: crimson;">{avg_r2 * 100:.2f}%</strong> of the variance in the target variable, 
        providing an overall indication of the model's performance across multiple train-test splits.
    </p>
""", unsafe_allow_html=True)

                    # Save the trained model to joblib
                    model_filename = "LR_random_splits.joblib"
                    joblib.dump(model, model_filename)

                    # Download button for the trained model
                    st.subheader("Download Trained Model")
                    st.download_button(label="Download Model", 
                                    data=open(model_filename, "rb"), 
                                    file_name=model_filename, 
                                    mime="application/octet-stream")
        else:
            st.error(f"The dataset must contain the following columns: {', '.join(required_columns)}")
elif choice =="Health":
            
            
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
        # Dropdown for selecting cross-validation method
    cross_val_method = st.selectbox(
            "Select the cross-validation method",
            options=[None, "Leave-One-Out Cross-Validation (LOOCV)", "K-Fold Cross-Validation"],
            index=0  # Default to None, meaning no selection
        )
    loocv_container = st.empty()
    kfold_container = st.empty()
        # Check for missing values
    if cross_val_method is not None:  # Check if a valid selection is made
        if cross_val_method == "Leave-One-Out Cross-Validation (LOOCV)":
            st.title("Leave-One-Out Cross-Validation (LOOCV)")
            st.markdown("""
            <style>
            .zoom-container {
                display: inline-block;
                transition: transform 0.3s ease-in-out, z-index 0.3s ease-in-out;
                position: relative; /* Ensures proper layering */
                margin: 10px;
                border-radius: 10px;
                overflow: hidden;
                z-index: 1; /* Normal z-index */
            }

            .zoom-container:hover {
                transform: scale(4.5); /* Adjust zoom scale as needed */
                z-index: 1000; /* Higher value to overlay everything */
                position: relative; /* Remains positioned properly */
            }

            .zoom-container img {
                width: 100%;
                border-radius: 10px;
            }
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
                    margin-top:20px;
                    position:absolute;
                }

                .column-2 {
                    background-color: #e0f7fa;  /* Light blue background */
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    margin-top:30px;
                        position:relative;
                }

                .column-3 {
                    background-color: #fff3e0;  /* Light orange background */
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    margin-top:40px;
                        position:relative;
                    
                }
                .column-4 {
                    background-color: #fff3e0;  /* Light orange background */
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    margin-top:40px;
                        position:relative; margin-left:2000px;
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
                    .behind{
                    z-index:0;
                    }
            </style>
        """, unsafe_allow_html=True)  
            st.markdown('</div>', unsafe_allow_html=True)  # Close div
            # Create a layout with 3 columns
            col1, col2, col3 = st.columns(3)
            
            st.markdown('<div class="container_body">', unsafe_allow_html=True)  # Add custom class
            
            loocv = LeaveOneOut()
            model = LogisticRegression(max_iter=1000)

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
            with col1:
                st.header(f"Classification Accuracy")
                st.markdown('<div class="column-1">', unsafe_allow_html=True)  # Add custom class
                st.markdown('</div>', unsafe_allow_html=True)  # Close div
                
                plt.figure(figsize=(10, 5))
                plt.boxplot(loocv_accuracies)
                plt.title('Leave-One-Out Cross-Validation Accuracy')
                plt.ylabel('Accuracy (%)')
                
                plt.xticks([1], ['LOO'])  # Label for the x-axis
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png')
                buffer.seek(0)
                base64_img = base64.b64encode(buffer.read()).decode()
                st.markdown(
            f"""
            <div class="zoom-container">
                <img src="data:image/png;base64,{base64_img}" alt="Generated Plot">
            </div>
            """,
            unsafe_allow_html=True
        )        
                st.write(f"Mean Classification Accuracy (LOOCV): {mean_accuracy * 100:.3f}%")
                st.markdown('</div>', unsafe_allow_html=True)  # Close div
                
            with col2:
                st.header(f"Logarithmic Loss")
                st.markdown('<div class="column-2">', unsafe_allow_html=True)  # Add custom class
                
            
                
                # Boxplot for LOOCV Accuracy
            
                # Plot for Logarithmic Loss (LOOCV)
                plt.figure(figsize=(10, 5))
                plt.boxplot(loocv_log_losses)
                plt.title('Leave-One-Out Cross-Validation Logarithmic Loss')
                plt.ylabel('Logarithmic Loss')
                plt.xticks([1], ['LOO'])
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png')
                buffer.seek(0)
                base64_img = base64.b64encode(buffer.read()).decode()
                st.markdown(
            f"""
            <div class="zoom-container">
                <img src="data:image/png;base64,{base64_img}" alt="Generated Plot">
            </div>
            """,
            unsafe_allow_html=True
        )
                st.write(f"Mean Logarithmic Loss (LOOCV): {mean_log_loss:.3f}")
                st.markdown('</div>', unsafe_allow_html=True)  # Close div
            # Fit the model on the entire dataset for further metrics
            model.fit(X, y)

            # Predictions on the whole dataset
            y_all_pred = model.predict(X)
            y_prob = model.predict_proba(X)[:, 1]
            
            with col3:
                
                st.subheader('Area Under ROC Curve')
                st.markdown('<div class="column-3">', unsafe_allow_html=True)  # Add custom class
                test_roc_auc = roc_auc_score(y, y_prob)
                fpr, tpr, thresholds = roc_curve(y, y_prob)

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
                st.markdown('</div>', unsafe_allow_html=True)  # Close div
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png')
                buffer.seek(0)
                base64_img = base64.b64encode(buffer.read()).decode()
                st.markdown(
            f"""
            <div class="zoom-container">
                <img src="data:image/png;base64,{base64_img}" alt="Generated Plot">
            </div>
            """,
            unsafe_allow_html=True
        )
                st.write(f"Area Under ROC Curve: {test_roc_auc:.2f}")  

            col1, col2, col3 = st.columns(3)
            with col1:
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
            y_all_pred = model.predict(X)
            y_prob = model.predict_proba(X)[:, 1]
            with col2:
            # Confusion Matrix  
                
                conf_matrix = confusion_matrix(y, y_all_pred)
                st.subheader('Confusion Matrix')
                fig, ax = plt.subplots()
                sns.heatmap(conf_matrix, annot=True, fmt='d', ax=ax, cmap='Blues')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png')
                buffer.seek(0)
                base64_img = base64.b64encode(buffer.read()).decode()
                st.markdown(
            f"""
            <div class="zoom-container">
                <img src="data:image/png;base64,{base64_img}" alt="Generated Plot">
            </div>
            """,
            unsafe_allow_html=True
        )       
                
            
            # Save the trained model
            model_filename = "LOOCV.joblib"
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
        
            
        
        elif cross_val_method == "K-Fold Cross-Validation":
            st.markdown("""
            <style>
            .zoom-container {
                display: inline-block;
                transition: transform 0.3s ease-in-out, z-index 0.3s ease-in-out;
                position: relative; /* Ensures proper layering */
                margin: 10px;
                border-radius: 10px;
                overflow: hidden;
                z-index: 1; /* Normal z-index */
            }

            .zoom-container:hover {
                transform: scale(4.5); /* Adjust zoom scale as needed */
                z-index: 1000; /* Higher value to overlay everything */
                position: relative; /* Remains positioned properly */
            }

            .zoom-container img {
                width: 100%;
                border-radius: 10px;
            }
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
                    .behind{
                    z-index:0;
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
            st.session_state.k = k

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
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png')
                buffer.seek(0)
                base64_img = base64.b64encode(buffer.read()).decode()
                st.markdown(
            f"""
            <div class="zoom-container">
                <img src="data:image/png;base64,{base64_img}" alt="Generated Plot">
            </div>
            """,
            unsafe_allow_html=True
        )
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
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png')
                buffer.seek(0)
                base64_img = base64.b64encode(buffer.read()).decode()
                st.markdown(
            f"""
            <div class="zoom-container">
                <img src="data:image/png;base64,{base64_img}" alt="Generated Plot">
            </div>
            """,
            unsafe_allow_html=True
        )
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
            
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png')
                buffer.seek(0)
                base64_img = base64.b64encode(buffer.read()).decode()
                st.markdown(
            f"""
            <div class="zoom-container">
                <img src="data:image/png;base64,{base64_img}" alt="Generated Plot">
            </div>
            """,
            unsafe_allow_html=True
        )

                
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
                
                
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png')
                buffer.seek(0)
                base64_img = base64.b64encode(buffer.read()).decode()
                st.markdown(
            f"""
            <div class="zoom-container">
                <img src="data:image/png;base64,{base64_img}" alt="Generated Plot">
            </div>
            """,
            unsafe_allow_html=True
        )

                st.write(f"Area Under ROC Curve: {test_roc_auc:.2f}")

                # Save the plot to a buffer in memory
            
            # Save the trained model
            model_filename = "K_fold.joblib"
            joblib.dump(model, model_filename)

            # Download link for the model
            st.markdown('<div class="behind">', unsafe_allow_html=True)  # Add custom class
            with open(model_filename, "rb") as f:
                model_data = f.read()

            st.download_button(
                label="Download Trained Model",
                data=model_data,
                file_name=model_filename,
                mime="application/octet-stream"
            )
            st.markdown('</div>', unsafe_allow_html=True)  # Close div
            # Clean up the model file if needed (optional)
            os.remove(model_filename)
            st.markdown('</div>', unsafe_allow_html=True)  # Close div

elif choice == "Predict":
        
    # Step 1: Upload the pre-trained model
    uploaded_model = st.file_uploader("Upload your trained .joblib model", type=["joblib"])
    option2 = st.selectbox("Select what to Predict", ["Sea Rise Level", "Heart Failure"])
    if option2=="Sea Rise Level":
        if uploaded_model is not None:
            # Load the trained model
            model = joblib.load(uploaded_model)
            st.success("Model loaded successfully!")

            # Step 2: Input data for prediction
            st.subheader("Input Data for Prediction")

            # Input fields for each feature
            year = st.number_input("Year", value=2023, step=1, format="%d")
            TotalWeightedObservations = st.number_input("Total Weighted Observations", value=0.0, format="%.2f")
            GMSL_noGIA = st.number_input("GMSL (Global Isostatic Adjustment (GIA) not applied) variation (mm) with respect to 20-year TOPEX/Jason collinear mean reference:", value=0.0, format="%.2f")
            StdDevGMSL_noGIA = st.number_input("Standard Deviation of GMSL (GIA not applied) variation estimate (mm)", value=0.0, format="%.2f")
            SmoothedGSML_noGIA = st.number_input("Smoothed (60-day Gaussian type filter) GMSL (GIA not applied) variation (mm)", value=0.0, format="%.2f")
            GMSL_GIA = st.number_input("GMSL (Global Isostatic Adjustment (GIA) applied) variation (mm) with respect to 20-year TOPEX/Jason collinear mean reference:", value=0.0, format="%.2f")
            StdDevGMSL_GIA = st.number_input("Standard Deviation of GMSL (GIA applied) variation estimate (mm)", value=0.0, format="%.2f")
            SmoothedGSML_GIA = st.number_input("Smoothed (60-day Gaussian type filter) GMSL (GIA applied) variation (mm)", value=0.0, format="%.2f")

            # Organize inputs into a single feature array
            input_features = np.array([[TotalWeightedObservations, GMSL_noGIA, StdDevGMSL_noGIA,
                                        SmoothedGSML_noGIA, GMSL_GIA, StdDevGMSL_GIA, SmoothedGSML_GIA]])

            # Step 3: Make prediction
            if st.button("Predict"):
                prediction = model.predict(input_features)[0]  # Get the prediction
                st.subheader("Prediction Result")
                st.write(f"The Predicted Sea Rise Level by  {int(year):d} is  {prediction:.2f} in millimeters")

        else:
            st.warning("Please upload a trained .joblib model to proceed.")
    elif option2=="Heart Failure":
            
        if uploaded_model is not None:
            # Load the trained model
            model = joblib.load(uploaded_model)

            # Input form for user data
            age = st.number_input("Age", min_value=0, max_value=120, value=30)
            anaemia = st.selectbox("Anaemia (0: No, 1: Yes)", [0, 1])
            creatinine_phosphokinase = st.number_input("Creatinine Phosphokinase Level", min_value=0)
            diabetes = st.selectbox("Diabetes (0: No, 1: Yes)", [0, 1])
            ejection_fraction = st.number_input("Ejection Fraction (%)", min_value=0, max_value=100)
            high_blood_pressure = st.selectbox("High Blood Pressure (0: No, 1: Yes)", [0, 1])
            platelets = st.number_input("Platelets Count", min_value=0)
            serum_creatinine = st.number_input("Serum Creatinine", min_value=0.0)
            serum_sodium = st.number_input("Serum Sodium", min_value=0.0)
            sex = st.selectbox("Sex (0: Female, 1: Male)", [0, 1])
            smoking = st.selectbox("Smoking (0: No, 1: Yes)", [0, 1])
            time = st.number_input("Time", min_value=0)  # Ensure you include this if it's a feature

            # Create a button to predict
            if st.button("Predict"):
                # Prepare the input data as a DataFrame
                input_data = pd.DataFrame({
                    "age": [age],
                    "anaemia": [anaemia],
                    "creatinine_phosphokinase": [creatinine_phosphokinase],
                    "diabetes": [diabetes],
                    "ejection_fraction": [ejection_fraction],
                    "high_blood_pressure": [high_blood_pressure],
                    "platelets": [platelets],
                    "serum_creatinine": [serum_creatinine],
                    "serum_sodium": [serum_sodium],
                    "sex": [sex],
                    "smoking": [smoking],
                    "time": [time]  # Include 'time' if it's used in the model
                })

                # Make predictions
                prediction = model.predict(input_data)
                predicted_probabilities = model.predict_proba(input_data)

                # Display results
                if prediction[0] == 1:
                    st.write("Prediction: The patient is likely to have heart failure.")
                else:
                    st.write("Prediction: The patient is unlikely to have heart failure.")

                st.write("Probability of Heart Failure:", predicted_probabilities)

        else:
            st.warning("Please upload your trained model to make predictions.")

                    
                