import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load Data from Excel File
file_path = "C:\\Users\\kirti\\OneDrive\\Desktop\\finalcrops.xlsx"
df = pd.read_excel(file_path)

# Preprocessing: One-hot encoding for categorical variables and drop 'Weather' column
df_processed = pd.get_dummies(df.drop(columns=['Weather']))
df_processed = df_processed.dropna()  # Drop rows with missing values

# Check if 'Crop' column is present in the original DataFrame
if 'Crop' in df.columns:
    # Ensure 'Crop' column is retained in the processed DataFrame
    if 'Crop' not in df_processed.columns:
        df_processed['Crop'] = df['Crop']

    # User Interface
    st.title('Farm Management and Prediction System')
    st.sidebar.title('Options')

    # Select Soil Type
    soil_type = st.sidebar.selectbox('Select Soil Type', df['Soil_type'].unique())

    # Filter data for selected soil type
    filtered_data = df[df['Soil_type'] == soil_type]

    # Display crop information including additional details
    st.write('Crop Information:')
    st.write(filtered_data[['Crop', 'Temperature(°C)', 'Rainfall(mm)', 'Soil_pH', 
                            'Nitrogen', 'Phosphorus', 'Potassium', 'Crop_Disease',
                            'Planting_Density', 'Harvesting_Method',
                            'Actual_Yield', 'Predicted_Yield', 'Soil_type']])

    # Plot crop trend
    st.write('Crop Trend:')
    fig, ax = plt.subplots()
    sns.lineplot(data=filtered_data, x='Crop', y='Actual_Yield', label='Actual Yield', ax=ax)
    plt.xticks(rotation=45)
    plt.xlabel('Crop')
    plt.ylabel('Yield')
    plt.title('Crop Trend')
    st.pyplot(fig)

    # Add more interactive elements to the UI
    st.sidebar.header('Advanced Options')

    # Feature selection
    selected_features = st.sidebar.multiselect('Select Features to Visualize:',
                                               ['Temperature(°C)', 'Rainfall(mm)', 'Soil_pH', 
                                                'Nitrogen', 'Phosphorus', 'Potassium'])

    # Visualize selected features
    if selected_features:
        st.write('Selected Features:')
        st.write(filtered_data[selected_features])

    # Explore data distribution
    st.write('Data Distribution:')
    for feature in selected_features:
        fig, ax = plt.subplots()
        sns.histplot(data=filtered_data, x=feature, kde=True, bins=20)
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.title(f'{feature} Distribution')
        st.pyplot(fig)

    # Model Training and Evaluation
    if st.sidebar.button('Train Model'):
        def train_model():
            # Split the data into features (X) and target (y)
            X = df_processed.drop(columns=['Actual_Yield', 'Crop'])  # Exclude 'Crop' from features
            y = df_processed['Actual_Yield']

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Model Training with pipeline for scaling
            pipeline = Pipeline([('scaler', StandardScaler()), ('model', RandomForestRegressor())])

            # Hyperparameter tuning using GridSearchCV
            param_grid = {'model__n_estimators': [50, 100, 200], 'model__max_depth': [None, 10, 20]}
            grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1)  # Adjusted cv value
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_

            return model, X_test, y_test

        # Call the cached function
        model, X_test, y_test = train_model()

        # Make predictions
        predictions = model.predict(X_test)

        
        # Evaluation
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        st.write(f'Mean Squared Error: {mse}')
        st.write(f'R^2 Score: {r2}')

        # Plot actual vs. predicted yield for all data
        st.write('Actual vs Predicted Yield:')
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df_processed, x='Actual_Yield', y='Predicted_Yield')

        # Annotate each point with crop name
        for i in range(len(df_processed)):
            ax.text(df_processed['Actual_Yield'][i], df_processed['Predicted_Yield'][i], df_processed['Crop'][i])

        plt.xlabel('Actual Yield')
        plt.ylabel('Predicted Yield')
        plt.title('Actual vs Predicted Yield')
        st.pyplot(fig)

else:
    st.write("Error: 'Crop' column not found in the original DataFrame.")
