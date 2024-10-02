import streamlit as st

st.title("🎈 My new app")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)
@ -0,0 +1,109 @@
# streamlit_thermoelectric_ml.py

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Function to generate features from the dataset
def generate_features(df):
    features = pd.DataFrame()
    features['Max_Seebeck'] = df['Seebeck_Coefficient'].max()
    features['Min_Seebeck'] = df['Seebeck_Coefficient'].min()
    features['Mean_Seebeck'] = df['Seebeck_Coefficient'].mean()
    
    features['Max_Electrical'] = df['Electrical_Conductivity'].max()
    features['Min_Electrical'] = df['Electrical_Conductivity'].min()
    features['Mean_Electrical'] = df['Electrical_Conductivity'].mean()
    
    features['Max_Thermal'] = df['Thermal_Conductivity'].max()
    features['Min_Thermal'] = df['Thermal_Conductivity'].min()
    features['Mean_Thermal'] = df['Thermal_Conductivity'].mean()
    
    return features

# Load dataset from file upload
def load_data(file):
    df = pd.read_csv(file)
    return df

# Streamlit app definition
def main():
    st.title("Sustainable Thermoelectric Materials Prediction")

    st.write("""
    Upload a CSV file containing the following columns:
    - Material
    - Seebeck_Coefficient
    - Electrical_Conductivity
    - Thermal_Conductivity
    - Toxic (0 for non-toxic, 1 for toxic materials)
    """)

    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Load the dataset
        df = load_data(uploaded_file)
        st.write("### Dataset Preview")
        st.write(df.head())
        
        # Display the features
        st.write("### Feature Engineering")
        features_df = generate_features(df)
        st.write(features_df)

        # Set thresholds as defined in the paper
        thresholds = {
            'Seebeck': 100,
            'Electrical': 100,
            'Thermal': 10
        }

        # Create labels for classification
        df['Label'] = ((df['Seebeck_Coefficient'] > thresholds['Seebeck']) &
                        (df['Electrical_Conductivity'] > thresholds['Electrical']) &
                        (df['Thermal_Conductivity'] < thresholds['Thermal'])).astype(int)

        st.write("### Dataset with Labels")
        st.write(df)

        # Model Development
        st.write("### Model Training and Evaluation")

        X = df[['Seebeck_Coefficient', 'Electrical_Conductivity', 'Thermal_Conductivity']]
        y = df['Label']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        model.fit(X_train, y_train)

        # Predictions and Evaluation
        y_pred = model.predict(X_test)
        st.write("### Model Performance")
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y_test, y_pred))
        
        st.write("Classification Report:")
        st.text(classification_report(y_test, y_pred))

        # Make predictions on the full dataset
        st.write("### Predictions on New Data")
        df['Prediction'] = model.predict(X)
        st.write(df[['Material', 'Seebeck_Coefficient', 'Electrical_Conductivity', 'Thermal_Conductivity', 'Prediction']])

        # Plot Results
        st.write("### Data Distribution")
        fig, ax = plt.subplots()
        df['Seebeck_Coefficient'].hist(ax=ax, bins=10)
        ax.set_title("Seebeck Coefficient Distribution")
        st.pyplot(fig)

# Streamlit command to run the app
if __name__ == '__main__':
    main()