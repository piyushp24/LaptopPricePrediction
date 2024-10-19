import streamlit as st
import pandas as pd
import joblib

# Load the trained models and scaler
models = joblib.load('models.pkl')  # Load all models
scaler = joblib.load('scaler.pkl')   # Load the StandardScaler

# Function to make predictions
def predict_price(input_data):
    # Scale the input data
    scaled_data = scaler.transform(input_data)
    
    # Make predictions using each model
    rf_prediction = models['Random Forest'].predict(scaled_data)
    dtree_prediction = models['Decision Tree'].predict(scaled_data)
    knn_prediction = models['KNN'].predict(scaled_data)

    # Return the predictions in a dictionary
    return {
        'Random Forest': rf_prediction[0],
        # 'Decision Tree': dtree_prediction[0],
        # 'KNN': knn_prediction[0]
    }

# Streamlit application interface
st.title("Laptop Price Prediction")
st.write("Enter the specifications of the laptop:")

# Create input fields for the user
ram = st.number_input("RAM (in GB)", min_value=2, max_value=64, step=1)
weight = st.number_input("Weight (in kg)", min_value=0.1, max_value=5.0, step=0.1)
width = st.number_input("Width (in cm)", min_value=10, max_value=200, step=1)
height = st.number_input("Height (in cm)", min_value=10, max_value=200, step=1)
cpu_freq = st.number_input("CPU Frequency (in GHz)", min_value=0.1, max_value=5.0, step=0.1)

# Input for laptop brand selection
brands = ['Acer', 'Apple', 'Asus', 'Chuwi', 'Dell', 'Fujitsu', 
          'Google', 'HP', 'Huawei', 'LG', 'Lenovo', 'MSI', 
          'Mediacom', 'Microsoft', 'Razer', 'Samsung', 'Toshiba', 
          'Vero', 'Xiaomi']
selected_brand = st.selectbox("Select Brand", brands)

# Input for binary features
gaming = st.selectbox("Gaming", [0, 1])
ultrabook = st.selectbox("Ultrabook", [0, 1])
workstation = st.selectbox("Workstation", [0, 1])
notebook = st.selectbox("Notebook", [0, 1])

# Assuming you have binary features for operating systems and CPU/GPU
memory_type = st.selectbox("Memory Type (0 for DDR, 1 for DDR2, etc.)", range(3))  # Adjust range accordingly
amd_cpu = st.selectbox("AMD CPU", [0, 1])
intel_cpu = st.selectbox("Intel CPU", [0, 1])
amd_gpu = st.selectbox("AMD GPU", [0, 1])
nvidia_gpu = st.selectbox("Nvidia GPU", [0, 1])

# Create a DataFrame for the input data
input_data = pd.DataFrame({
    'Windows 7': [0],  # Initialize as 0, to be set later based on conditions
    'Linux': [0],
    'No OS': [0],
    'MSI': [1 if selected_brand == 'MSI' else 0],
    'AMD_CPU': [amd_cpu],
    'Intel_CPU': [intel_cpu],
    'Intel_GPU': [0],  # Assuming you handle Intel GPU selection separately
    'AMD_GPU': [amd_gpu],
    'Acer': [1 if selected_brand == 'Acer' else 0],
    'Weight': [weight],
    'Razer': [1 if selected_brand == 'Razer' else 0],
    'Workstation': [workstation],
    'Ultrabook': [ultrabook],
    'Nvidia_GPU': [nvidia_gpu],
    'Gaming': [gaming],
    'CPU Frequency': [cpu_freq],
    'Notebook': [notebook],
    'height': [height],
    'width': [width],
    'Ram': [ram],
})

# When the user clicks the button, make predictions
if st.button("Predict Price"):
    predictions = predict_price(input_data)
    st.write("Predicted Prices:")
    st.write(f"Random Forest: €{predictions['Random Forest']:.2f}")
    # st.write(f"Decision Tree: €{predictions['Decision Tree']:.2f}")
    # st.write(f"KNN: €{predictions['KNN']:.2f}")
