import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

st.title("🔧 Predictive Maintenance System")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.subheader("📊 Dataset Preview")
    st.write(data.head())

    # Convert text to numbers
    data = pd.get_dummies(data)

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Split data (for accuracy)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Accuracy
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    st.subheader("📈 Model Performance")
    st.write(f"Accuracy: {acc:.2f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    st.write("Confusion Matrix:")
    st.write(cm)

    st.subheader("🧠 Enter Machine Data")

    temp = st.slider("Temperature", 0, 120, 60)
    vib = st.slider("Vibration", 0, 100, 10)
    pres = st.slider("Pressure", 0, 100, 30)

    if st.button("Predict"):

        input_data = pd.DataFrame([[temp, vib, pres]],
                                  columns=["Temperature", "Vibration", "Pressure"])

        input_data = pd.get_dummies(input_data)
        input_data = input_data.reindex(columns=X.columns, fill_value=0)

        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)

        st.subheader("🔍 Prediction Result")

        if prediction[0] == 1:
            st.error("⚠️ Machine likely to FAIL")
        else:
            st.success("✅ Machine is SAFE")

        st.write(f"Failure Probability: {probability[0][1]*100:.2f}%")

else:
    st.warning("Please upload a dataset first")