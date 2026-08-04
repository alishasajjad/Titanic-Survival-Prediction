# Titanic Survival Prediction App
import streamlit as st
import pandas as pd
import joblib

# Load Saved Model & Scaler

model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Page Configuration
st.set_page_config(page_title="Titanic Survival Prediction",page_icon="🚢", layout="wide")

st.markdown("""
<style>

.stButton>button{
    width:100%;
    border-radius:12px;
    height:55px;
    font-size:18px;
    font-weight:bold;
}

div[data-testid="stMetric"]{
    border-radius:12px;
    border:1px solid #E5E7EB;
    padding:15px;
}

</style>
""", unsafe_allow_html=True)

# Title

st.title("🚢 Titanic Survival Prediction System")

st.markdown("""Predict passenger survival using an XGBoost Machine Learning model trained on the Titanic dataset.""")

st.divider()

# Passenger Information

st.subheader("Passenger Information")

col1, col2 = st.columns(2)

with col1:

    pclass = st.selectbox("Passenger Class",[1, 2, 3])

    sex_text = st.selectbox("Sex",["Male", "Female"])
    if sex_text == "Male":
        sex = 1
    else:
        sex = 0

    age = st.slider("Age",min_value=0,max_value=80,value=25)

    fare = st.number_input("Fare",min_value=0.0,value=50.0)

with col2:

    sibsp = st.number_input("Siblings / Spouses",min_value=0,max_value=10,value=0)

    parch = st.number_input("Parents / Children",min_value=0,max_value=10,value=0)

    embarked = st.selectbox("Embarked Port",["S", "C", "Q"])

    title = st.selectbox(
        "Passenger Title",
        ["Mr","Mrs","Miss","Master","Dr","Rev","Col","Major","Capt","Sir","Lady","Don","Jonkheer","Mme","Mlle","Ms","the Countess"]
    )

st.divider()

# Derived Features

# Family Size
family_size = sibsp + parch + 1
# Is Alone
is_alone = 1 if family_size == 1 else 0

# Age Group
if age <= 12:
    Age_Group = 1
elif age <= 19:
    Age_Group = 3
elif age <= 35:
    Age_Group = 4
elif age <= 60:
    Age_Group = 0
else:
    Age_Group = 2

# Encode Embarked

embarked_mapping = {
    "S": 0,
    "C": 1,
    "Q": 2
}

embarked_text = embarked      
embarked = embarked_mapping[embarked]

# Create Input DataFrame

input_df = pd.DataFrame({
    "Pclass": [pclass],
    "Sex": [sex],
    "Age": [age],
    "SibSp": [sibsp],
    "Parch": [parch],
    "Fare": [fare],
    "Embarked": [embarked],
    "FamilySize": [family_size],
    "IsAlone": [is_alone],
    "Title": [title],
    "AgeGroup": [Age_Group]
})

title_mapping = {"Mr":0,"Miss":1,"Mrs":2,"Master":3,"Dr":4,"Rev":5,"Col":6,"Major":7,"Capt":8,"Sir":9,"Lady":10,"Don":11,"Jonkheer":12,"Mme":13,"Mlle":14,"Ms":15,"the Countess":16}

input_df["Title"] = input_df["Title"].map(title_mapping)

# Scale Numerical Features

input_df[["Age", "Fare"]] = scaler.transform(
    input_df[["Age", "Fare"]]
)

# Prediction

if st.button("🚀 Predict Survival", use_container_width=True):

    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0]

    survive_prob = probability[1] * 100
    death_prob = probability[0] * 100

    st.divider()

    st.subheader("🎯 Prediction Result")

    if prediction == 1:

        st.success("🎉 Passenger is likely to SURVIVE")

    else:

        st.error("❌ Passenger is NOT likely to survive")

    # Probability

    st.subheader("📊 Prediction Probability")

    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Survival Probability", f"{survive_prob:.2f}%")
        st.progress(float(survive_prob) / 100)
    
    with col2:
        st.metric("Death Probability", f"{death_prob:.2f}%")
        st.progress(float(death_prob) / 100)

    st.divider()

    # Passenger Summary

    st.subheader("Passenger Summary")
    
    summary = pd.DataFrame({
    "Feature": ["Passenger Class","Sex","Age","Fare","Family Size","Embarked"],
    "Value": [str(pclass),sex_text,str(age),str(fare),str(family_size),embarked_text]
})

    st.dataframe(summary, use_container_width=True, hide_index=True)
    
# Sidebar

st.sidebar.title("🚢 Titanic Predictor")

st.sidebar.markdown("### 📌 Model Information")

st.sidebar.metric("Accuracy", "82.12%")

st.sidebar.info("""
**Model:** XGBoost

**Dataset:** Titanic

""")

st.sidebar.divider()

st.sidebar.markdown("### 💻 Developer")

st.sidebar.write("**Alisha Sajjad**")
st.sidebar.caption("Machine Learning Project")

st.sidebar.divider()

st.sidebar.caption("Powered by Streamlit")

# Footer

st.divider()

st.markdown("""
<div style='text-align:center;color:gray;font-size:15px;padding:10px;'>

🚢 Titanic Survival Prediction System<br>
Powered by XGBoost • Built with ❤️ using Streamlit

</div>
""", unsafe_allow_html=True)


