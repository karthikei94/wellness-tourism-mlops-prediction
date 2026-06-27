import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

HF_USERNAME = "karthikei94"
MODEL_REPO_ID = f"{HF_USERNAME}/tourism-purchase-model"
MODEL_FILENAME = "best_tourism_purchase_model_v1.joblib"

@st.cache_resource
def load_model():
    model_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename=MODEL_FILENAME)
    return joblib.load(model_path)

model = load_model()

st.title("Wellness Tourism Package Purchase Prediction")
st.write("Enter customer and interaction details to estimate whether the customer is likely to purchase the Wellness Tourism Package.")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    typeof_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    duration_of_pitch = st.number_input("Duration of Pitch", min_value=0.0, max_value=120.0, value=15.0, step=1.0)
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    gender = st.selectbox("Gender", ["Female", "Male"])
    number_of_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=3)
    number_of_followups = st.number_input("Number of Followups", min_value=0, max_value=10, value=3)
    product_pitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])

with col2:
    preferred_property_star = st.selectbox("Preferred Property Star", [3, 4, 5])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
    number_of_trips = st.number_input("Number of Trips", min_value=0, max_value=30, value=2)
    passport = st.selectbox("Passport", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    pitch_satisfaction_score = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
    own_car = st.selectbox("Own Car", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    number_of_children_visiting = st.number_input("Number of Children Visiting", min_value=0, max_value=10, value=1)
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    monthly_income = st.number_input("Monthly Income", min_value=0.0, max_value=200000.0, value=25000.0, step=1000.0)

input_data = pd.DataFrame([{
    "Age": age,
    "TypeofContact": typeof_contact,
    "CityTier": city_tier,
    "DurationOfPitch": duration_of_pitch,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": number_of_person_visiting,
    "NumberOfFollowups": number_of_followups,
    "ProductPitched": product_pitched,
    "PreferredPropertyStar": preferred_property_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": number_of_trips,
    "Passport": passport,
    "PitchSatisfactionScore": pitch_satisfaction_score,
    "OwnCar": own_car,
    "NumberOfChildrenVisiting": number_of_children_visiting,
    "Designation": designation,
    "MonthlyIncome": monthly_income,
}])

if st.button("Predict Purchase"):
    probability = model.predict_proba(input_data)[0, 1]
    prediction = int(probability >= 0.45)
    st.subheader("Prediction Result")
    if prediction == 1:
        st.success(f"Likely to purchase. Estimated probability: {probability:.1%}")
    else:
        st.info(f"Not likely to purchase. Estimated probability: {probability:.1%}")
