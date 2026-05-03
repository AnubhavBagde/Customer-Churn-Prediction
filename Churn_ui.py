import streamlit as st
import joblib
import pandas as pd

st.title("📊 Customer Churn Prediction")
st.write("This project leverages predictive analytics to address one of the most critical"
         " challenges in subscription-based services: Customer Churn. "
         "By analyzing historical user data and engagement patterns, "
         "the application identifies key indicators that lead to subscription cancellations. "
         "The primary goal is to empower businesses with actionable insights, "
         "moving from reactive responses to proactive retention strategies. "
         "Through this machine learning approach, companies can anticipate at-risk customers "
         "and implement targeted interventions "
         "to improve long-term loyalty and stabilize recurring revenue.")

col1, col2, col3 = st.columns(3)

# -------------------------------
# Column 1 (User Info)
# -------------------------------
with col1:
    st.subheader("👤 Account Info")

    account_age = st.slider("Account Age (months)", 1, 60, 12)
    monthly_charges = st.number_input("Monthly Charges", 0.0, 500.0, 50.0)
    support_tickets = st.slider("Support Tickets", 0, 10, 1)
# -------------------------------
# Column 2 (Usage Info)
# -------------------------------
with col2:
    st.subheader("📺 Usage Behavior")

    viewing_hours = st.slider("Viewing Hours per Week", 0, 50, 10)
    downloads = st.slider("Downloads per Month", 0, 50, 5)
    avg_duration = st.slider("Avg Viewing Duration", 0, 300, 60)

# -------------------------------
# Column 3 (Other Factors)
# -------------------------------
with col3:
    st.subheader("⚙️ Preferences")



    subscription_type = st.selectbox(
        "Subscription Type",
        ["Basic", "Standard", "Premium"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Credit card"]
    )

    genre = st.selectbox(
        "Genre Preference",
        ["Comedy", "Sci-Fi", "Drama", "Fantasy"]
    )

    user_rating = st.slider("User Rating", 1.0, 5.0, 3.0)
if st.button("🚀 Predict Churn"):

    input_data = pd.DataFrame([{
        "AccountAge": account_age,
        "MonthlyCharges": monthly_charges,
        "ViewingHoursPerWeek": viewing_hours,
        "ContentDownloadsPerMonth": downloads,
        "AverageViewingDuration": avg_duration,
        "SupportTicketsPerMonth": support_tickets,
        "SubscriptionType": subscription_type,
        "PaymentMethod": payment_method,
        "GenrePreference": genre,
        "UserRating": user_rating
    }])
    # Load pipeline
    pipeline = joblib.load("churn_model.pkl")
    columns = joblib.load("columns_model.pkl")
    threshold = 0.41667178

    # Encode
    input_data = pd.get_dummies(input_data)
    input_data = input_data.reindex(columns=columns, fill_value=0)

    # Predict
    prob = pipeline.predict_proba(input_data)[:, 1][0]
    prediction = int(prob > threshold)

    # Output
    st.subheader("📈 Result")

    if prediction == 1:
        st.error(f"⚠️ Likely to churn (Churn Prediction: {prob*100:.2f})")
    else:
        st.success(f"✅ Likely to stay (Churn Prediction: {prob*100:.2f})")
