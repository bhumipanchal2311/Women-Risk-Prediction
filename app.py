import streamlit as st
import joblib
import pandas as pd

df=pd.read_csv("C:\\Users\\Bhumi\\OneDrive\\Desktop\\project\\Woman_Safety_Dataset_Management.csv")

# ---------------------------------------------------------
# LOAD MODEL AND SCALER
# ---------------------------------------------------------

model = joblib.load(r'C:\\Users\\Bhumi\\OneDrive\\Desktop\\project\\women_safety_risk_prediction_model.pkl')
scaler = joblib.load(r'C:\\Users\\Bhumi\\OneDrive\\Desktop\\project\\scaler.pkl')


# ---------------------------------------------------------
# STREAMLIT PAGE
# ---------------------------------------------------------

st.set_page_config(
    page_title="Women Safety Prediction",
    page_icon="🛡️"
)

st.title("🛡️ Women Safety Prediction")
st.write("Enter details to predict Risk :.")


# ---------------------------------------------------------
# USER INPUT
# ---------------------------------------------------------

city = st.selectbox(
    "Enter your City:",
    sorted(df["city"].unique())
)


area = st.selectbox(
    "Enter your Area:",
    sorted(df["area"].unique())
)


latitude = st.number_input(
    "Enter Latitude:",
    min_value=float(df["latitude"].min()),
    max_value=float(df["latitude"].max()),
    value=float(df["latitude"].mean()),
    step=0.0001
)


longitude = st.number_input(
    "Enter Longitude:",
    min_value=float(df["longitude"].min()),
    max_value=float(df["longitude"].max()),
    value=float(df["longitude"].mean()),
    step=0.0001
)


crime_type = st.selectbox(
    "Enter Crime Type:",
    sorted(df["crime_type"].unique())
)


crime_count = st.number_input(
    "Enter Crime Count:",
    min_value=int(df["crime_count"].min()),
    max_value=int(df["crime_count"].max()),
    value=int(df["crime_count"].mean()),
    step=1
)


time_of_day = st.selectbox(
    "Enter Time of Day:",
    sorted(df["time_of_day"].unique())
)


lighting_score = st.number_input(
    "Enter Lighting Score:",
    min_value=float(df["lighting_score"].min()),
    max_value=float(df["lighting_score"].max()),
    value=float(df["lighting_score"].mean()),
    step=0.1
)


police_station_distance = st.number_input(
    "Enter Police Station Distance (km):",
    min_value=float(df["police_station_distance_km"].min()),
    max_value=float(df["police_station_distance_km"].max()),
    value=float(df["police_station_distance_km"].mean()),
    step=0.1
)


crowd_density = st.number_input(
    "Enter Crowd Density:",
    min_value=float(df["crowd_density"].min()),
    max_value=float(df["crowd_density"].max()),
    value=float(df["crowd_density"].mean()),
    step=1.0
)


weather_condition = st.selectbox(
    "Enter Weather Condition:",
    sorted(df["weather_condition"].unique())
)


safety_score = st.number_input(
    "Enter Safety Score:",
    min_value=float(df["safety_score"].min()),
    max_value=float(df["safety_score"].max()),
    value=float(df["safety_score"].mean()),
    step=0.1
)






# ---------------------------------------------------------
# FEATURE ENGINEERING
# Same formulas as train.py
# ---------------------------------------------------------

def label_encode(column, value):

    categories = sorted(
        df[column].dropna().unique()
    )

    mapping = {
        category: index
        for index, category in enumerate(categories)
    }

    return mapping[value]


city_encoded = label_encode("city", city)

area_encoded = label_encode("area", area)

crime_type_encoded = label_encode(
    "crime_type",
    crime_type
)

time_of_day_encoded = label_encode(
    "time_of_day",
    time_of_day
)

weather_condition_encoded = label_encode(
    "weather_condition",
    weather_condition
)



# ---------------------------------------------------------
# CREATE INPUT DATA
# EXACT SAME ORDER AS TRAINING DATA
# ---------------------------------------------------------

input_data = pd.DataFrame({

    "city": [city_encoded],

    "area": [area_encoded],

    "latitude": [latitude],

    "longitude": [longitude],

    "crime_type": [crime_type_encoded],

    "crime_count": [crime_count],

    "time_of_day": [time_of_day_encoded],

    "lighting_score": [lighting_score],

    "police_station_distance_km": [
        police_station_distance
    ],

    "crowd_density": [crowd_density],

    "weather_condition": [
        weather_condition_encoded
    ],

    "safety_score": [safety_score]

})



# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------

if st.button("🔮 Predict Risk Level"):

    try:

        # Prediction
        prediction = model.predict(input_data)

        # Result
        st.subheader("📊 Prediction Result")


        # Risk level mapping
        risk_labels = {
            0: "Critical",
            1: "High",
            2: "Low",
            3: "Medium"
        }


        predicted_risk = risk_labels.get(
            int(prediction[0]),
            str(prediction[0])
        )


        # Display result

        if predicted_risk == "Low":

            st.success(
                "🟢 Low Risk - This location is comparatively safe."
            )


        elif predicted_risk == "Medium":

            st.warning(
                "🟡 Medium Risk - Be careful in this location."
            )


        elif predicted_risk == "High":

            st.error(
                "🟠 High Risk - Extra caution is recommended."
            )


        elif predicted_risk == "Critical":

            st.error( "🔴 Critical Risk - Avoid this location if possible.")

        # ====================================================
        # DISPLAY CALCULATED FEATURES
        # ====================================================

        st.subheader("📋 Location Profile")

        col1, col2 = st.columns(2)


        with col1:

            st.write(
                "**City:**",
                city
            )

            st.write(
                "**Area:**",
                area
            )

            st.write(
                "**Crime Type:**",
                crime_type
            )

            st.write(
                "**Crime Count:**",
                crime_count
            )

            st.write(
                "**Lighting Score:**",
                lighting_score
            )

            st.write(
                "**Safety Score:**",
                safety_score
            )


        with col2:

            st.write(
                "**Latitude:**",
                latitude
            )

            st.write(
                "**Longitude:**",
                longitude
            )

            st.write(
                "**Time of Day:**",
                time_of_day
            )

            st.write(
                "**Police Station Distance:**",
                police_station_distance
            )

            st.write(
                "**Crowd Density:**",
                crowd_density
            )

            st.write(
                "**Weather Condition:**",
                weather_condition
            )


    except Exception as e:

        st.error("Prediction Error")

        st.exception(e)

        #  python -m streamlit run app.py
