import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="⚙️",
    layout="wide"
)

model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("⚙️ Predictive Maintenance Dashboard")
st.markdown("---")

# #########  Sidebarr

st.sidebar.header("Equipment Sensor Inputs")

temperature = st.sidebar.number_input("Temperature", 25.0)
vibration = st.sidebar.number_input("Vibration", 0.0)
pressure = st.sidebar.number_input("Pressure", 0.0)
humidity = st.sidebar.number_input("Humidity", 0.0)
rotation_speed = st.sidebar.number_input("Rotation Speed", 0.0)
voltage = st.sidebar.number_input("Voltage", 0.0)
current = st.sidebar.number_input("Current", 0.0)
oil_level = st.sidebar.number_input("Oil Level", 0.0)
load = st.sidebar.number_input("Load", 0.0)
motor_temperature = st.sidebar.number_input("Motor Temperature", 0.0)
gearbox_temperature = st.sidebar.number_input("Gearbox Temperature", 0.0)
sound_level = st.sidebar.number_input("Sound Level", 0.0)
fan_speed = st.sidebar.number_input("Fan Speed", 0.0)
reactive_power = st.sidebar.number_input("Reactive Power", 0.0)
active_power = st.sidebar.number_input("Active Power", 0.0)

temp_difference = motor_temperature - gearbox_temperature


col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Temperature", f"{temperature:.1f} °C")

with col2:
    st.metric("Pressure", f"{pressure:.1f}")

with col3:
    st.metric("Vibration", f"{vibration:.1f}")

with col4:
    st.metric("Load", f"{load:.1f}%")


input_data = pd.DataFrame({
    'temperature':[temperature],
    'vibration':[vibration],
    'pressure':[pressure],
    'humidity':[humidity],
    'rotation_speed':[rotation_speed],
    'voltage':[voltage],
    'current':[current],
    'oil_level':[oil_level],
    'load':[load],
    'motor_temperature':[motor_temperature],
    'gearbox_temperature':[gearbox_temperature],
    'sound_level':[sound_level],
    'fan_speed':[fan_speed],
    'reactive_power':[reactive_power],
    'active_power':[active_power],
    'temp_difference':[temp_difference]
})

st.subheader("📊 Sensor Readings")

display_table = pd.DataFrame({
    "Sensor": input_data.columns,
    "Value": input_data.iloc[0].values
})

st.dataframe(
    display_table,
    use_container_width=True
)

if st.sidebar.button("Predict Failure Risk"):

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    probability = model.predict_proba(input_scaled)[0][1]

    failure_percent = probability * 100

    st.subheader("🔍 Failure Risk Assessment")

    st.progress(int(failure_percent))

    st.metric(
        "Failure Probability",
        f"{failure_percent:.2f}%"
    )

    st.subheader("🛠 Maintenance Recommendation")

    if failure_percent >= 70:

        recommendation = """
        - Perform immediate inspection
        - Check vibration levels
        - Inspect motor bearings
        - Verify lubrication
        - Schedule maintenance
        """

    elif failure_percent >= 30:

        recommendation = """
        - Monitor equipment closely
        - Increase inspection frequency
        - Review sensor readings
        """

    else:

        recommendation = """
        - Continue normal operation
        - Follow scheduled maintenance plan
        """

    st.markdown(recommendation)



    st.subheader("📋 Equipment Summary")

    summary = pd.DataFrame({
        "Metric":[
            "Prediction",
            "Failure Probability",
            "Temperature",
            "Load",
            "Vibration"
        ],
        "Value":[
            "Failure" if prediction[0]==1 else "Normal",
            f"{failure_percent:.2f}%",
            temperature,
            load,
            vibration
        ]
    })

    st.table(summary)


st.subheader("📈 Key Failure Indicators")

importance = pd.DataFrame({
    "Feature":[
        "Vibration",
        "Temperature",
        "Load",
        "Active Power",
        "Oil Level"
    ],
    "Importance":[0.35,0.25,0.15,0.12,0.08]
})

st.bar_chart(
    importance.set_index("Feature")
)