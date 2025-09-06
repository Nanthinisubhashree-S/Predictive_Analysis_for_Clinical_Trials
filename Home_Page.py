import streamlit as st

st.title("CLINICAL TRAILS")

st.write("Choose a model to predict:")

col1, col2 = st.columns(2)

with col1:
    if st.button("Country Prediction"):
        st.switch_page("pages/Country_Prediction.py")

with col2:
    if st.button("Weeks Prediction"):
        st.switch_page("pages/Weeks_Prediction.py")
