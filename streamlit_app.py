import streamlit as st
import pandas as pd
import numpy as np

st.title("Anisotropic waveguide modes calculation and identification program")

uploaded_file = st.file_uploader("Choose a file with material properties...")
if uploaded_file is not None:
    st.write(uploaded_file.name)

with st.form("PML_input_container"):
    st.header("Add PML layer")
    PML_type = st.selectbox("PML type",
    ("pml", "abc", "abc+pml", "same", "none"),)
    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    if submitted:
        st.write("PML", PML_type)