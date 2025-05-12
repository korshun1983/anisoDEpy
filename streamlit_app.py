import streamlit as st
import pandas as pd
import numpy as np

number = st.number_input(
    "Insert a number", value=None, placeholder="Type a number..."
)

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    st.write(uploaded_file.name)