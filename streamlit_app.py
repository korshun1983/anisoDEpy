import streamlit as st
import pandas as pd
import numpy as np

t = np.arange(0, 100, 0.1, dtype = float)
y = np.sin(t)

chart_data = pd.DataFrame(
    {
        "col1": t,
        "col2": y
    }
)

st.line_chart(
    chart_data,
    x="col1",
    y="col2",
    x_label="t",
    y_label="y"
)
