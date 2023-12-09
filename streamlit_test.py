
import streamlit as st

# Title
st.title("My Streamlit App")

# Data input
user_input = st.text_input("Enter some text")

# Display output
st.write("You entered:", user_input)
