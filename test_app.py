import streamlit as st

st.title("🚔 Police Call Analytics - Test")
st.write("If you can see this, Streamlit is working!")

# Simple file uploader test
uploaded_file = st.file_uploader("Test Upload", type=['txt', 'wav', 'mp3'])

if uploaded_file is not None:
    st.success(f"File uploaded: {uploaded_file.name}")
    st.write(f"File size: {len(uploaded_file.getvalue())} bytes")

st.write("✅ Basic Streamlit functionality working!")
