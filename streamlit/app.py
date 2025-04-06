import streamlit as st
import requests
import time
import pandas as pd
import os
from PIL import Image

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="🔍 Two-Step Product Search", layout="centered")

# -------------------------------------------------
# SESSION STATE
# -------------------------------------------------
if 'session_id' not in st.session_state:
    st.session_state.session_id = None
if 'product_data' not in st.session_state:
    st.session_state.product_data = ""
if 'llm_output' not in st.session_state:
    st.session_state.llm_output = ""

# -------------------------------------------------
# BACKEND URLS
# -------------------------------------------------
# 1) For text-based product search + LLM
BASE_URL_TEXT = "http://192.168.1.227:8006"
SEARCH_URL_TEXT = f"{BASE_URL_TEXT}/product_search"
LLM_URL = f"{BASE_URL_TEXT}/product_search_llm"

# 2) For image-based similarity search
BASE_URL_IMAGE = "http://192.168.1.227:8001"
SEARCH_URL_IMAGE = f"{BASE_URL_IMAGE}/search-image"

# -------------------------------------------------
# DATAFRAME FOR LOCAL IMAGE LOOKUP
# -------------------------------------------------
# If you want to display matched images from disk, load a CSV that has:
#    "ProductId" and "full_path" columns
df = pd.read_csv(r"D:\Anand\Jsearch_ai\image search\data\fashion_with_image_paths.csv")
# Example row:  ProductId=42419, full_path=D:\some\folder\42419.jpg

# -------------------------------------------------
# MAIN TITLE
# -------------------------------------------------
st.title("🔍 Two-Step Product Search with LLM Stream")

# -------------------------------------------------
# RADIO: CHOOSE SEARCH MODE
# -------------------------------------------------
search_mode = st.radio("Search mode:", ["Text", "Image"])

# -------------------------------------------------
# 1) TEXT MODE
# -------------------------------------------------
if search_mode == "Text":
    query = st.text_input("Enter your text query", placeholder="e.g. 'I need black sneakers'")
    k_value = st.number_input("Number of results (k)", min_value=1, max_value=2000, value=5, step=1)
    
    # When user clicks "Search," call the text-based endpoint
    if st.button("Search"):
        if not query.strip():
            st.error("Please enter a valid query.")
            st.stop()
        
        # Call your text-based product_search API
        with st.spinner("Fetching product data..."):
            resp = requests.post(SEARCH_URL_TEXT, json={"query": query, "k": k_value})
            if resp.status_code != 200:
                st.error(f"Text Search Error {resp.status_code}: {resp.text}")
                st.stop()
            
            data = resp.json()
            st.session_state.session_id = data.get("session_id")
            st.session_state.product_data = data.get("product_data", "")
            st.session_state.llm_output = ""
        
        st.success("Product data retrieved! See below.")

    # Display product data if available
    if st.session_state.product_data:
        st.subheader("📦 Retrieved Products")
        st.text_area("Product Results:", st.session_state.product_data, height=200)
    
    # Separate button for LLM summary (only if we have a session ID)
    if st.session_state.session_id:
        if st.button("✨ Generate LLM Summary"):
            llm_placeholder = st.empty()
            st.session_state.llm_output = ""
            
            with st.spinner("Generating summary..."):
                with requests.post(LLM_URL, json={"session_id": st.session_state.session_id}, stream=True) as r:
                    if r.status_code != 200:
                        st.error(f"LLM Error {r.status_code}: {r.text}")
                        st.stop()
                    
                    # Stream the chunks
                    for chunk in r.iter_content(chunk_size=128):
                        if chunk:
                            st.session_state.llm_output += chunk.decode("utf-8")
                            llm_placeholder.markdown(st.session_state.llm_output + " ▌")
                            time.sleep(0.02)
            
            llm_placeholder.markdown(st.session_state.llm_output)
            st.success("🎉 Summary generation complete!")

# -------------------------------------------------
# 2) IMAGE MODE
# -------------------------------------------------
elif search_mode == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
    top_k_img = st.slider("Number of results (top_k)", 1, 20, 5)

    # If user uploads a file and clicks "Search"
    if uploaded_file is not None:
        if st.button("Search"):
            with st.spinner("Searching by image..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                params = {"top_k": top_k_img}
                resp = requests.post(SEARCH_URL_IMAGE, files=files, params=params)
                
                if resp.status_code != 200:
                    st.error(f"Image Search Error {resp.status_code}: {resp.text}")
                    st.stop()

                data = resp.json()
                results = data.get("results", [])
                
                st.write(f"Found {len(results)} image-based results.")
                
                # Display them in a grid of 3 per row
                images_per_row = 3
                for i in range(0, len(results), images_per_row):
                    row_results = results[i : i + images_per_row]
                    cols = st.columns(len(row_results))
                    
                    for colz, item in zip(cols, row_results):
                        pid = item["product_id"]
                        score = item["similarity_score"]
                        
                        # Look up local path in DataFrame
                        matching = df.loc[df["ProductId"] == pid]
                        if not matching.empty:
                            row_ = matching.iloc[0]
                            local_path = row_["full_path"]
                            
                            if os.path.isfile(local_path):
                                colz.write(f"**PID:** {pid}, Score: {score:.4f}")
                                colz.image(local_path, use_container_width=True)
                            else:
                                colz.write(f"File not found on disk: {local_path}")
                        else:
                            colz.write(f"No row in CSV for PID={pid}")
