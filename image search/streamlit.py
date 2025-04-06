import streamlit as st
import pandas as pd
import requests
from PIL import Image
import os

# Point this to your FastAPI endpoint
API_ENDPOINT = "http://192.168.1.227:8001/search-image"


# 1) Load your CSV into a pandas DataFrame
#    Make sure your CSV includes columns: [ProductId, ..., full_path]
df = pd.read_csv(r"D:\Anand\Jsearch_ai\image search\data\fashion_with_image_paths.csv")  # adjust the path/filename
st.write("DataFrame loaded with", len(df), "rows.")
st.title("Image Similarity Search")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
top_k = st.slider("Number of results (top_k):", 1, 20, 5)

if uploaded_file is not None:
    st.image(uploaded_file, caption="Query Image", use_container_width=True)
    
    if st.button("Search"):
        # Send the image to FastAPI
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        params = {"top_k": top_k}
        response = requests.post(API_ENDPOINT, files=files, params=params)
        
        if response.status_code == 200:
            data = response.json()
            results = data["results"]  # list of {product_id, similarity_score}
            
            st.write(f"Found {len(results)} results.")
            
            # Let's show 3 images per row
            images_per_row = 3
            
            for i in range(0, len(results), images_per_row):
                row_results = results[i : i + images_per_row]
                
                # Create a set of columns (one per image in this row)
                cols = st.columns(len(row_results))
                
                for col, item in zip(cols, row_results):
                    pid = item["product_id"]
                    score = item["similarity_score"]
                    
                    # Look up the path in our DataFrame
                    matching_rows = df.loc[df["ProductId"] == pid]
                    if not matching_rows.empty:
                        row = matching_rows.iloc[0]
                        local_path = row["full_path"]
                        
                        if os.path.isfile(local_path):
                            # Display the image and info in this column
                            col.write(f"**PID:** {pid}, Score: {score:.4f}")
                            col.image(local_path, caption=os.path.basename(local_path), use_container_width=True)
                        else:
                            col.write(f"File not found on disk: {local_path}")
                    else:
                        col.write(f"No row found for ProductId={pid}")
        else:
            st.error(f"API Error: {response.status_code}")