import streamlit as st
import pandas as pd
from main import load_data, create_model, search
import re

# Page config
st.set_page_config(page_title="LyriFind", layout="centered")

df = load_data()

# Sidebar filter
selected_artist = st.sidebar.selectbox("Select Artist", ["All"] + sorted(df['artist'].unique().tolist()))

if selected_artist != "All":
    df = df[df['artist'] == selected_artist].reset_index(drop=True)

# ✅ NOW create model AFTER filtering
vectorizer, tfidf_matrix = create_model(df)

# Title
st.markdown("<h1 style='text-align: center;'>🎧 LyriFind</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Search songs using lyrics</p>", unsafe_allow_html=True)

# Input
query = st.text_input("🔍 Enter lyrics:")

# Highlight function
def highlight(text, query):
    words = query.split()
    for w in words:
        text = re.sub(f"({w})", r"<mark>\1</mark>", text, flags=re.IGNORECASE)
    return text

# Search
if query:
    results = search(query, df, vectorizer, tfidf_matrix)

    st.subheader("🎯 Top Results")

    for res in results[:5]:  # Top 5
        st.markdown(f"### 🎵 {res['title']}")
        
        # Artist
        song_artist = df[df['title'] == res['title']]['artist'].values[0]
        st.write(f"👤 Artist: {song_artist}")

        # Score bar
        st.progress(res['score'])

        # Highlight preview
        preview = highlight(res['preview'], query)
        st.markdown(f"📝 {preview}...", unsafe_allow_html=True)

        st.divider()