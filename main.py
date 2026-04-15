import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# 🔹 Load Dataset (Hybrid: Online + Local)
def load_data():
    try:
        df = pd.read_csv("lyrics_dataset_100.csv")
        df = df[['Title', 'Lyric']].dropna()
        df.rename(columns={'Title': 'title', 'Lyric': 'lyrics'}, inplace=True)

        # Add dummy artist column if not present
        df['artist'] = ["Artist " + str(i % 10 + 1) for i in range(len(df))]

        df = df.head(200)
        print("✅ Loaded online dataset")

    except:
        df = pd.read_csv("lyrics_dataset_100.csv")

        # Ensure required columns exist
        if 'artist' not in df.columns:
            df['artist'] = "Unknown Artist"

        print("✅ Loaded local dataset")

    return df


# 🔹 Create TF-IDF Model
def create_model(df):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['lyrics'])
    return vectorizer, tfidf_matrix


# 🔹 Search Function
def search(query, df, vectorizer, tfidf_matrix):
    query_vec = vectorizer.transform([query])
    similarity = cosine_similarity(query_vec, tfidf_matrix)

    scores = list(enumerate(similarity[0]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    results = []

    for i in scores[:10]:  # Top 10 for flexibility
        results.append({
            "title": df.iloc[i[0]]['title'],
            "artist": df.iloc[i[0]]['artist'],
            "score": float(i[1]),
            "preview": df.iloc[i[0]]['lyrics'][:150]
        })

    return results