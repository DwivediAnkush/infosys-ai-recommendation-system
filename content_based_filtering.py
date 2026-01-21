# import pandas as pd
# import numpy as np
# import sklearn
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# def content_based_recommendation(data, item_name, top_n=10):
#   if item_name not in data['Name'].values:
#     print(f"item '{item_name}' not found in the data.")
#     return pd.DataFrame()

#   tfidf_vectorizer = TfidfVectorizer(stop_words='english')
#   tfidf_matrix_content = tfidf_vectorizer.fit_transform(data['Tags'])
#   cosine_similarity_content = cosine_similarity(tfidf_matrix_content, tfidf_matrix_content)

#   item_index = data[data['Name']==item_name].index[0]

#   similar_items = list(enumerate(cosine_similarity_content[item_index]))
#   similar_prod = sorted(similar_items, key=lambda x:x[1], reverse=True)

#   top_similar_prod = similar_prod[:top_n]

#   recommended_items_indices = [x[0] for x in top_similar_prod]

#   recommended_item_details = data.iloc[recommended_items_indices][['Name', 'ReviewCount', 'Brand']]

#   return recommended_item_details


# # TO test the system
# if _name_ == "_main_":
#     import pandas as pd
#     from preprocess_data import process_data

#     raw_data = pd.read_csv("clean_data.csv")
#     data = process_data(raw_data)

#     item_name = "OPI Infinite Shine, Nail Lacquer Nail Polish, Bubble Bath"
#     result = content_based_recommendation(data, item_name, top_n=5)
#     print(result)



import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


def content_based_recommendation(data, item_name, top_n=10):
    """
    Content-based recommendation using TF-IDF and cosine similarity
    """

    # -----------------------------
    # Use a clean copy & reset index
    # -----------------------------
    df = data.copy().reset_index(drop=True)

    # -----------------------------
    # Check item existence
    # -----------------------------
    if item_name not in df["Name"].values:
        return pd.DataFrame()

    # -----------------------------
    # Text column preparation
    # -----------------------------
    df["combined_features"] = (
        df["Name"].fillna("") + " " +
        df["Category"].fillna("")
    )

    # -----------------------------
    # TF-IDF Vectorization
    # -----------------------------
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["combined_features"])

    # -----------------------------
    # Cosine Similarity
    # -----------------------------
    cosine_sim = cosine_similarity(tfidf_matrix)

    # -----------------------------
    # Get index safely
    # -----------------------------
    item_index = df[df["Name"] == item_name].index[0]

    # -----------------------------
    # Similarity scores
    # -----------------------------
    similarity_scores = list(enumerate(cosine_sim[item_index]))

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    similarity_scores = similarity_scores[1:top_n + 1]

  
    # Fetch recommendations
    
    item_indices = [i[0] for i in similarity_scores]

    return df.iloc[item_indices]