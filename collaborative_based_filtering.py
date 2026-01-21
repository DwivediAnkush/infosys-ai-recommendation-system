import pandas as pd
import numpy as np

# Try to import sklearn's cosine_similarity, otherwise use a numpy fallback.
try:
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    def cosine_similarity(X, Y=None):
        if Y is None:
            Y = X
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        # normalize rows
        X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)
        Y_norm = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-10)
        return np.dot(X_norm, Y_norm.T)

def collaborative_filtering_recommendations(data, target_user_id, top_n=10):
    """
    Simple user-based collaborative filtering recommendations.
    Returns a DataFrame of recommended products with predicted score and basic metadata.
    """
    # Build user-item matrix
    user_item_matrix = data.pivot_table(index='ID', columns='ProdID', values='Rating', aggfunc='mean').fillna(0)
    if target_user_id not in user_item_matrix.index:
        raise ValueError(f"target_user_id {target_user_id} not found in data")

    # Compute user-user similarity
    user_similarity = cosine_similarity(user_item_matrix.values)

    # Predict scores for all items as weighted sum of other users' ratings
    target_idx = user_item_matrix.index.get_loc(target_user_id)
    sim_scores = user_similarity[target_idx]  # shape (n_users,)

    ratings_matrix = user_item_matrix.values  # shape (n_users, n_items)
    # Weighted sum of ratings
    denom = np.sum(np.abs(sim_scores)) + 1e-10
    predicted_scores = np.dot(sim_scores, ratings_matrix) / denom  # shape (n_items,)

    # Exclude items the target user has already rated
    target_rated_mask = user_item_matrix.iloc[target_idx] > 0
    predicted_scores[target_rated_mask.values] = -np.inf

    # Select top N items
    top_indices = np.argsort(predicted_scores)[-top_n:][::-1]
    prod_ids = user_item_matrix.columns[top_indices]

    # Gather product metadata (first occurrence per ProdID)
    prod_info = data.drop_duplicates(subset=['ProdID']).set_index('ProdID')
    # Ensure requested columns exist; use safe .reindex to preserve order
    cols = []
    for c in ['Name', 'ReviewCount', 'Brand', 'ImageURL']:
        if c in prod_info.columns:
            cols.append(c)
    recommended = prod_info.reindex(prod_ids)[cols].copy()
    recommended['PredictedScore'] = predicted_scores[top_indices]
    # Add average rating if available
    if 'Rating' in data.columns:
        avg_rating = data.groupby('ProdID')['Rating'].mean()
        recommended['Rating'] = avg_rating.reindex(prod_ids).values

    recommended = recommended.reset_index()
    return recommended

if __name__ == "__main__":
    from preprocess_data import process_data

    raw_data = pd.read_csv("clean_data.csv")
    data = process_data(raw_data)

    target_user_id = 4
    print(collaborative_filtering_recommendations(data, target_user_id))