import pandas as pd
import numpy as np

DATA_PATH = 'dataset_features_b.csv'
N_TEST_SAMPLES = 500


def main():
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print("Error: dataset_features_b.csv not found.")
        return

    sauces = [
        "Crazy Sauce", "Cheddar Sauce", "Extra Cheddar Sauce",
        "Garlic Sauce", "Tomato Sauce", "Blueberry Sauce",
        "Spicy Sauce", "Pink Sauce"
    ]
    meta_cols = ['id_bon', 'data_bon', 'total_value', 'cart_size', 'distinct_products', 'day_of_week', 'is_weekend',
                 'hour']
    candidate_products = [c for c in df.columns if c not in meta_cols and c not in sauces]

    # Calculate Global Popularity
    popularity = df[candidate_products].sum().sort_values(ascending=False)
    global_ranking = popularity.index.tolist()

    print("\n Global Top 5 Products:")
    print(popularity.head(5))

    # Evaluate Baseline (Leave-One-Out)
    print(f"\n Evaluating Baseline on {N_TEST_SAMPLES} random carts...")

    df['candidate_count'] = df[candidate_products].sum(axis=1)
    valid_carts = df[df['candidate_count'] > 0].copy()

    # Sample random carts
    np.random.seed(42)
    test_indices = np.random.choice(valid_carts.index, size=min(N_TEST_SAMPLES, len(valid_carts)), replace=False)
    test_set = valid_carts.loc[test_indices]

    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0

    for idx, row in test_set.iterrows():
        actual_items = [p for p in candidate_products if row[p] == 1]

        # Hide one item randomly
        hidden_item = np.random.choice(actual_items)

        # Suggest the Global Top K items that are NOT already in the visible cart
        visible_items = set(actual_items) - {hidden_item}

        recommendations = []
        for item in global_ranking:
            if item not in visible_items:
                recommendations.append(item)
            if len(recommendations) >= 5:
                break

        if hidden_item in recommendations[:1]:
            hits_at_1 += 1
        if hidden_item in recommendations[:3]:
            hits_at_3 += 1
        if hidden_item in recommendations[:5]:
            hits_at_5 += 1

    print("\n BASELINE RESULTS (Popularity):")
    print(f"Hit@1: {hits_at_1 / len(test_set):.4f}")
    print(f"Hit@3: {hits_at_3 / len(test_set):.4f}")
    print(f"Hit@5: {hits_at_5 / len(test_set):.4f}")


if __name__ == "__main__":
    main()