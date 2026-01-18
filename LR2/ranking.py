import pandas as pd
import numpy as np
from models import NaiveBayesRanking

DATA_PATH = 'dataset_features_b.csv'
N_TEST_SAMPLES = 500


def main():
    print("Loading data for Ranking Evaluation...")
    df = pd.read_csv(DATA_PATH)

    sauces = [
        "Crazy Sauce", "Cheddar Sauce", "Extra Cheddar Sauce",
        "Garlic Sauce", "Tomato Sauce", "Blueberry Sauce",
        "Spicy Sauce", "Pink Sauce"
    ]
    meta_cols = ['id_bon', 'data_bon', 'total_value', 'cart_size', 'distinct_products', 'day_of_week', 'is_weekend',
                 'hour']

    # Everything else is a product we can recommend
    candidate_products = [c for c in df.columns if c not in meta_cols and c not in sauces]
    print(f"Ranking {len(candidate_products)} candidate products.")

    # Train Naive Bayes Models (One per product)
    print("Training Naive Bayes models...")
    feature_cols = ['day_of_week', 'is_weekend', 'hour']
    X = df[feature_cols]

    nb_models = {}
    for product in candidate_products:
        y = df[product].values
        if y.sum() > 0:
            model = NaiveBayesRanking()
            model.fit(X, y)
            nb_models[product] = model

    # Systematic Evaluation (Leave-One-Out)
    print(f"\n Starting Evaluation on {N_TEST_SAMPLES} random carts...")

    df['candidate_count'] = df[candidate_products].sum(axis=1)
    valid_carts = df[df['candidate_count'] > 0].copy()

    test_indices = np.random.choice(valid_carts.index, size=min(N_TEST_SAMPLES, len(valid_carts)), replace=False)
    test_set = valid_carts.loc[test_indices]

    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0

    for i, (idx, row) in enumerate(test_set.iterrows()):
        if i % 50 == 0:
            print(f"Processing cart {i}/{len(test_set)}...")

        actual_items = [p for p in candidate_products if row[p] == 1]

        hidden_item = np.random.choice(actual_items)

        context = pd.DataFrame([row[feature_cols]])

        scores = []
        for product, model in nb_models.items():
            if product in actual_items and product != hidden_item:
                continue

            prob = model.predict_proba(context)[0]
            scores.append((product, prob))

        scores.sort(key=lambda x: x[1], reverse=True)

        if i == 0:
            print(f"\n Example Recommendation for Cart #{idx}:")
            print(f"Real Hidden Item: {hidden_item}")
            print("System Recommended:")
            for rank, (prod, prob) in enumerate(scores[:5]):
                print(f"  #{rank + 1}: {prod}")
            print(f"\n")

        ranked_products = [s[0] for s in scores]

        if hidden_item in ranked_products[:1]:
            hits_at_1 += 1
        if hidden_item in ranked_products[:3]:
            hits_at_3 += 1
        if hidden_item in ranked_products[:5]:
            hits_at_5 += 1

    print("\n RESULTS: General Product Ranking:")
    print(f"Total Carts Tested: {len(test_set)}")
    print(f"Hit@1: {hits_at_1 / len(test_set):.4f}")
    print(f"Hit@3: {hits_at_3 / len(test_set):.4f}")
    print(f"Hit@5: {hits_at_5 / len(test_set):.4f}")


if __name__ == "__main__":
    main()