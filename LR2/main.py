import pandas as pd
import numpy as np
from models import LogisticRegressionScratch

DATA_PATH = 'dataset_features_b.csv'
SAUCES = [
    "Crazy Sauce", "Cheddar Sauce", "Extra Cheddar Sauce",
    "Garlic Sauce", "Tomato Sauce", "Blueberry Sauce",
    "Spicy Sauce", "Pink Sauce"
]
TOP_K = 3


def main():
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print("Error: File not found. Did you run 'preprocessing.py' first?")
        return

    available_sauces = [s for s in SAUCES if s in df.columns]

    # Identify feature columns (everything that is NOT a sauce and NOT an ID)
    non_feature_cols = ['id_bon', 'data_bon', 'retail_product_name'] + available_sauces
    feature_cols = [c for c in df.columns if c not in non_feature_cols]

    print(f"Training on {len(feature_cols)} features (Products + Time data).")
    print(f"Targets: {available_sauces}")

    # Split Data (80% Train, 20% Test)
    np.random.seed(42)
    mask = np.random.rand(len(df)) < 0.8
    train_df = df[mask]
    test_df = df[~mask]

    X_train = train_df[feature_cols].values
    X_test = test_df[feature_cols].values

    # Train 8 Models (one per sauce)
    models = {}
    print("\n Starting Training:")
    for sauce_name in available_sauces:
        print(f"\n Training model for: {sauce_name}...")

        # Get target (1 if bought, 0 if not)
        y_train = train_df[sauce_name].values

        clf = LogisticRegressionScratch(learning_rate=0.1, n_iterations=1000)
        clf.fit(X_train, y_train)

        models[sauce_name] = clf

        k_values = [1, 3, 5]
        print(f" Evaluation:")

        all_probs = {}
        for sauce_name, model in models.items():
            all_probs[sauce_name] = model.predict_proba(X_test)

        probs_df = pd.DataFrame(all_probs, index=test_df.index)

        for k in k_values:
            hits = 0
            total_valid_cases = 0

            for idx in probs_df.index:
                actual_row = test_df.loc[idx]
                real_sauces = [s for s in available_sauces if actual_row[s] == 1]

                if len(real_sauces) == 0:
                    continue

                total_valid_cases += 1

                # Get recommendations
                pred_row = probs_df.loc[idx]
                recommendations = pred_row.sort_values(ascending=False).head(k).index.tolist()

                if any(s in recommendations for s in real_sauces):
                    hits += 1

            if total_valid_cases > 0:
                score = hits / total_valid_cases
                print(f"Hit@{k}: {score:.4f} ({score * 100:.2f}%)")


if __name__ == "__main__":
    main()