import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from models import LogisticRegressionScratch

DATA_PATH = 'dataset_features_b.csv'
sns.set_theme(style="whitegrid")


def get_data_for_confusion_matrix():

    print(" Preparing data for Confusion Matrix...")
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print("Error: Could not find dataset_features_b.csv")
        return None, None

    SAUCES = ["Crazy Sauce", "Cheddar Sauce", "Extra Cheddar Sauce", "Garlic Sauce",
              "Tomato Sauce", "Blueberry Sauce", "Spicy Sauce", "Pink Sauce"]

    # Identify non-feature columns
    non_feature_cols = ['id_bon', 'data_bon', 'retail_product_name'] + SAUCES
    feature_cols = [c for c in df.columns if c not in non_feature_cols]

    np.random.seed(42)
    mask = np.random.rand(len(df)) < 0.8
    train_df = df[mask]
    test_df = df[~mask]

    # Scaling / Standardization
    mean = train_df[feature_cols].mean()
    std = train_df[feature_cols].std().replace(0, 1)

    X_train = ((train_df[feature_cols] - mean) / std).values
    X_test = ((test_df[feature_cols] - mean) / std).values

    target_sauce = "Garlic Sauce"
    y_train = train_df[target_sauce].values
    y_test = test_df[target_sauce].values

    print(f"Quick training for {target_sauce}...")
    clf = LogisticRegressionScratch(learning_rate=0.1, n_iterations=1000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test, threshold=0.5)
    return y_test, y_pred


def plot_confusion_matrix():
    #Generates and saves the Confusion Matrix for Garlic Sauce.
    y_true, y_pred = get_data_for_confusion_matrix()
    if y_true is None: return

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix: Garlic Sauce', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('Actual Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('confusion_matrix_garlic.png', dpi=300)
    print("[OK] Saved: confusion_matrix_garlic.png")
    plt.close()


def plot_task_2_2_metrics():
    #Generates the bar chart for Task 2.2 using previous results
    metrics = ['Hit@1', 'Hit@3', 'Hit@5']
    values = [41.80, 80.36, 94.25]

    plt.figure(figsize=(8, 5))
    ax = sns.barplot(x=metrics, y=values, palette="viridis")

    # Add percentage text on top of bars
    for i, v in enumerate(values):
        ax.text(i, v + 2, f"{v}%", ha='center', fontweight='bold')

    plt.ylim(0, 110)
    plt.title('Task 2.2: Sauce Recommendation Performance', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.tight_layout()
    plt.savefig('hit_k_task_2_2.png', dpi=300)
    print("[OK] Saved: hit_k_task_2_2.png")
    plt.close()


def plot_task_3_comparison():
    # Generates the comparison chart for Task 3:
    # Data structure for Seaborn
    data = {
        'Metric': ['Hit@1', 'Hit@1', 'Hit@3', 'Hit@3', 'Hit@5', 'Hit@5'],
        'Model': ['Baseline', 'Naive Bayes', 'Baseline', 'Naive Bayes', 'Baseline', 'Naive Bayes'],
        'Score (%)': [13.40, 18.40, 33.40, 37.80, 42.40, 50.60]
    }
    df_plot = pd.DataFrame(data)

    plt.figure(figsize=(9, 6))

    ax = sns.barplot(x='Metric', y='Score (%)', hue='Model', data=df_plot, palette="muted")

    # Add percentage text on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', padding=3, fontweight='bold')

    plt.ylim(0, 60)
    plt.title('Task 3: Baseline (Popularity) vs. Naive Bayes (Ranking)', fontsize=14)
    plt.legend(title='Model', loc='upper left')
    plt.tight_layout()
    plt.savefig('comparison_task_3.png', dpi=300)
    print(" Saved: comparison_task_3.png")
    plt.close()


if __name__ == "__main__":
    plot_confusion_matrix()
    plot_task_2_2_metrics()
    plot_task_3_comparison()
    print("\n All plots have been generated successfully!")