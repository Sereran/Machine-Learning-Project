import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from preprocessing import load_and_process_data, create_basket_dataset
from models import CustomLogisticRegression

sns.set_theme(style="whitegrid")

def plot_model_performance(y_test, y_prob, y_pred, feature_names, weights):
    script_dir = os.path.dirname(os.path.abspath(__file__))

    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Label')
    plt.tight_layout()

    save_path = os.path.join(script_dir, 'confusion_matrix.png')

    plt.savefig(save_path)
    print(f"Saved in {save_path}")
    plt.close()

    try:
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        auc_score = roc_auc_score(y_test, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.tight_layout()

        save_path = os.path.join(script_dir, 'roc_curve.png')
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"Could not generate ROC curve {e}")

    coeffs = pd.Series(weights, index=feature_names).sort_values()
    top_features = pd.concat([coeffs.head(5), coeffs.tail(5)])
    
    plt.figure(figsize=(10, 6))
    top_features.plot(kind='barh', color='teal')
    plt.title('Top factors influencing purchase')
    plt.xlabel('Weight coefficient')
    plt.tight_layout()

    save_path = os.path.join(script_dir, 'feature_importance.png')
    plt.savefig(save_path)
    plt.close()

def run_task_2_1(data):
    if 'Crazy Schnitzel' not in data.columns:
        print("Error 'Crazy Schnitzel' not found")
        return
        
    subset = data[data['Crazy Schnitzel'] > 0].copy()

    target_col = 'Crazy Sauce'
    if target_col not in subset.columns:
        subset[target_col] = 0

    y = subset[target_col].apply(lambda x: 1 if x > 0 else 0).values

    drop_cols = [target_col, 'Crazy Schnitzel', 'id_bon', 'data_bon', 'total_value']
    feature_cols = [c for c in subset.columns if c not in drop_cols]
    
    X = subset[feature_cols].values

    X_norm = X.copy()
    mean = np.mean(X_norm, axis=0)
    std = np.std(X_norm, axis=0)
    std[std == 0] = 1
    X_norm = (X_norm - mean) / std

    X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=42)

    clf = CustomLogisticRegression(learning_rate=0.1, iterations=3000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_prob(X_test)

    print("MODEL EVALUATION RESULTS")
    print(f"Accuracy   {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision  {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"Recall     {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"F1 Score   {f1_score(y_test, y_pred, zero_division=0):.4f}")
    
    try:
        print(f"ROC-AUC    {roc_auc_score(y_test, y_prob):.4f}")
    except:
        print("ROC-AUC    N/A")

#BASELINE

    baseline_acc = max(y_test.mean(), 1 - y_test.mean())
    print(f"Baseline accuracy {baseline_acc:.4f}")

    if accuracy_score(y_test, y_pred) > baseline_acc:
        print("Our model is better")
    else:
        print("Same results as baseline")

    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Weight': clf.weights
    }).sort_values(by='Weight', ascending=False)

    print("FEATURE INTERPRETATION")
    
    print("Top 5 factors Iincreasing purchase probability")
    print(feature_importance.head(5).to_string(index=False))
    
    print("\nTop 5 factors decreasing purchase probability")
    print(feature_importance.tail(5).to_string(index=False))

    plot_model_performance(y_test, y_prob, y_pred, feature_cols, clf.weights)

if __name__ == "__main__":
    try:
        df_raw = load_and_process_data()

        if df_raw is not None:
            df_basket = create_basket_dataset(df_raw)
            run_task_2_1(df_basket)
        else:
            print("Could not load data")
        
    except FileNotFoundError:
        print("Error file not found")
    except Exception as e:
        print(f"An unexpected error occurred {e}")