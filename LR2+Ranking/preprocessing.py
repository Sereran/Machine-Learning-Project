import pandas as pd
import numpy as np
import os

def get_data_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    file_path = os.path.join(parent_dir, 'ap_dataset.csv')
    return file_path

def get_data(filepath=None):
    if filepath is None:
        filepath = get_data_path()
        
    if not os.path.exists(filepath):
        print(f"Error file not found at {filepath}")
        return None
        
    df = pd.read_csv(filepath)

    df['data_bon'] = pd.to_datetime(df['data_bon'])
    df['day_of_week'] = df['data_bon'].dt.dayofweek + 1
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 6 else 0)
    df['hour'] = df['data_bon'].dt.hour

    bon_stats = df.groupby('id_bon').agg({
        'SalePriceWithVAT': 'sum',
        'retail_product_name': ['count', 'nunique'],
        'day_of_week': 'first',
        'is_weekend': 'first',
        'hour': 'first'
    })

    bon_stats.columns = ['total_value', 'cart_size', 'distinct_products', 'day_of_week', 'is_weekend', 'hour']
    bon_stats.reset_index(inplace=True)

    # Rows = Receipts, Columns = Products, Value = 1 if bought, 0 if not
    product_matrix = df.pivot_table(index='id_bon',
                                    columns='retail_product_name',
                                    values='SalePriceWithVAT',
                                    aggfunc='count').fillna(0)

    product_matrix = (product_matrix > 0).astype(int)

    final_df = bon_stats.merge(product_matrix, on='id_bon')

    return final_df


if __name__ == "__main__":
    data = get_data()
    print(f"Data processed! Shape: {data.shape}")
    print(data.head())
    data.to_csv('dataset_features_B.csv', index=False)