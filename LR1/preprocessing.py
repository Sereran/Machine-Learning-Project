import pandas as pd
import numpy as np
import os

def get_data_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    file_path = os.path.join(parent_dir, 'ap_dataset.csv')
    return file_path

def load_and_process_data(filepath=None):
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
    
    return df

def create_basket_dataset(df):
    basket_meta = df.groupby('id_bon').agg({
        'day_of_week': 'first',
        'is_weekend': 'first',
        'hour': 'first',
        'SalePriceWithVAT': 'sum',
        'retail_product_name': 'count'
    }).rename(columns={'SalePriceWithVAT': 'total_value', 'retail_product_name': 'cart_size'})

    basket_items = df.pivot_table(
        index='id_bon', 
        columns='retail_product_name', 
        values='SalePriceWithVAT', 
        aggfunc='count', 
        fill_value=0
    )

    basket_meta['distinct_products'] = (basket_items > 0).sum(axis=1)

    full_data = basket_meta.join(basket_items)
    
    return full_data