"""
Generate synthetic health insurance claims data.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def generate_synthetic_data(n_samples=1000, seed=42):
    """
    Generate synthetic health insurance claims data.
    
    Args:
        n_samples (int): Number of samples to generate
        seed (int): Random seed for reproducibility
        
    Returns:
        pd.DataFrame: Generated data
    """
    np.random.seed(seed)
    
    # Generate basic demographic features
    data = {
        'age': np.random.randint(18, 65, n_samples),
        'gender': np.random.choice(['male', 'female'], n_samples),
        'bmi': np.random.normal(26, 4, n_samples),  # Mean BMI of 26 with std of 4
        'children': np.random.randint(0, 5, n_samples),
        'smoker': np.random.choice(['yes', 'no'], n_samples),
        'region': np.random.choice(['northeast', 'northwest', 'southeast', 'southwest'], n_samples)
    }
    
    # Clean up BMI to realistic ranges
    data['bmi'] = np.clip(data['bmi'], 16, 45)
    
    # Generate claim amount based on features
    base_amount = 5000
    claim_amount = base_amount + np.zeros(n_samples)
    
    # Age factor
    claim_amount += data['age'] * 100
    
    # BMI factor
    bmi_factor = np.where(data['bmi'] > 30, 
                         (data['bmi'] - 30) * 500, 
                         0)
    claim_amount += bmi_factor
    
    # Smoker factor
    smoker_factor = np.where(np.array(data['smoker']) == 'yes', 
                            15000, 
                            0)
    claim_amount += smoker_factor
    
    # Children factor
    claim_amount += data['children'] * 2000
    
    # Add some random variation
    claim_amount *= np.random.normal(1, 0.1, n_samples)
    
    # Ensure all amounts are positive and round to 2 decimal places
    data['claim_amount'] = np.round(np.maximum(claim_amount, 0), 2)
    
    return pd.DataFrame(data)

def save_data(df, output_dir, filename='health_claims.csv'):
    """
    Save the generated data to a CSV file.
    
    Args:
        df (pd.DataFrame): Data to save
        output_dir (str): Output directory
        filename (str): Output filename
    """
    output_path = Path(output_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")
    
    # Print data statistics
    print("\nData Statistics:")
    print("-" * 40)
    print(f"Number of samples: {len(df)}")
    print("\nNumerical Features Statistics:")
    print(df[['age', 'bmi', 'children', 'claim_amount']].describe())
    print("\nCategorical Features Distribution:")
    for col in ['gender', 'smoker', 'region']:
        print(f"\n{col.title()} Distribution:")
        print(df[col].value_counts())

if __name__ == "__main__":
    # Generate synthetic data
    data = generate_synthetic_data(n_samples=1000, seed=42)
    
    # Save the data
    save_data(data, 'health-insurance-mlops/data')
