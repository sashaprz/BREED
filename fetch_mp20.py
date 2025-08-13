from mp_api.client import MPRester
import pandas as pd
import os

with MPRester("tQ53EaqRe8UndenrzdDrDcg3vZypqn0d") as mpr:
    docs = mpr.materials.summary.search(
        fields=["material_id", "structure", "formation_energy_per_atom"],
        is_stable=True
    )
    data = [
        {
            'material_id': doc.material_id,
            'cif': doc.structure.to(fmt='cif'),
            'formation_energy': doc.formation_energy_per_atom
        } for doc in docs
    ]
    
    df = pd.DataFrame(data)
    
    # Reset index to ensure clean integer indices (fixes KeyError issues)
    df = df.reset_index(drop=True)
    
    # Train set: 80% sampled randomly from full df
    df_train = df.sample(frac=0.8, random_state=42)
    
    # Remaining 20%
    df_remaining = df.drop(df_train.index, errors='ignore')
    
    # Validation set: 50% of remaining (i.e. 10% of full df)
    df_val = df_remaining.sample(frac=0.5, random_state=42)
    
    # Test set: rest of remaining (i.e. other 10%)
    df_test = df_remaining.drop(df_val.index, errors='ignore')

    # Sanity checks (optional)
    assert len(df_train) + len(df_val) + len(df_test) == len(df)
    assert not set(df_train.index) & set(df_val.index)
    assert not set(df_train.index) & set(df_test.index)
    assert not set(df_val.index) & set(df_test.index)

    os.makedirs('data/mp_20', exist_ok=True)
    
    df_train.to_pickle('data/mp_20/train.pkl')
    df_val.to_pickle('data/mp_20/val.pkl')
    df_test.to_pickle('data/mp_20/test.pkl')
