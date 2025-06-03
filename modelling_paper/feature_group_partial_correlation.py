import pandas as pd
import numpy as np
from pingouin import partial_corr

# Load original features
print("Loading original features...")
original_features = pd.read_csv("/Users/davidwhyatt/Documents/GitHub/PhDMelodySet/modelling_paper/testing.csv")
print("Loading odd one out features...")
odd_one_out_features = pd.read_csv("/Users/davidwhyatt/Documents/GitHub/PhDMelodySet/modelling_paper/miq_mels2.csv")
print("Loading participant responses...")
participant_responses = pd.read_csv("/Users/davidwhyatt/Downloads/miq_trials.csv", nrows=int(1e6))
print("Loading item bank...")
item_bank = pd.read_csv("/Users/davidwhyatt/Documents/GitHub/PhDMelodySet/modelling_paper/item-bank.csv")
item_bank = item_bank.rename(columns={'id': 'item_id'})

# Filter participant_responses to only include 'mdt' test
participant_responses = participant_responses[participant_responses['test'] == 'mdt']

# Drop duration features from original and odd-one-out features
duration_cols_orig = [col for col in original_features.columns if 'duration_features.' in col]
duration_cols_ooo = [col for col in odd_one_out_features.columns if 'duration_features.' in col]
print("Dropping duration features from original:", duration_cols_orig)
print("Dropping duration features from odd-one-out:", duration_cols_ooo)
original_features = original_features.drop(columns=duration_cols_orig)
odd_one_out_features = odd_one_out_features.drop(columns=duration_cols_ooo)

# Merge features on melody_id
features_final = original_features.merge(odd_one_out_features, on='melody_id', how='inner')

# Get IRT features
irt_cols = ['item_id', 'ability_WL', 'difficulty', 'score']
irt_features = participant_responses[irt_cols].copy()
irt_features = irt_features.groupby('item_id').agg({
    'ability_WL': 'mean',
    'difficulty': 'mean', 
    'score': 'first'
}).reset_index()

# Merge with existing features
data_with_irt = features_final.merge(irt_features, left_on='melody_id', right_on='item_id', how='left')

# Define feature groups (adjust as needed)
feature_groups = {
    'contour': [col for col in data_with_irt.columns if 'contour' in col.lower()],
    'corpus': [col for col in data_with_irt.columns if 'corpus' in col.lower()],
    'duration': [col for col in data_with_irt.columns if 'duration' in col.lower()],
    'interval': [col for col in data_with_irt.columns if 'interval' in col.lower()],
    'melodic_movement': [col for col in data_with_irt.columns if 'melodic_movement' in col.lower()],
    'mtype': [col for col in data_with_irt.columns if 'mtype' in col.lower()],
    'narmour': [col for col in data_with_irt.columns if 'narmour' in col.lower()],
    'pitch': [col for col in data_with_irt.columns if 'pitch' in col.lower()],
    'tonality': [col for col in data_with_irt.columns if 'tonality' in col.lower() or 'key' in col.lower()],
    'ability_WL': [col for col in data_with_irt.columns if col == 'ability_WL'],
}

# Target variable
y = data_with_irt['score']

# Prepare a DataFrame for all group summaries (mean of group)
group_means = {}
for group, cols in feature_groups.items():
    # Only use numeric columns
    numeric_cols = [col for col in cols if pd.api.types.is_numeric_dtype(data_with_irt[col])]
    if numeric_cols:
        group_means[group] = data_with_irt[numeric_cols].mean(axis=1)
group_means = pd.DataFrame(group_means)
group_means['score'] = y

print('Partial correlations (controlling for all other groups):')
for group in feature_groups:
    if not feature_groups[group]:
        continue
    covariates = [g for g in group_means.columns if g not in [group, 'score']]
    result = partial_corr(data=group_means, x=group, y='score', covar=covariates, method='pearson')
    print(f"{group}: r = {result['r'].values[0]:.4f}, p = {result['p-val'].values[0]:.4g}") 