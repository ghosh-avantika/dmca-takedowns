"""
Import new cases from CSV template
Easier than editing Python code - just fill out the CSV!
"""

import pandas as pd
from datetime import datetime

print("="*80)
print("IMPORT NEW CASES FROM CSV")
print("="*80)

# Load the template CSV with new cases
print("\nLooking for: new_cases_template.csv")

try:
    new_cases_df = pd.read_csv('new_cases_template.csv')
except FileNotFoundError:
    print("\n❌ ERROR: new_cases_template.csv not found!")
    print("\n   Please create the file or edit the existing template.")
    exit(1)

# Remove example rows (ones with incomplete data)
print(f"\nLoaded {len(new_cases_df)} rows from template")

# Filter out empty or example rows
new_cases_df = new_cases_df.dropna(subset=['original_designer_name', 'original_design_elements'])
new_cases_df = new_cases_df[new_cases_df['original_designer_name'] != 'Original Designer Name']

if len(new_cases_df) == 0:
    print("\n⚠️  No valid cases found in template!")
    print("   Please fill out new_cases_template.csv with real data.")
    print("\n   Keep the header row and add your cases below.")
    print("   Make sure to fill in at minimum:")
    print("   - original_designer_name")
    print("   - original_design_elements")
    print("   - infringement_label (knockoff or similar)")
    print("   - confidence (high or medium)")
    exit(0)

print(f"Found {len(new_cases_df)} new cases to import")

# Validate
print(f"\n{'='*80}")
print("VALIDATING CASES")
print("="*80)

errors = []
for idx, row in new_cases_df.iterrows():
    case_id = row['case_id']

    # Check label
    if row['infringement_label'] not in ['knockoff', 'similar']:
        errors.append(f"{case_id}: infringement_label must be 'knockoff' or 'similar'")

    # Check confidence
    if row['confidence'] not in ['high', 'medium']:
        errors.append(f"{case_id}: confidence must be 'high' or 'medium'")

    # Check design elements not empty
    if pd.isna(row['original_design_elements']) or len(str(row['original_design_elements'])) < 10:
        errors.append(f"{case_id}: original_design_elements too short (need detailed description)")

if errors:
    print("\n❌ VALIDATION ERRORS:")
    for error in errors:
        print(f"   - {error}")
    exit(1)

print(f"✓ All {len(new_cases_df)} cases validated!")

# Show summary
print(f"\n{'='*80}")
print("NEW CASES SUMMARY")
print("="*80)
print(f"\nLabel distribution:")
print(f"  - knockoff: {sum(new_cases_df['infringement_label'] == 'knockoff')}")
print(f"  - similar: {sum(new_cases_df['infringement_label'] == 'similar')}")

print(f"\nConfidence distribution:")
print(f"  - high: {sum(new_cases_df['confidence'] == 'high')}")
print(f"  - medium: {sum(new_cases_df['confidence'] == 'medium')}")

if 'source' in new_cases_df.columns:
    print(f"\nSources:")
    for source, count in new_cases_df['source'].value_counts().items():
        print(f"  - {source}: {count}")

# Load existing gold standard
existing_df = pd.read_csv('gold-standard-cases-CORRECTED.csv')
existing_df = existing_df.dropna(how='all')

print(f"\n{'='*80}")
print("CURRENT DATASET")
print("="*80)
print(f"\nExisting cases: {len(existing_df)}")
print(f"  - knockoff: {sum(existing_df['infringement_label'] == 'knockoff')}")
print(f"  - similar: {sum(existing_df['infringement_label'] == 'similar')}")

# Check for duplicate case IDs
duplicate_ids = set(new_cases_df['case_id']) & set(existing_df['case_id'])
if duplicate_ids:
    print(f"\n⚠️  WARNING: Found {len(duplicate_ids)} duplicate case IDs:")
    for dup_id in duplicate_ids:
        print(f"   - {dup_id}")
    print("\n   These will be skipped. Please use unique IDs.")
    new_cases_df = new_cases_df[~new_cases_df['case_id'].isin(duplicate_ids)]
    print(f"\n   Continuing with {len(new_cases_df)} unique cases...")

# Combine
combined_df = pd.concat([existing_df, new_cases_df], ignore_index=True)

print(f"\n{'='*80}")
print("UPDATED DATASET")
print("="*80)
print(f"\nTotal cases: {len(combined_df)}")
print(f"  - knockoff: {sum(combined_df['infringement_label'] == 'knockoff')}")
print(f"  - similar: {sum(combined_df['infringement_label'] == 'similar')}")

# Calculate expected dataset size after pseudo-labeling
# Rough estimate: ~50-60% of scraped cases get pseudo-labeled
estimated_total = len(combined_df) + int((len(combined_df) - 34) * 0.55)
estimated_train = int(estimated_total * 0.70)
estimated_val = int(estimated_total * 0.15)
estimated_test = estimated_total - estimated_train - estimated_val

print(f"\nEstimated final dataset (after pseudo-labeling):")
print(f"  - Total: ~{estimated_total} samples")
print(f"  - Train: ~{estimated_train} samples")
print(f"  - Val: ~{estimated_val} samples")
print(f"  - Test: ~{estimated_test} samples")

# Save with backup
backup_file = f'gold-standard-cases-CORRECTED-backup-{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
existing_df.to_csv(backup_file, index=False)
print(f"\n✓ Created backup: {backup_file}")

# Save updated dataset
combined_df.to_csv('gold-standard-cases-CORRECTED.csv', index=False)
print(f"✓ Updated: gold-standard-cases-CORRECTED.csv")

# Clear the template for next batch
template_df = pd.DataFrame(columns=new_cases_df.columns)
template_df.loc[0] = {
    'case_id': 'NEW_XXX',
    'original_designer_name': 'Original Designer Name',
    'original_brand_name': 'Original Brand',
    'original_item_type': 'Apparel',
    'original_design_elements': 'Describe design: colors, patterns, unique elements, motifs, silhouette',
    'original_year': 2020,
    'copier_brand_name': 'Fast Fashion Brand',
    'copier_item_type': 'Apparel',
    'copy_year': 2021,
    'infringement_label': 'knockoff',
    'confidence': 'high',
    'source': 'Diet Prada',
    'notes': 'Additional context'
}
template_df.to_csv('new_cases_template.csv', index=False)
print(f"✓ Reset template for next batch")

print(f"\n{'='*80}")
print("NEXT STEPS")
print("="*80)
print("""
1. ✓ New cases added to gold-standard-cases-CORRECTED.csv

2. Run: python3 stabilize-clip-embeddings.py
   This will regenerate embeddings including new cases

3. Run: python3 create_clean_labeled_splits.py
   This will create new train/val/test splits with expanded data

Your dataset is now larger and better for MLP training!

To add more cases:
- Fill out new_cases_template.csv again
- Run this script again
- Repeat until you have 100-200+ total cases
""")
