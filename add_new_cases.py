"""
Add new design infringement cases to the gold standard dataset
Formats new cases and appends them to the corrected CSV
"""

import pandas as pd
from datetime import datetime

print("="*80)
print("ADD NEW DESIGN INFRINGEMENT CASES")
print("="*80)

# Load existing gold standard
existing_df = pd.read_csv('gold-standard-cases-CORRECTED.csv')
existing_df = existing_df.dropna(how='all')

print(f"\nCurrent dataset: {len(existing_df)} cases")
print(f"  - knockoff: {sum(existing_df['infringement_label'] == 'knockoff')}")
print(f"  - similar: {sum(existing_df['infringement_label'] == 'similar')}")

# Template for new cases
# You can modify this list to add as many cases as you want
new_cases = [
    # EXAMPLE - Delete this and add your real cases
    {
        'case_id': 'NEW_001',
        'original_designer_name': 'Designer Name',
        'original_brand_name': 'Brand Name',
        'original_item_type': 'Apparel',
        'original_design_elements': 'Describe the key design elements here - colors, patterns, shapes, unique features',
        'original_year': 2020,
        'copier_brand_name': 'Fast Fashion Brand',
        'copier_item_type': 'Apparel',
        'copy_year': 2021,
        'infringement_label': 'knockoff',  # or 'similar'
        'confidence': 'high',  # or 'medium'
        'source': 'Diet Prada',  # or 'Press', 'Legal filing', etc.
        'notes': 'Additional context about the case'
    },
    # ADD MORE CASES HERE - Copy the template above
]

print(f"\n{'='*80}")
print("INSTRUCTIONS FOR ADDING CASES")
print("="*80)
print("""
1. Edit this script (add_new_cases.py)
2. Delete the EXAMPLE case above
3. Copy the template for each new case you want to add
4. Fill in all fields with accurate information
5. Run this script to append to the gold standard CSV
6. Then run stabilize-clip-embeddings.py to regenerate embeddings
7. Finally run create_clean_labeled_splits.py to create new train/val/test splits

Fields explanation:
- case_id: Unique ID (e.g., NEW_001, NEW_002, ...)
- original_designer_name: Name of the designer who was copied
- original_brand_name: Brand of original design
- original_item_type: Type (Apparel, Sneakers, Jewelry, Accessories, etc.)
- original_design_elements: DETAILED description - this is what CLIP encodes!
- original_year: Year original was created
- copier_brand_name: Who made the copy
- copier_item_type: Type of copy (usually same as original)
- copy_year: Year copy appeared
- infringement_label: 'knockoff' (near-identical) or 'similar' (inspired)
- confidence: 'high' (undeniable) or 'medium' (noticeable but some differences)
- source: 'Diet Prada', 'Press', 'Legal filing', etc.
- notes: Any additional context
""")

# Validate new cases
print(f"\n{'='*80}")
print("VALIDATING NEW CASES")
print("="*80)

if len(new_cases) == 1 and new_cases[0]['case_id'] == 'NEW_001':
    print("\n⚠️  WARNING: You still have the EXAMPLE case!")
    print("   Please edit this script and add your real cases.")
    print("   Delete the example and copy the template for each new case.\n")
    exit(0)

valid_cases = []
errors = []

for i, case in enumerate(new_cases):
    case_num = i + 1

    # Check required fields
    required = ['case_id', 'original_designer_name', 'original_brand_name',
                'original_design_elements', 'infringement_label', 'confidence']

    missing = [f for f in required if not case.get(f)]
    if missing:
        errors.append(f"Case {case_num}: Missing fields {missing}")
        continue

    # Validate label
    if case['infringement_label'] not in ['knockoff', 'similar']:
        errors.append(f"Case {case_num}: infringement_label must be 'knockoff' or 'similar'")
        continue

    # Validate confidence
    if case['confidence'] not in ['high', 'medium']:
        errors.append(f"Case {case_num}: confidence must be 'high' or 'medium'")
        continue

    # Check for duplicate case_id
    if case['case_id'] in existing_df['case_id'].values:
        errors.append(f"Case {case_num}: case_id '{case['case_id']}' already exists")
        continue

    valid_cases.append(case)

# Report validation results
if errors:
    print("\n❌ VALIDATION ERRORS:")
    for error in errors:
        print(f"   - {error}")
    print(f"\n   {len(valid_cases)} valid cases, {len(errors)} errors")
    print("   Please fix errors and run again.")
    exit(1)

print(f"✓ All {len(valid_cases)} cases validated successfully!")

# Show summary
print(f"\n{'='*80}")
print("NEW CASES SUMMARY")
print("="*80)

new_df = pd.DataFrame(valid_cases)
print(f"\nAdding {len(new_df)} new cases:")
print(f"  - knockoff: {sum(new_df['infringement_label'] == 'knockoff')}")
print(f"  - similar: {sum(new_df['infringement_label'] == 'similar')}")
print(f"\nSources:")
for source, count in new_df['source'].value_counts().items():
    print(f"  - {source}: {count}")

# Combine with existing
combined_df = pd.concat([existing_df, new_df], ignore_index=True)

print(f"\n{'='*80}")
print("FINAL DATASET")
print("="*80)
print(f"\nTotal cases: {len(combined_df)}")
print(f"  - knockoff: {sum(combined_df['infringement_label'] == 'knockoff')}")
print(f"  - similar: {sum(combined_df['infringement_label'] == 'similar')}")

# Save with backup
backup_file = f'gold-standard-cases-CORRECTED-backup-{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
existing_df.to_csv(backup_file, index=False)
print(f"\n✓ Created backup: {backup_file}")

# Save updated dataset
combined_df.to_csv('gold-standard-cases-CORRECTED.csv', index=False)
print(f"✓ Updated: gold-standard-cases-CORRECTED.csv")

print(f"\n{'='*80}")
print("NEXT STEPS")
print("="*80)
print("""
1. ✓ New cases added to gold-standard-cases-CORRECTED.csv

2. Run: python3 stabilize-clip-embeddings.py
   This will regenerate embeddings with the new cases

3. Run: python3 create_clean_labeled_splits.py
   This will create new train/val/test splits with more data

Your dataset will be larger and better for training!
""")
