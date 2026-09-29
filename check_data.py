import pandas as pd

matches = pd.read_csv("data/ipl_matches_clean.csv")
deliveries = pd.read_csv("data/ipl_deliveries_clean.csv")

print("MATCHES DATASET")
print("Shape:", matches.shape)
print(matches.head())
print("\nColumns:")
print(matches.columns.tolist())

print("\n" + "=" * 50)

print("DELIVERIES DATASET")
print("Shape:", deliveries.shape)
print(deliveries.head())
print("\nColumns:")
print(deliveries.columns.tolist())