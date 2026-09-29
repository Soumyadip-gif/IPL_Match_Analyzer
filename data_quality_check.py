import pandas as pd

matches = pd.read_csv("data/ipl_matches_clean.csv")
deliveries = pd.read_csv("data/ipl_deliveries_clean.csv")

print("=" * 60)
print("1. DATASET SHAPE")
print("=" * 60)

print("Matches:", matches.shape)
print("Deliveries:", deliveries.shape)


print("\n" + "=" * 60)
print("2. MISSING VALUES")
print("=" * 60)

print("\nMatches:")
print(matches.isnull().sum())

print("\nDeliveries:")
print(deliveries.isnull().sum())


print("\n" + "=" * 60)
print("3. DUPLICATES")
print("=" * 60)

print("Duplicate matches:", matches.duplicated().sum())
print("Duplicate deliveries:", deliveries.duplicated().sum())


print("\n" + "=" * 60)
print("4. SEASONS")
print("=" * 60)

print("Number of seasons:", matches["season"].nunique())
print("Seasons:", sorted(matches["season"].unique()))


print("\n" + "=" * 60)
print("5. TEAMS")
print("=" * 60)

teams = sorted(
    set(matches["team1"].dropna())
    | set(matches["team2"].dropna())
)

print("Number of teams:", len(teams))
print("Teams:")

for team in teams:
    print("-", team)


print("\n" + "=" * 60)
print("6. VENUES")
print("=" * 60)

print("Number of venues:", matches["venue"].nunique())


print("\n" + "=" * 60)
print("7. MATCH RESULTS")
print("=" * 60)

print(matches["result_type"].value_counts(dropna=False))

print("\nWinner missing:", matches["winner"].isnull().sum())


print("\n" + "=" * 60)
print("8. DELIVERY VALUE CHECK")
print("=" * 60)

print("Negative batter runs:",
      (deliveries["runs_batter"] < 0).sum())

print("Negative extras:",
      (deliveries["runs_extras"] < 0).sum())

print("Negative total runs:",
      (deliveries["runs_total"] < 0).sum())

print("Invalid wickets:",
      (~deliveries["wicket"].isin([0, 1])).sum())


print("\n" + "=" * 60)
print("9. MATCH ID CONSISTENCY")
print("=" * 60)

match_ids = set(matches["match_id"])
delivery_match_ids = set(deliveries["match_id"])

print("Matches without delivery data:",
      len(match_ids - delivery_match_ids))

print("Delivery match IDs without match data:",
      len(delivery_match_ids - match_ids))


print("\n" + "=" * 60)
print("QUALITY CHECK COMPLETED")
print("=" * 60)