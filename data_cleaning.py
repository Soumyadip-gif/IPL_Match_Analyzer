import pandas as pd
from pathlib import Path


# ============================================================
# 1. LOAD DATA
# ============================================================

MATCHES_FILE = "data/ipl_matches_clean.csv"
DELIVERIES_FILE = "data/ipl_deliveries_clean.csv"

matches = pd.read_csv(MATCHES_FILE)
deliveries = pd.read_csv(DELIVERIES_FILE)

print("=" * 60)
print("IPL DATA CLEANING")
print("=" * 60)


# ============================================================
# 2. REMOVE EXACT DUPLICATES
# ============================================================

matches = matches.drop_duplicates().copy()
deliveries = deliveries.drop_duplicates().copy()

print("\nDuplicates removed.")


# ============================================================
# 3. CLEAN MATCH DATA TYPES
# ============================================================

# Convert date to datetime
matches["date"] = pd.to_datetime(matches["date"], errors="coerce")

# Convert numeric columns
matches["match_id"] = pd.to_numeric(matches["match_id"], errors="coerce")
matches["season"] = pd.to_numeric(matches["season"], errors="coerce")
matches["result_margin"] = pd.to_numeric(
    matches["result_margin"], errors="coerce"
)


# ============================================================
# 4. CLEAN TEXT COLUMNS
# ============================================================

match_text_columns = [
    "venue",
    "city",
    "team1",
    "team2",
    "toss_winner",
    "toss_decision",
    "winner",
    "result_type",
    "player_of_match"
]

for column in match_text_columns:
    matches[column] = matches[column].astype("string").str.strip()


delivery_text_columns = [
    "batting_team",
    "batter",
    "bowler",
    "non_striker",
    "wicket_player_out",
    "wicket_kind"
]

for column in delivery_text_columns:
    deliveries[column] = deliveries[column].astype("string").str.strip()


# ============================================================
# 5. HANDLE CITY MISSING VALUES
# ============================================================

# Some matches do not have city information.
# We keep them as "Unknown" instead of deleting those matches.

matches["city"] = matches["city"].fillna("Unknown")


# ============================================================
# 6. HANDLE PLAYER-OF-THE-MATCH MISSING VALUES
# ============================================================

# No player was recorded for some matches.
# Keep the match and mark the value as Unknown.

matches["player_of_match"] = matches["player_of_match"].fillna("Unknown")


# ============================================================
# 7. HANDLE WINNER VALUES
# ============================================================

# Do NOT fill missing winners with a fake team.
# Missing winners are important for ties/no-results.

matches["winner"] = matches["winner"].replace(
    ["<NA>", "nan", "None", ""],
    pd.NA
)

# Verify missing winners
print("\nMissing winners after cleaning:",
      matches["winner"].isna().sum())


# ============================================================
# 8. CLEAN DELIVERY NUMERIC COLUMNS
# ============================================================

delivery_numeric_columns = [
    "match_id",
    "innings",
    "over",
    "ball",
    "runs_batter",
    "runs_extras",
    "runs_total",
    "wicket"
]

for column in delivery_numeric_columns:
    deliveries[column] = pd.to_numeric(
        deliveries[column], errors="coerce"
    )


# ============================================================
# 9. VALIDATE RUN VALUES
# ============================================================

# Remove impossible negative run values if any appear.
deliveries = deliveries[
    (deliveries["runs_batter"] >= 0) &
    (deliveries["runs_extras"] >= 0) &
    (deliveries["runs_total"] >= 0)
].copy()


# ============================================================
# 10. VALIDATE WICKET VALUES
# ============================================================

deliveries = deliveries[
    deliveries["wicket"].isin([0, 1])
].copy()


# ============================================================
# 11. REMOVE ROWS WITH CRITICAL MISSING VALUES
# ============================================================

# Match ID is required to connect deliveries with matches.

deliveries = deliveries.dropna(
    subset=[
        "match_id",
        "innings",
        "batting_team",
        "over",
        "ball",
        "batter",
        "bowler",
        "runs_batter",
        "runs_extras",
        "runs_total",
        "wicket"
    ]
).copy()


# ============================================================
# 12. SORT DATA
# ============================================================

matches = matches.sort_values(
    by=["date", "match_id"]
).reset_index(drop=True)

deliveries = deliveries.sort_values(
    by=["match_id", "innings", "over", "ball"]
).reset_index(drop=True)


# ============================================================
# 13. CREATE CLEAN DATA FOLDER
# ============================================================

clean_folder = Path("data/clean")
clean_folder.mkdir(parents=True, exist_ok=True)


# ============================================================
# 14. SAVE CLEANED DATA
# ============================================================

matches.to_csv(
    "data/clean/matches.csv",
    index=False
)

deliveries.to_csv(
    "data/clean/deliveries.csv",
    index=False
)


# ============================================================
# 15. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("CLEANING COMPLETED")
print("=" * 60)

print("\nMatches:")
print("Rows:", len(matches))
print("Columns:", len(matches.columns))

print("\nDeliveries:")
print("Rows:", len(deliveries))
print("Columns:", len(deliveries.columns))

print("\nCleaned files created:")
print("data/clean/matches.csv")
print("data/clean/deliveries.csv")

print("\n" + "=" * 60)
print("STEP 6 COMPLETED")
print("=" * 60)