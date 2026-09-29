import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# ============================================================
# 1. LOAD CLEAN DATA
# ============================================================

matches = pd.read_csv("data/clean/matches.csv")
deliveries = pd.read_csv("data/clean/deliveries.csv")

matches["date"] = pd.to_datetime(matches["date"])

print("=" * 60)
print("IPL MATCH ANALYZER - EDA")
print("=" * 60)


# ============================================================
# 2. CREATE EDA OUTPUT FOLDER
# ============================================================

Path("eda_output").mkdir(exist_ok=True)


# ============================================================
# 3. BASIC INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("1. BASIC DATA INFORMATION")
print("=" * 60)

print("Total matches:", len(matches))
print("Total deliveries:", len(deliveries))
print("Total seasons:", matches["season"].nunique())
print("Total teams:", len(
    set(matches["team1"]) | set(matches["team2"])
))
print("Total venues:", matches["venue"].nunique())


# ============================================================
# 4. SEASON-WISE MATCHES
# ============================================================

season_matches = (
    matches.groupby("season")
    .size()
    .sort_index()
)

print("\n" + "=" * 60)
print("2. MATCHES BY SEASON")
print("=" * 60)

print(season_matches)


plt.figure(figsize=(12, 6))

season_matches.plot(
    kind="bar"
)

plt.title("IPL Matches by Season")
plt.xlabel("Season")
plt.ylabel("Number of Matches")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    "eda_output/matches_by_season.png",
    dpi=150
)

plt.close()


# ============================================================
# 5. TEAM WINS
# ============================================================

team_wins = (
    matches["winner"]
    .dropna()
    .value_counts()
)

print("\n" + "=" * 60)
print("3. TEAM WINS")
print("=" * 60)

print(team_wins)


plt.figure(figsize=(12, 7))

team_wins.sort_values().plot(
    kind="barh"
)

plt.title("IPL Team Wins")
plt.xlabel("Number of Wins")
plt.ylabel("Team")
plt.tight_layout()

plt.savefig(
    "eda_output/team_wins.png",
    dpi=150
)

plt.close()


# ============================================================
# 6. TOSS DECISION ANALYSIS
# ============================================================

toss_decisions = matches["toss_decision"].value_counts()

print("\n" + "=" * 60)
print("4. TOSS DECISIONS")
print("=" * 60)

print(toss_decisions)


plt.figure(figsize=(7, 5))

toss_decisions.plot(
    kind="bar"
)

plt.title("Toss Decision Distribution")
plt.xlabel("Decision")
plt.ylabel("Number of Matches")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    "eda_output/toss_decisions.png",
    dpi=150
)

plt.close()


# ============================================================
# 7. TOSS WINNER vs MATCH WINNER
# ============================================================

matches["toss_winner_won_match"] = (
    matches["toss_winner"] == matches["winner"]
)

toss_match_result = (
    matches["toss_winner_won_match"]
    .value_counts()
)

print("\n" + "=" * 60)
print("5. TOSS WINNER vs MATCH WINNER")
print("=" * 60)

print(toss_match_result)


# ============================================================
# 8. TOP BATTERS BY RUNS
# ============================================================

top_batters = (
    deliveries.groupby("batter")["runs_batter"]
    .sum()
    .sort_values(ascending=False)
    .head(15)
)

print("\n" + "=" * 60)
print("6. TOP 15 BATTERS BY RUNS")
print("=" * 60)

print(top_batters)


plt.figure(figsize=(12, 7))

top_batters.sort_values().plot(
    kind="barh"
)

plt.title("Top 15 IPL Run Scorers")
plt.xlabel("Runs")
plt.ylabel("Batter")
plt.tight_layout()

plt.savefig(
    "eda_output/top_batters.png",
    dpi=150
)

plt.close()


# ============================================================
# 9. TOP BOWLERS BY WICKETS
# ============================================================

top_bowlers = (
    deliveries[deliveries["wicket"] == 1]
    .groupby("bowler")
    .size()
    .sort_values(ascending=False)
    .head(15)
)

print("\n" + "=" * 60)
print("7. TOP 15 BOWLERS BY WICKETS")
print("=" * 60)

print(top_bowlers)


plt.figure(figsize=(12, 7))

top_bowlers.sort_values().plot(
    kind="barh"
)

plt.title("Top 15 IPL Wicket Takers")
plt.xlabel("Wickets")
plt.ylabel("Bowler")
plt.tight_layout()

plt.savefig(
    "eda_output/top_bowlers.png",
    dpi=150
)

plt.close()


# ============================================================
# 10. VENUE ANALYSIS
# ============================================================

venue_matches = (
    matches["venue"]
    .value_counts()
    .head(15)
)

print("\n" + "=" * 60)
print("8. TOP 15 VENUES BY MATCHES")
print("=" * 60)

print(venue_matches)


plt.figure(figsize=(12, 7))

venue_matches.sort_values().plot(
    kind="barh"
)

plt.title("Top 15 IPL Venues")
plt.xlabel("Number of Matches")
plt.ylabel("Venue")
plt.tight_layout()

plt.savefig(
    "eda_output/top_venues.png",
    dpi=150
)

plt.close()


# ============================================================
# 11. TOTAL RUNS BY SEASON
# ============================================================

season_runs = (
    deliveries.groupby(
        deliveries["match_id"].map(
            matches.set_index("match_id")["season"]
        )
    )["runs_total"]
    .sum()
)

season_runs.index.name = "season"

print("\n" + "=" * 60)
print("9. TOTAL RUNS BY SEASON")
print("=" * 60)

print(season_runs)


plt.figure(figsize=(12, 6))

season_runs.sort_index().plot(
    kind="line",
    marker="o"
)

plt.title("Total IPL Runs by Season")
plt.xlabel("Season")
plt.ylabel("Total Runs")
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    "eda_output/runs_by_season.png",
    dpi=150
)

plt.close()


# ============================================================
# 12. MATCH RESULT TYPES
# ============================================================

result_types = matches["result_type"].value_counts()

print("\n" + "=" * 60)
print("10. MATCH RESULT TYPES")
print("=" * 60)

print(result_types)


plt.figure(figsize=(7, 5))

result_types.plot(
    kind="bar"
)

plt.title("IPL Match Result Types")
plt.xlabel("Result Type")
plt.ylabel("Matches")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    "eda_output/result_types.png",
    dpi=150
)

plt.close()


# ============================================================
# 13. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("EDA COMPLETED")
print("=" * 60)

print("\nCharts saved inside:")
print("eda_output/")

print("\nGenerated files:")

for file in sorted(Path("eda_output").glob("*.png")):
    print("-", file.name)

print("\n" + "=" * 60)