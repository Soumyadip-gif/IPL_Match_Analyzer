import pandas as pd
import numpy as np
from collections import defaultdict, deque


class IPLPredictionEngine:

    def __init__(self, matches_path, deliveries_path):

        self.matches = pd.read_csv(matches_path)
        self.deliveries = pd.read_csv(deliveries_path)

        self.matches["date"] = pd.to_datetime(
            self.matches["date"],
            errors="coerce"
        )

        self.matches = self.matches.sort_values(
            ["date", "match_id"]
        ).reset_index(drop=True)

        numeric_columns = [
            "season",
            "result_margin"
        ]

        for col in numeric_columns:
            if col in self.matches.columns:
                self.matches[col] = pd.to_numeric(
                    self.matches[col],
                    errors="coerce"
                )

        self.match_stats = self._calculate_match_stats()

        # Prepare innings-level data for score prediction
        self.score_match_data = self._prepare_score_match_data()

    # ========================================================
    # MATCH STATISTICS
    # ========================================================

    def _calculate_match_stats(self):

        d = self.deliveries.copy()

        runs = (
            d.groupby(
                ["match_id", "batting_team"]
            )["runs_total"]
            .sum()
            .reset_index()
        )

        runs = runs.rename(
            columns={"runs_total": "team_runs"}
        )

        wickets = (
            d[d["wicket"] == 1]
            .groupby(
                ["match_id", "batting_team"]
            )
            .size()
            .reset_index(name="wickets_taken")
        )

        stats = runs.merge(
            wickets,
            on=["match_id", "batting_team"],
            how="left"
        )

        stats["wickets_taken"] = (
            stats["wickets_taken"]
            .fillna(0)
        )

        return stats

    # ========================================================
    # SCORE MODEL - PREPARE INNINGS DATA
    # ========================================================

    def _prepare_score_match_data(self):

        """
        Prepare innings-level information for
        first innings score prediction.

        This follows the same historical logic used
        while training the score model.
        """

        d = self.deliveries.copy()

        d["runs_total"] = pd.to_numeric(
            d["runs_total"],
            errors="coerce"
        ).fillna(0)

        d["wicket"] = pd.to_numeric(
            d["wicket"],
            errors="coerce"
        ).fillna(0)

        d["innings"] = pd.to_numeric(
            d["innings"],
            errors="coerce"
        )

        match_data = {}

        for match_id, group in d.groupby("match_id"):

            innings_numbers = sorted(
                group["innings"]
                .dropna()
                .unique()
            )

            if len(innings_numbers) == 0:
                continue

            innings_info = []

            for innings_number in innings_numbers:

                innings_group = group[
                    group["innings"] == innings_number
                ]

                if innings_group.empty:
                    continue

                batting_team = (
                    innings_group["batting_team"]
                    .iloc[0]
                )

                score = float(
                    innings_group["runs_total"].sum()
                )

                wickets_lost = int(
                    innings_group["wicket"].sum()
                )

                innings_info.append({
                    "team": batting_team,
                    "score": score,
                    "wickets_lost": wickets_lost
                })

            if innings_info:

                match_data[match_id] = innings_info

        return match_data

    # ========================================================
    # TEAM HISTORY
    # ========================================================

    def _team_history(self, team, historical_matches):

        team_matches = historical_matches[
            (historical_matches["team1"] == team)
            |
            (historical_matches["team2"] == team)
        ].copy()

        if len(team_matches) == 0:

            return {
                "win_rate": 0.5,
                "form_5": 0.5,
                "form_10": 0.5,
                "recent_runs": 0.0,
                "recent_wickets": 0.0,
                "run_rate": 0.0,
                "wickets_per_match": 0.0
            }

        wins = (
            team_matches["winner"] == team
        ).sum()

        completed = team_matches[
            team_matches["winner"].notna()
        ]

        if len(completed) > 0:
            win_rate = float(
                wins / len(completed)
            )
        else:
            win_rate = 0.5

        recent = team_matches.tail(10)

        form_values = []

        for _, row in recent.iterrows():

            if row["winner"] == team:
                form_values.append(1.0)

            elif pd.isna(row["winner"]):
                form_values.append(0.5)

            else:
                form_values.append(0.0)

        if form_values:

            form_10 = float(
                np.mean(form_values)
            )

            form_5 = float(
                np.mean(form_values[-5:])
            )

        else:

            form_5 = 0.5
            form_10 = 0.5

        match_ids = team_matches[
            "match_id"
        ].tolist()

        team_stats = self.match_stats[
            (
                self.match_stats["match_id"]
                .isin(match_ids)
            )
            &
            (
                self.match_stats["batting_team"]
                == team
            )
        ].copy()

        if len(team_stats) > 0:

            recent_stats = team_stats.tail(10)

            recent_runs = float(
                recent_stats["team_runs"].sum()
            )

            recent_wickets = float(
                recent_stats["wickets_taken"].sum()
            )

            total_runs = float(
                team_stats["team_runs"].sum()
            )

            total_wickets = float(
                team_stats["wickets_taken"].sum()
            )

            run_rate = float(
                total_runs / len(team_stats)
            )

            wickets_per_match = float(
                total_wickets / len(team_stats)
            )

        else:

            recent_runs = 0.0
            recent_wickets = 0.0
            run_rate = 0.0
            wickets_per_match = 0.0

        return {
            "win_rate": float(win_rate),
            "form_5": float(form_5),
            "form_10": float(form_10),
            "recent_runs": float(recent_runs),
            "recent_wickets": float(recent_wickets),
            "run_rate": float(run_rate),
            "wickets_per_match": float(
                wickets_per_match
            )
        }

    # ========================================================
    # ELO RATING
    # ========================================================

    def _calculate_elo(self, historical_matches):

        elo = {}

        K = 20
        BASE = 1500

        for _, match in historical_matches.iterrows():

            team1 = match["team1"]
            team2 = match["team2"]
            winner = match["winner"]

            if team1 not in elo:
                elo[team1] = BASE

            if team2 not in elo:
                elo[team2] = BASE

            if pd.isna(winner):
                continue

            rating1 = elo[team1]
            rating2 = elo[team2]

            expected1 = (
                1 /
                (
                    1 +
                    10 ** (
                        (rating2 - rating1) / 400
                    )
                )
            )

            actual1 = 1 if winner == team1 else 0

            elo[team1] += K * (
                actual1 - expected1
            )

            elo[team2] += K * (
                (1 - actual1)
                -
                (1 - expected1)
            )

        return elo

    # ========================================================
    # VENUE WIN RATE
    # ========================================================

    def _venue_rate(
        self,
        team,
        venue,
        historical_matches
    ):

        venue_matches = historical_matches[
            (
                (
                    historical_matches["team1"]
                    == team
                )
                |
                (
                    historical_matches["team2"]
                    == team
                )
            )
            &
            (
                historical_matches["venue"]
                == venue
            )
        ]

        completed = venue_matches[
            venue_matches["winner"].notna()
        ]

        if len(completed) == 0:
            return 0.5

        wins = (
            completed["winner"] == team
        ).sum()

        return float(
            wins / len(completed)
        )

    # ========================================================
    # HEAD TO HEAD
    # ========================================================

    def _h2h_rate(
        self,
        team1,
        team2,
        historical_matches
    ):

        h2h = historical_matches[
            (
                (
                    historical_matches["team1"]
                    == team1
                )
                &
                (
                    historical_matches["team2"]
                    == team2
                )
            )
            |
            (
                (
                    historical_matches["team1"]
                    == team2
                )
                &
                (
                    historical_matches["team2"]
                    == team1
                )
            )
        ]

        completed = h2h[
            h2h["winner"].notna()
        ]

        if len(completed) == 0:
            return 0.5

        wins = (
            completed["winner"] == team1
        ).sum()

        return float(
            wins / len(completed)
        )

    # ========================================================
    # HISTORICAL DATA FILTER
    # ========================================================

    def _get_historical_matches(
        self,
        season,
        match_date=None
    ):

        """
        Return only matches known before
        the prediction time.
        """

        season = int(season)

        if match_date:

            prediction_date = pd.to_datetime(
                match_date,
                errors="coerce"
            )

            if pd.isna(prediction_date):

                raise ValueError(
                    "Invalid match_date. "
                    "Use YYYY-MM-DD."
                )

            historical_matches = (
                self.matches[
                    self.matches["date"]
                    < prediction_date
                ]
                .copy()
            )

        else:

            historical_matches = (
                self.matches[
                    self.matches["season"]
                    < season
                ]
                .copy()
            )

        return (
            historical_matches
            .sort_values(
                ["date", "match_id"]
            )
            .reset_index(drop=True)
        )

    # ========================================================
    # SCORE MODEL - FEATURE GENERATOR
    # ========================================================

    def generate_score_features(
        self,
        team1,
        team2,
        venue,
        season,
        toss_winner=None,
        toss_decision=None,
        match_date=None
    ):

        """
        Generate the exact 20 features used
        by train_score_model.py.

        Only matches before the prediction date
        are used.
        """

        # ----------------------------------------------------
        # Historical matches
        # ----------------------------------------------------

        historical_matches = (
            self._get_historical_matches(
                season=season,
                match_date=match_date
            )
        )

        # ----------------------------------------------------
        # Determine batting team
        # ----------------------------------------------------

        if (
            toss_winner == team1
            and toss_decision == "bat"
        ):

            batting_team = team1

        elif (
            toss_winner == team2
            and toss_decision == "bat"
        ):

            batting_team = team2

        elif (
            toss_winner == team1
            and toss_decision == "field"
        ):

            batting_team = team2

        elif (
            toss_winner == team2
            and toss_decision == "field"
        ):

            batting_team = team1

        else:

            batting_team = team1

        # ----------------------------------------------------
        # Bowling team
        # ----------------------------------------------------

        if batting_team == team1:
            bowling_team = team2
        else:
            bowling_team = team1

        # ----------------------------------------------------
        # Historical structures
        # ----------------------------------------------------

        team_scores = defaultdict(
            lambda: deque(maxlen=10)
        )

        team_wickets = defaultdict(
            lambda: deque(maxlen=10)
        )

        team_matches = defaultdict(int)

        venue_scores = defaultdict(
            lambda: deque(maxlen=20)
        )

        h2h_scores = defaultdict(
            lambda: deque(maxlen=10)
        )

        # ----------------------------------------------------
        # Process historical matches
        # ----------------------------------------------------

        for _, historical_match in (
            historical_matches.iterrows()
        ):

            historical_match_id = (
                historical_match["match_id"]
            )

            if (
                historical_match_id
                not in self.score_match_data
            ):
                continue

            innings_info = (
                self.score_match_data[
                    historical_match_id
                ]
            )

            if len(innings_info) == 0:
                continue

            historical_team1 = (
                historical_match["team1"]
            )

            historical_team2 = (
                historical_match["team2"]
            )

            historical_venue = (
                historical_match["venue"]
            )

            # ------------------------------------------------
            # FIRST INNINGS
            # ------------------------------------------------

            first_innings = innings_info[0]

            first_batting_team = (
                first_innings["team"]
            )

            first_score = float(
                first_innings["score"]
            )

            first_wickets_lost = int(
                first_innings["wickets_lost"]
            )

            if (
                first_batting_team
                == historical_team1
            ):

                first_bowling_team = (
                    historical_team2
                )

            else:

                first_bowling_team = (
                    historical_team1
                )

            # Team batting score
            team_scores[
                first_batting_team
            ].append(
                first_score
            )

            # Wickets taken by bowling team
            team_wickets[
                first_bowling_team
            ].append(
                first_wickets_lost
            )

            # Venue score
            venue_scores[
                historical_venue
            ].append(
                first_score
            )

            # H2H score
            h2h_scores[
                (
                    first_batting_team,
                    first_bowling_team
                )
            ].append(
                first_score
            )

            # ------------------------------------------------
            # SECOND INNINGS
            # ------------------------------------------------

            if len(innings_info) > 1:

                second_innings = (
                    innings_info[1]
                )

                second_batting_team = (
                    second_innings["team"]
                )

                second_score = float(
                    second_innings["score"]
                )

                second_wickets_lost = int(
                    second_innings[
                        "wickets_lost"
                    ]
                )

                # Add chasing team's score
                team_scores[
                    second_batting_team
                ].append(
                    second_score
                )

                # First batting team took
                # second innings wickets
                team_wickets[
                    first_batting_team
                ].append(
                    second_wickets_lost
                )

            # Match count
            team_matches[
                historical_team1
            ] += 1

            team_matches[
                historical_team2
            ] += 1

        # ----------------------------------------------------
        # Helper functions
        # ----------------------------------------------------

        def average(values):

            values = list(values)

            if not values:
                return 0.0

            return float(
                np.mean(values)
            )

        def recent_average(
            values,
            n
        ):

            values = list(values)

            if not values:
                return 0.0

            return float(
                np.mean(
                    values[-n:]
                )
            )

        def recent_sum(
            values,
            n
        ):

            values = list(values)

            if not values:
                return 0.0

            return float(
                np.sum(
                    values[-n:]
                )
            )

        # ----------------------------------------------------
        # Get histories
        # ----------------------------------------------------

        batting_history = team_scores[
            batting_team
        ]

        bowling_history = team_scores[
            bowling_team
        ]

        batting_wickets = team_wickets[
            batting_team
        ]

        bowling_wickets = team_wickets[
            bowling_team
        ]

        venue_history = venue_scores[
            venue
        ]

        h2h_history = h2h_scores[
            (
                batting_team,
                bowling_team
            )
        ]

        # ----------------------------------------------------
        # Toss features
        # ----------------------------------------------------

        toss_winner_batting = int(
            toss_winner == batting_team
        )

        toss_bat_decision = int(
            toss_decision == "bat"
        )

        # ====================================================
        # EXACT SCORE MODEL FEATURES
        # ====================================================

        features = {

            "season": float(
                season
            ),

            "batting_team_matches": float(
                team_matches[
                    batting_team
                ]
            ),

            "bowling_team_matches": float(
                team_matches[
                    bowling_team
                ]
            ),

            "batting_recent_score_5":
                recent_average(
                    batting_history,
                    5
                ),

            "batting_recent_score_10":
                recent_average(
                    batting_history,
                    10
                ),

            "batting_recent_runs_5":
                recent_sum(
                    batting_history,
                    5
                ),

            "batting_average_score":
                average(
                    batting_history
                ),

            "batting_team_wickets_5":
                recent_average(
                    batting_wickets,
                    5
                ),

            "bowling_recent_score_5":
                recent_average(
                    bowling_history,
                    5
                ),

            "bowling_recent_wickets_5":
                recent_average(
                    bowling_wickets,
                    5
                ),

            "bowling_recent_wickets_10":
                recent_average(
                    bowling_wickets,
                    10
                ),

            "bowling_average_wickets":
                average(
                    bowling_wickets
                ),

            "venue_average_score":
                average(
                    venue_history
                ),

            "venue_recent_score":
                recent_average(
                    venue_history,
                    10
                ),

            "venue_matches":
                float(
                    len(venue_history)
                ),

            "h2h_average_score":
                average(
                    h2h_history
                ),

            "h2h_recent_score":
                recent_average(
                    h2h_history,
                    5
                ),

            "h2h_matches":
                float(
                    len(h2h_history)
                ),

            "toss_winner_batting":
                toss_winner_batting,

            "toss_bat_decision":
                toss_bat_decision
        }

        return features

    # ========================================================
    # MATCH ANALYSIS
    # ========================================================

    def get_match_analysis(
        self,
        team1,
        team2,
        venue,
        season,
        match_date=None
    ):

        season = int(season)

        historical_matches = (
            self._get_historical_matches(
                season=season,
                match_date=match_date
            )
        )

        team1_history = self._team_history(
            team1,
            historical_matches
        )

        team2_history = self._team_history(
            team2,
            historical_matches
        )

        elo = self._calculate_elo(
            historical_matches
        )

        team1_elo = float(
            elo.get(team1, 1500)
        )

        team2_elo = float(
            elo.get(team2, 1500)
        )

        team1_venue_rate = float(
            self._venue_rate(
                team1,
                venue,
                historical_matches
            )
        )

        team2_venue_rate = float(
            self._venue_rate(
                team2,
                venue,
                historical_matches
            )
        )

        h2h_matches = historical_matches[
            (
                (
                    historical_matches["team1"]
                    == team1
                )
                &
                (
                    historical_matches["team2"]
                    == team2
                )
            )
            |
            (
                (
                    historical_matches["team1"]
                    == team2
                )
                &
                (
                    historical_matches["team2"]
                    == team1
                )
            )
        ]

        completed_h2h = h2h_matches[
            h2h_matches["winner"].notna()
        ]

        team1_h2h_wins = int(
            (
                completed_h2h["winner"]
                == team1
            ).sum()
        )

        team2_h2h_wins = int(
            (
                completed_h2h["winner"]
                == team2
            ).sum()
        )

        total_h2h = int(
            team1_h2h_wins
            +
            team2_h2h_wins
        )

        if total_h2h > 0:

            team1_h2h_rate = float(
                team1_h2h_wins
                / total_h2h
            )

            team2_h2h_rate = float(
                team2_h2h_wins
                / total_h2h
            )

        else:

            team1_h2h_rate = 0.5
            team2_h2h_rate = 0.5

        return {

            "team1": {

                "name": str(team1),

                "elo": float(
                    round(
                        team1_elo,
                        2
                    )
                ),

                "win_rate": float(
                    round(
                        team1_history[
                            "win_rate"
                        ] * 100,
                        2
                    )
                ),

                "form_5": float(
                    round(
                        team1_history[
                            "form_5"
                        ] * 100,
                        2
                    )
                ),

                "form_10": float(
                    round(
                        team1_history[
                            "form_10"
                        ] * 100,
                        2
                    )
                ),

                "recent_runs": float(
                    round(
                        team1_history[
                            "recent_runs"
                        ],
                        2
                    )
                ),

                "recent_wickets": float(
                    round(
                        team1_history[
                            "recent_wickets"
                        ],
                        2
                    )
                ),

                "run_rate": float(
                    round(
                        team1_history[
                            "run_rate"
                        ],
                        2
                    )
                ),

                "wickets_per_match": float(
                    round(
                        team1_history[
                            "wickets_per_match"
                        ],
                        2
                    )
                ),

                "venue_win_rate": float(
                    round(
                        team1_venue_rate * 100,
                        2
                    )
                ),

                "h2h_wins": int(
                    team1_h2h_wins
                ),

                "h2h_win_rate": float(
                    round(
                        team1_h2h_rate * 100,
                        2
                    )
                )
            },

            "team2": {

                "name": str(team2),

                "elo": float(
                    round(
                        team2_elo,
                        2
                    )
                ),

                "win_rate": float(
                    round(
                        team2_history[
                            "win_rate"
                        ] * 100,
                        2
                    )
                ),

                "form_5": float(
                    round(
                        team2_history[
                            "form_5"
                        ] * 100,
                        2
                    )
                ),

                "form_10": float(
                    round(
                        team2_history[
                            "form_10"
                        ] * 100,
                        2
                    )
                ),

                "recent_runs": float(
                    round(
                        team2_history[
                            "recent_runs"
                        ],
                        2
                    )
                ),

                "recent_wickets": float(
                    round(
                        team2_history[
                            "recent_wickets"
                        ],
                        2
                    )
                ),

                "run_rate": float(
                    round(
                        team2_history[
                            "run_rate"
                        ],
                        2
                    )
                ),

                "wickets_per_match": float(
                    round(
                        team2_history[
                            "wickets_per_match"
                        ],
                        2
                    )
                ),

                "venue_win_rate": float(
                    round(
                        team2_venue_rate * 100,
                        2
                    )
                ),

                "h2h_wins": int(
                    team2_h2h_wins
                ),

                "h2h_win_rate": float(
                    round(
                        team2_h2h_rate * 100,
                        2
                    )
                )
            },

            "head_to_head": {

                "total_matches":
                    int(total_h2h),

                "team1_wins":
                    int(team1_h2h_wins),

                "team2_wins":
                    int(team2_h2h_wins)
            },

            "elo_difference": float(
                round(
                    team1_elo
                    -
                    team2_elo,
                    2
                )
            )
        }

    # ========================================================
    # WIN PREDICTION FEATURE GENERATOR
    # ========================================================

    def generate_features(
        self,
        team1,
        team2,
        venue,
        season,
        toss_winner=None,
        toss_decision=None,
        match_date=None
    ):

        season = int(season)

        historical_matches = (
            self._get_historical_matches(
                season=season,
                match_date=match_date
            )
        )

        team1_history = self._team_history(
            team1,
            historical_matches
        )

        team2_history = self._team_history(
            team2,
            historical_matches
        )

        elo = self._calculate_elo(
            historical_matches
        )

        team1_elo = float(
            elo.get(team1, 1500)
        )

        team2_elo = float(
            elo.get(team2, 1500)
        )

        team1_venue_rate = float(
            self._venue_rate(
                team1,
                venue,
                historical_matches
            )
        )

        team2_venue_rate = float(
            self._venue_rate(
                team2,
                venue,
                historical_matches
            )
        )

        h2h = float(
            self._h2h_rate(
                team1,
                team2,
                historical_matches
            )
        )

        features = {

            "season": int(season),

            "team1_elo": team1_elo,

            "team2_elo": team2_elo,

            "elo_difference":
                float(
                    team1_elo
                    -
                    team2_elo
                ),

            "team1_form_5":
                float(
                    team1_history[
                        "form_5"
                    ]
                ),

            "team2_form_5":
                float(
                    team2_history[
                        "form_5"
                    ]
                ),

            "team1_form_10":
                float(
                    team1_history[
                        "form_10"
                    ]
                ),

            "team2_form_10":
                float(
                    team2_history[
                        "form_10"
                    ]
                ),

            "team1_recent_runs":
                float(
                    team1_history[
                        "recent_runs"
                    ]
                ),

            "team2_recent_runs":
                float(
                    team2_history[
                        "recent_runs"
                    ]
                ),

            "team1_recent_wickets":
                float(
                    team1_history[
                        "recent_wickets"
                    ]
                ),

            "team2_recent_wickets":
                float(
                    team2_history[
                        "recent_wickets"
                    ]
                ),

            "team1_run_rate":
                float(
                    team1_history[
                        "run_rate"
                    ]
                ),

            "team2_run_rate":
                float(
                    team2_history[
                        "run_rate"
                    ]
                ),

            "team1_wickets_per_match":
                float(
                    team1_history[
                        "wickets_per_match"
                    ]
                ),

            "team2_wickets_per_match":
                float(
                    team2_history[
                        "wickets_per_match"
                    ]
                ),

            "team1_venue_rate":
                team1_venue_rate,

            "team2_venue_rate":
                team2_venue_rate,

            "h2h_team1_rate":
                h2h,

            "win_rate_difference":
                float(
                    team1_history[
                        "win_rate"
                    ]
                    -
                    team2_history[
                        "win_rate"
                    ]
                ),

            "form_difference":
                float(
                    team1_history[
                        "form_5"
                    ]
                    -
                    team2_history[
                        "form_5"
                    ]
                ),

            "form_10_difference":
                float(
                    team1_history[
                        "form_10"
                    ]
                    -
                    team2_history[
                        "form_10"
                    ]
                ),

            "batting_difference":
                float(
                    team1_history[
                        "run_rate"
                    ]
                    -
                    team2_history[
                        "run_rate"
                    ]
                ),

            "bowling_difference":
                float(
                    team1_history[
                        "wickets_per_match"
                    ]
                    -
                    team2_history[
                        "wickets_per_match"
                    ]
                ),

            "venue_difference":
                float(
                    team1_venue_rate
                    -
                    team2_venue_rate
                ),

            "recent_runs_difference":
                float(
                    team1_history[
                        "recent_runs"
                    ]
                    -
                    team2_history[
                        "recent_runs"
                    ]
                ),

            "recent_wickets_difference":
                float(
                    team1_history[
                        "recent_wickets"
                    ]
                    -
                    team2_history[
                        "recent_wickets"
                    ]
                )
        }

        features["toss_team1"] = int(
            toss_winner == team1
        )

        features["toss_bat"] = int(
            toss_decision == "bat"
        )

        features[
            f"team1_{team1}"
        ] = 1

        features[
            f"team2_{team2}"
        ] = 1

        features[
            f"venue_{venue}"
        ] = 1

        return features