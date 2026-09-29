/* =========================================================
   IPL FUTURE MATCH PREDICTOR
   Frontend API Controller
   ========================================================= */

document.addEventListener("DOMContentLoaded", () => {

    // -----------------------------------------------------
    // FORM ELEMENTS
    // -----------------------------------------------------

    const team1Select = document.getElementById("team1");
    const team2Select = document.getElementById("team2");
    const venueSelect = document.getElementById("venue");
    const matchDateInput = document.getElementById("match-date");
    const tossWinnerSelect = document.getElementById("toss-winner");
    const tossDecisionSelect = document.getElementById("toss-decision");

    const predictBtn = document.getElementById("predict-btn");
    const loading = document.getElementById("loading");
    const errorMessage = document.getElementById("error-message");
    const resultSection = document.getElementById("prediction-result");


    // -----------------------------------------------------
    // WIN PREDICTION
    // -----------------------------------------------------

    const predictedWinner =
        document.getElementById("predicted-winner");

    const team1Probability =
        document.getElementById("team1-probability");

    const team2Probability =
        document.getElementById("team2-probability");

    const team1Progress =
        document.getElementById("team1-progress");

    const team2Progress =
        document.getElementById("team2-progress");

    const modelConfidence =
        document.getElementById("model-confidence");


    // -----------------------------------------------------
    // SCORE PREDICTION
    // -----------------------------------------------------

    const expectedScore =
        document.getElementById("expected-score");

    const scoreRange =
        document.getElementById("score-range");

    const scorePredictionCard =
        document.querySelector(".score-prediction-card");


    // -----------------------------------------------------
    // MATCH REPORT
    // -----------------------------------------------------

    const matchSummary =
        document.getElementById("match-summary");


    // -----------------------------------------------------
    // VENUE DASHBOARD
    // -----------------------------------------------------

    const reportVenue =
        document.getElementById("report-venue");

    const venueAverageScore =
        document.getElementById("venue-average-score");

    const venueMatches =
        document.getElementById("venue-matches");


    // -----------------------------------------------------
    // APPLICATION STATE
    // -----------------------------------------------------

    let optionsLoaded = false;


    // -----------------------------------------------------
    // INITIAL STATE
    // -----------------------------------------------------

    if (predictBtn) {
        predictBtn.disabled = true;
        predictBtn.style.opacity = "0.65";
        predictBtn.style.cursor = "wait";
    }

    if (resultSection) {
        resultSection.classList.add("hidden");
    }

    if (scorePredictionCard) {
        scorePredictionCard.classList.add("hidden");
    }


    // -----------------------------------------------------
    // LOAD IPL OPTIONS
    // -----------------------------------------------------

    async function loadOptions() {

        try {

            const response =
                await fetch("/api/options");

            const data =
                await response.json();

            if (!response.ok) {
                throw new Error(
                    data.error ||
                    "Unable to load options."
                );
            }

            if (
                !Array.isArray(data.teams) ||
                !Array.isArray(data.venues)
            ) {
                throw new Error(
                    "Invalid IPL options received from server."
                );
            }

            if (data.teams.length === 0) {
                throw new Error(
                    "No IPL teams were received from the server."
                );
            }

            if (data.venues.length === 0) {
                throw new Error(
                    "No IPL venues were received from the server."
                );
            }

            populateSelect(
                team1Select,
                data.teams,
                "Select Team 1"
            );

            populateSelect(
                team2Select,
                data.teams,
                "Select Team 2"
            );

            populateSelect(
                venueSelect,
                data.venues,
                "Select Venue"
            );

            updateTossTeams();

            optionsLoaded = true;

            if (predictBtn) {
                predictBtn.disabled = false;
                predictBtn.style.opacity = "1";
                predictBtn.style.cursor = "pointer";
            }

            console.log(
                "IPL options loaded successfully."
            );

        } catch (error) {

            optionsLoaded = false;

            console.error(
                "Options error:",
                error
            );

            showError(
                "Unable to load IPL data. Please make sure Flask is running."
            );

            if (predictBtn) {
                predictBtn.disabled = true;
                predictBtn.style.opacity = "0.65";
                predictBtn.style.cursor = "not-allowed";
            }
        }
    }


    // -----------------------------------------------------
    // POPULATE SELECT
    // -----------------------------------------------------

    function populateSelect(
        selectElement,
        values,
        placeholder
    ) {

        if (!selectElement) return;

        selectElement.innerHTML = "";

        const placeholderOption =
            document.createElement("option");

        placeholderOption.value = "";
        placeholderOption.textContent = placeholder;
        placeholderOption.disabled = true;
        placeholderOption.selected = true;

        selectElement.appendChild(
            placeholderOption
        );

        values.forEach(value => {

            const option =
                document.createElement("option");

            option.value = value;
            option.textContent = value;

            selectElement.appendChild(
                option
            );
        });
    }


    // -----------------------------------------------------
    // UPDATE TOSS TEAMS
    // -----------------------------------------------------

    function updateTossTeams() {

        if (
            !tossWinnerSelect ||
            !team1Select ||
            !team2Select
        ) {
            return;
        }

        tossWinnerSelect.innerHTML = "";

        const placeholder =
            document.createElement("option");

        placeholder.value = "";
        placeholder.textContent =
            "Select Toss Winner";

        placeholder.disabled = true;
        placeholder.selected = true;

        tossWinnerSelect.appendChild(
            placeholder
        );

        if (team1Select.value) {

            const option1 =
                document.createElement("option");

            option1.value =
                team1Select.value;

            option1.textContent =
                team1Select.value;

            tossWinnerSelect.appendChild(
                option1
            );
        }

        if (
            team2Select.value &&
            team2Select.value !== team1Select.value
        ) {

            const option2 =
                document.createElement("option");

            option2.value =
                team2Select.value;

            option2.textContent =
                team2Select.value;

            tossWinnerSelect.appendChild(
                option2
            );
        }
    }


    // -----------------------------------------------------
    // EVENTS
    // -----------------------------------------------------

    if (team1Select) {
        team1Select.addEventListener(
            "change",
            updateTossTeams
        );
    }

    if (team2Select) {
        team2Select.addEventListener(
            "change",
            updateTossTeams
        );
    }

    if (predictBtn) {
        predictBtn.addEventListener(
            "click",
            makePrediction
        );
    }


    // -----------------------------------------------------
    // MAKE PREDICTION
    // -----------------------------------------------------

    async function makePrediction() {

        clearError();

        if (!optionsLoaded) {

            showError(
                "IPL data is still loading. Please wait a moment and try again."
            );

            return;
        }


        // ---------------------------------------------
        // VALIDATION
        // ---------------------------------------------

        if (!team1Select.value) {

            showError(
                "Please select Team 1."
            );

            return;
        }

        if (!team2Select.value) {

            showError(
                "Please select Team 2."
            );

            return;
        }

        if (
            team1Select.value ===
            team2Select.value
        ) {

            showError(
                "Team 1 and Team 2 must be different."
            );

            return;
        }

        if (!venueSelect.value) {

            showError(
                "Please select a venue."
            );

            return;
        }

        if (!matchDateInput.value) {

            showError(
                "Please select the future match date."
            );

            return;
        }

        const selectedDate =
            new Date(
                matchDateInput.value +
                "T00:00:00"
            );

        const today =
            new Date();

        today.setHours(
            0,
            0,
            0,
            0
        );

        if (
            Number.isNaN(
                selectedDate.getTime()
            )
        ) {

            showError(
                "Please select a valid match date."
            );

            return;
        }

        if (selectedDate <= today) {

            showError(
                "Please select a future match date."
            );

            return;
        }

        if (!tossWinnerSelect.value) {

            showError(
                "Please select the toss winner."
            );

            return;
        }

        if (!tossDecisionSelect.value) {

            showError(
                "Please select the toss decision."
            );

            return;
        }


        // ---------------------------------------------
        // AUTOMATIC SEASON
        // ---------------------------------------------

        const season =
            selectedDate.getFullYear();


        // ---------------------------------------------
        // REQUEST
        // ---------------------------------------------

        const requestData = {

            team1:
                team1Select.value,

            team2:
                team2Select.value,

            venue:
                venueSelect.value,

            match_date:
                matchDateInput.value,

            season:
                season,

            toss_winner:
                tossWinnerSelect.value,

            toss_decision:
                tossDecisionSelect.value
        };


        console.log(
            "Future prediction request:",
            requestData
        );


        // ---------------------------------------------
        // RESET RESULTS
        // ---------------------------------------------

        resetScorePrediction();

        resetVenueDashboard();

        if (resultSection) {

            resultSection.classList.add(
                "hidden"
            );
        }

        if (scorePredictionCard) {

            scorePredictionCard.classList.add(
                "hidden"
            );
        }


        // ---------------------------------------------
        // LOADING
        // ---------------------------------------------

        setLoading(true);


        try {

            const response =
                await fetch(
                    "/api/predict",
                    {
                        method: "POST",

                        headers: {
                            "Content-Type":
                                "application/json"
                        },

                        body:
                            JSON.stringify(
                                requestData
                            )
                    }
                );


            const data =
                await response.json();


            if (!response.ok) {

                throw new Error(
                    data.error ||
                    "Prediction failed."
                );
            }


            // -----------------------------------------
            // DISPLAY
            // -----------------------------------------

            displayPrediction(data);


        } catch (error) {

            console.error(
                "Prediction error:",
                error
            );

            showError(
                error.message ||
                "Something went wrong while predicting."
            );

        } finally {

            setLoading(false);
        }
    }


    // -----------------------------------------------------
    // DISPLAY PREDICTION
    // -----------------------------------------------------

    function displayPrediction(data) {

        const prediction =
            data.prediction;

        const analysis =
            data.analysis;

        const match =
            data.match;


        if (!prediction) {

            showError(
                "Prediction response is missing prediction data."
            );

            return;
        }


        // ---------------------------------------------
        // WINNER
        // ---------------------------------------------

        if (predictedWinner) {

            predictedWinner.textContent =
                prediction.winner || "—";
        }


        // ---------------------------------------------
        // PROBABILITIES
        // ---------------------------------------------

        const team1ProbabilityValue =
            safeNumber(
                prediction.team1_win_probability
            );

        const team2ProbabilityValue =
            safeNumber(
                prediction.team2_win_probability
            );


        if (team1Probability) {

            team1Probability.textContent =
                `${team1ProbabilityValue}%`;
        }


        if (team2Probability) {

            team2Probability.textContent =
                `${team2ProbabilityValue}%`;
        }


        // ---------------------------------------------
        // PROGRESS BARS
        // ---------------------------------------------

        if (team1Progress) {
            team1Progress.style.width = "0%";
        }

        if (team2Progress) {
            team2Progress.style.width = "0%";
        }


        setTimeout(() => {

            if (team1Progress) {

                team1Progress.style.width =
                    `${team1ProbabilityValue}%`;
            }

            if (team2Progress) {

                team2Progress.style.width =
                    `${team2ProbabilityValue}%`;
            }

        }, 100);


        // ---------------------------------------------
        // MODEL CONFIDENCE
        // ---------------------------------------------

        if (modelConfidence) {

            modelConfidence.textContent =
                `${safeNumber(
                    prediction.model_confidence
                )}%`;
        }


        // ---------------------------------------------
        // SCORE PREDICTION
        // ---------------------------------------------

        displayScorePrediction(
            prediction
        );


        // ---------------------------------------------
        // TEAM ANALYSIS
        // ---------------------------------------------

        if (
            analysis &&
            analysis.team1 &&
            analysis.team2
        ) {

            updateTeamAnalysis(
                "team1",
                analysis.team1
            );

            updateTeamAnalysis(
                "team2",
                analysis.team2
            );
        }


        // ---------------------------------------------
        // HEAD TO HEAD
        // ---------------------------------------------

        if (
            analysis &&
            analysis.head_to_head
        ) {

            updateH2H(
                analysis.head_to_head,
                analysis.team1,
                analysis.team2
            );
        }


        // ---------------------------------------------
        // VENUE ANALYSIS
        // ---------------------------------------------

        updateVenueDashboard(
            match,
            analysis
        );


        // ---------------------------------------------
        // FINAL MATCH REPORT
        // ---------------------------------------------

        generateMatchReport(
            match,
            prediction,
            analysis
        );


        // ---------------------------------------------
        // SHOW RESULTS
        // ---------------------------------------------

        if (resultSection) {

            resultSection.classList.remove(
                "hidden"
            );
        }

        if (scorePredictionCard) {

            scorePredictionCard.classList.remove(
                "hidden"
            );
        }


        // ---------------------------------------------
        // SCROLL
        // ---------------------------------------------

        setTimeout(() => {

            if (resultSection) {

                resultSection.scrollIntoView({
                    behavior: "smooth",
                    block: "start"
                });
            }

        }, 150);
    }


    // -----------------------------------------------------
    // SCORE PREDICTION
    // -----------------------------------------------------

    function displayScorePrediction(prediction) {

        if (!prediction) return;

        const score =
            prediction.expected_first_innings_score;

        const range =
            prediction.estimated_score_range;


        if (expectedScore) {

            if (
                score !== null &&
                score !== undefined &&
                score !== ""
            ) {

                expectedScore.textContent =
                    `${score}`;

            } else {

                expectedScore.textContent =
                    "—";
            }
        }


        if (scoreRange) {

            if (
                range !== null &&
                range !== undefined &&
                range !== ""
            ) {

                scoreRange.textContent =
                    range;

            } else {

                scoreRange.textContent =
                    "—";
            }
        }
    }


    // -----------------------------------------------------
    // RESET SCORE PREDICTION
    // -----------------------------------------------------

    function resetScorePrediction() {

        if (expectedScore) {
            expectedScore.textContent = "—";
        }

        if (scoreRange) {
            scoreRange.textContent = "—";
        }
    }


    // -----------------------------------------------------
    // TEAM ANALYSIS
    // -----------------------------------------------------

    function updateTeamAnalysis(
        teamId,
        team
    ) {

        if (!team) return;

        const prefix =
            teamId === "team1"
                ? "team1"
                : "team2";


        const nameElement =
            document.getElementById(
                `analysis-${prefix}`
            );

        const eloElement =
            document.getElementById(
                `${prefix}-elo`
            );

        const winRateElement =
            document.getElementById(
                `${prefix}-win-rate`
            );

        const formElement =
            document.getElementById(
                `${prefix}-form`
            );

        const venueElement =
            document.getElementById(
                `${prefix}-venue`
            );

        const runsElement =
            document.getElementById(
                `${prefix}-runs`
            );

        const wicketsElement =
            document.getElementById(
                `${prefix}-wickets`
            );


        if (nameElement) {

            nameElement.textContent =
                team.name || "—";
        }

        if (eloElement) {

            eloElement.textContent =
                formatNumber(team.elo);
        }

        if (winRateElement) {

            winRateElement.textContent =
                `${formatNumber(team.win_rate)}%`;
        }

        if (formElement) {

            formElement.textContent =
                `${formatNumber(team.form_5)}%`;
        }

        if (venueElement) {

            venueElement.textContent =
                `${formatNumber(team.venue_win_rate)}%`;
        }

        if (runsElement) {

            runsElement.textContent =
                formatNumber(
                    team.recent_runs
                );
        }

        if (wicketsElement) {

            wicketsElement.textContent =
                formatNumber(
                    team.recent_wickets
                );
        }
    }


    // -----------------------------------------------------
    // HEAD TO HEAD
    // -----------------------------------------------------

    function updateH2H(
        h2h,
        team1,
        team2
    ) {

        if (!h2h) return;


        const total =
            document.getElementById(
                "h2h-total"
            );

        const team1Wins =
            document.getElementById(
                "h2h-team1"
            );

        const team2Wins =
            document.getElementById(
                "h2h-team2"
            );

        const team1Name =
            document.getElementById(
                "h2h-team1-name"
            );

        const team2Name =
            document.getElementById(
                "h2h-team2-name"
            );


        if (total) {

            total.textContent =
                safeNumber(
                    h2h.total_matches
                );
        }

        if (team1Wins) {

            team1Wins.textContent =
                safeNumber(
                    h2h.team1_wins
                );
        }

        if (team2Wins) {

            team2Wins.textContent =
                safeNumber(
                    h2h.team2_wins
                );
        }

        if (
            team1Name &&
            team1
        ) {

            team1Name.textContent =
                `${team1.name} Wins`;
        }

        if (
            team2Name &&
            team2
        ) {

            team2Name.textContent =
                `${team2.name} Wins`;
        }
    }


    // -----------------------------------------------------
    // VENUE DASHBOARD
    // -----------------------------------------------------

    function updateVenueDashboard(
        match,
        analysis
    ) {

        const venue =
            match?.venue;

        const venueAnalysis =
            analysis?.venue;


        if (reportVenue) {

            reportVenue.textContent =
                venue || "—";
        }


        /*
         * Only display venue statistics when
         * the backend actually provides them.
         *
         * This avoids inventing values.
         */

        if (venueAverageScore) {

            const average =
                getFirstAvailable(
                    venueAnalysis,
                    [
                        "average_score",
                        "venue_average_score",
                        "avg_score"
                    ]
                );

            venueAverageScore.textContent =
                average !== null
                    ? formatNumber(average)
                    : "—";
        }


        if (venueMatches) {

            const matches =
                getFirstAvailable(
                    venueAnalysis,
                    [
                        "matches",
                        "total_matches",
                        "venue_matches"
                    ]
                );

            venueMatches.textContent =
                matches !== null
                    ? safeNumber(matches)
                    : "—";
        }
    }


    // -----------------------------------------------------
    // RESET VENUE DASHBOARD
    // -----------------------------------------------------

    function resetVenueDashboard() {

        if (reportVenue) {
            reportVenue.textContent = "—";
        }

        if (venueAverageScore) {
            venueAverageScore.textContent = "—";
        }

        if (venueMatches) {
            venueMatches.textContent = "—";
        }
    }


    // -----------------------------------------------------
    // FINAL MATCH REPORT
    // -----------------------------------------------------

    function generateMatchReport(
        match,
        prediction,
        analysis
    ) {

        if (!matchSummary) return;

        if (
            !match ||
            !prediction ||
            !analysis
        ) {

            return;
        }


        const team1 =
            match.team1 || "Team 1";

        const team2 =
            match.team2 || "Team 2";

        const winner =
            prediction.winner || "—";

        const matchDate =
            match.match_date || "—";

        const venue =
            match.venue || "—";

        const tossWinner =
            match.toss_winner || "—";

        const tossDecision =
            match.toss_decision || "—";


        const team1Probability =
            safeNumber(
                prediction.team1_win_probability
            );

        const team2Probability =
            safeNumber(
                prediction.team2_win_probability
            );

        const confidence =
            safeNumber(
                prediction.model_confidence
            );


        const score =
            prediction.expected_first_innings_score;

        const scoreRange =
            prediction.estimated_score_range;


        const eloDifference =
            safeNumber(
                analysis.elo_difference
            );


        let report = "";


        // ---------------------------------------------
        // INTRO
        // ---------------------------------------------

        report +=
            `<p><strong>${winner}</strong> is the model's predicted winner `;

        report +=
            `for the future IPL match between `;

        report +=
            `<strong>${team1}</strong> and `;

        report +=
            `<strong>${team2}</strong>.</p>`;


        // ---------------------------------------------
        // MATCH CONDITIONS
        // ---------------------------------------------

        report +=
            `<p><strong>Match Conditions:</strong> `;

        report +=
            `${matchDate} at ${venue}. `;

        report +=
            `Toss: ${tossWinner}, chose to ${tossDecision}.</p>`;


        // ---------------------------------------------
        // PROBABILITY
        // ---------------------------------------------

        report +=
            `<p><strong>Win Probability:</strong> `;

        report +=
            `${team1} ${team1Probability}%`;

        report +=
            ` vs ${team2} ${team2Probability}%. `;

        report +=
            `The displayed model confidence is `;

        report +=
            `<strong>${confidence}%</strong>.</p>`;


        // ---------------------------------------------
        // SCORE
        // ---------------------------------------------

        if (
            score !== null &&
            score !== undefined &&
            score !== ""
        ) {

            report +=
                `<p><strong>Score Projection:</strong> `;

            report +=
                `The expected first-innings score is `;

            report +=
                `<strong>${score} runs</strong>`;

            if (scoreRange) {

                report +=
                    `, with an estimated range of `;

                report +=
                    `<strong>${scoreRange}</strong>`;
            }

            report +=
                `.</p>`;
        }


        // ---------------------------------------------
        // TEAM STRENGTH
        // ---------------------------------------------

        report +=
            `<p><strong>Team Strength:</strong> `;

        if (eloDifference > 0) {

            report +=
                `${team1} has the higher historical Elo rating `;

            report +=
                `in the current analysis.`;

        } else if (eloDifference < 0) {

            report +=
                `${team2} has the higher historical Elo rating `;

            report +=
                `in the current analysis.`;

        } else {

            report +=
                `Both teams have similar historical Elo ratings.`;
        }

        report +=
            `</p>`;


        // ---------------------------------------------
        // ANALYSIS FACTORS
        // ---------------------------------------------

        report +=
            `<p><strong>Analysis Factors:</strong> `;

        report +=
            `The prediction uses historical team performance, `;

        report +=
            `recent form, venue performance, head-to-head history, `;

        report +=
            `scoring and wicket statistics, toss information, `;

        report +=
            `and the selected future match date.</p>`;


        // ---------------------------------------------
        // DISCLAIMER
        // ---------------------------------------------

        report +=
            `<p class="report-disclaimer">`;

        report +=
            `This is a machine-learning estimate based on historical IPL data. `;

        report +=
            `It should not be interpreted as a guaranteed match result.`;

        report +=
            `</p>`;


        matchSummary.innerHTML =
            report;
    }


    // -----------------------------------------------------
    // LOADING
    // -----------------------------------------------------

    function setLoading(isLoading) {

        if (
            !loading ||
            !predictBtn
        ) {

            return;
        }


        if (isLoading) {

            loading.classList.remove(
                "hidden"
            );

            predictBtn.disabled = true;

            predictBtn.style.opacity =
                "0.65";

            predictBtn.style.cursor =
                "wait";

        } else {

            loading.classList.add(
                "hidden"
            );

            predictBtn.disabled =
                !optionsLoaded;

            predictBtn.style.opacity =
                optionsLoaded
                    ? "1"
                    : "0.65";

            predictBtn.style.cursor =
                optionsLoaded
                    ? "pointer"
                    : "not-allowed";
        }
    }


    // -----------------------------------------------------
    // ERROR
    // -----------------------------------------------------

    function showError(message) {

        if (!errorMessage) return;

        errorMessage.textContent =
            message;

        errorMessage.classList.remove(
            "hidden"
        );
    }


    function clearError() {

        if (!errorMessage) return;

        errorMessage.textContent =
            "";

        errorMessage.classList.add(
            "hidden"
        );
    }


    // -----------------------------------------------------
    // SAFE NUMBER
    // -----------------------------------------------------

    function safeNumber(value) {

        if (
            value === null ||
            value === undefined ||
            value === "" ||
            Number.isNaN(Number(value))
        ) {

            return 0;
        }

        return Number(value);
    }


    // -----------------------------------------------------
    // NUMBER FORMAT
    // -----------------------------------------------------

    function formatNumber(value) {

        if (
            value === null ||
            value === undefined ||
            value === "" ||
            Number.isNaN(Number(value))
        ) {

            return "0";
        }

        return Number(value).toFixed(2);
    }


    // -----------------------------------------------------
    // GET FIRST AVAILABLE VALUE
    // -----------------------------------------------------

    function getFirstAvailable(
        object,
        keys
    ) {

        if (!object) {
            return null;
        }

        for (const key of keys) {

            if (
                object[key] !== null &&
                object[key] !== undefined &&
                object[key] !== ""
            ) {

                return object[key];
            }
        }

        return null;
    }


    // -----------------------------------------------------
    // START
    // -----------------------------------------------------

    loadOptions();

});