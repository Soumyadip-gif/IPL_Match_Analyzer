🏏 IPL Match Analyzer

An AI/ML-based web application that analyzes IPL match data and predicts the match winner, win probability, and expected first-innings score for future matches.

🚀 Features
🏆 Match Winner Prediction
📊 Win Probability for both teams
🎯 Expected First-Innings Score
📅 Future Match Prediction using match date
🏟️ Venue Analysis
⚔️ Head-to-Head Analysis
🏏 Team Performance Analysis
🪙 Toss Analysis
📋 Automatic Match Report
💻 Professional Interactive Dashboard
🔒 Leakage-aware historical feature generation
🤖 Machine Learning
Winner Prediction
Random Forest
Gradient Boosting
XGBoost
Time-series based validation
Historical and recent team-performance features
Score Prediction
Regression-based ML model
Recent scoring patterns
Venue statistics
Team performance
Toss information
Historical match features

Predictions are estimates based on historical IPL data and should not be considered guaranteed results.

📊 Dataset

The project uses historical IPL match and ball-by-ball data.

1,997 matches
15 IPL seasons
Match-level and delivery-level information
Team, venue, toss, result, batting and bowling information
🛠️ Tech Stack

Python • Pandas • NumPy • Scikit-learn • XGBoost • Flask • Joblib • HTML • CSS • JavaScript • Chart.js
📁 Project Structure
IPL-Match-Analyzer/
│
├── data/
├── model/
├── notebooks/
├── templates/
├── static/
│   ├── css/
│   └── js/
│
├── app.py
├── prediction_engine.py
├── feature_engineering.py
├── train_model.py
├── train_score_model.py
├── data_cleaning.py
├── data_quality_check.py
├── eda.py
├── requirements.txt
└── .gitignore
⚙️ Installation
git clone https://github.com/Soumyadip-gif/IPL_Match_Analyzer.git
cd IPL_Match_Analyzer

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt

🔮 Example

Input:

Team 1: MI
Team 2: CSK
Venue: Selected IPL venue
Match Date: Future date
Toss Winner & Decision

Output:

Predicted Winner
Win Probability
Expected Score
Score Range
Team Analysis
Venue Analysis
Head-to-Head
Match Report
📌 Future Improvements
Playing XI-based prediction
Player performance analysis
Live match integration
Advanced deep-learning models
More detailed dashboards
👨‍💻 Author

Soumyadip Porel
BCA Student | AI/ML & Data Analytics Enthusiast

GitHub: IPL Match Analyzer Repository
