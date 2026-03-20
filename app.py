import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import altair as alt
import math
try:
    import joblib
except ModuleNotFoundError:
    joblib = None

# Reduce TensorFlow startup noise in Streamlit logs.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf


# 1. CONFIG & "DAYLIGHT" THEME CSS

st.set_page_config(page_title="PremierVision", layout="wide")

st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;900&display=swap" rel="stylesheet">
    
    <style>
    /* --- ROOT VARIABLES (LIGHT THEME) --- */
    :root {
        --primary-blue: #0052cc;
        --primary-purple: #6b38c7;
        --text-dark: #172b4d;
        --text-gray: #5e6c84;
        --bg-gradient-start: #f4f5f7;
        --bg-gradient-end: #dfe1e6;
    }

    /* --- GLOBAL STYLES --- */
    .stApp {
        background: radial-gradient(circle at 50% 10%, #ffffff 0%, #ebecf0 100%);
        color: var(--text-dark);
        font-family: 'Rajdhani', sans-serif;
        scroll-behavior: smooth;
        overflow-x: hidden;
    }
    
    .block-container {
        padding-top: 0rem;
        padding-bottom: 5rem;
    }

    /* --- ANIMATIONS --- */
    @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
    @keyframes slideUp { from { transform: translateY(50px); opacity: 0; } to { transform: translateY(0); opacity: 1; } }
    @keyframes scroll {
        0% { transform: translateX(0); }
        100% { transform: translateX(-50%); }
    }
    
    /* --- HERO SECTION --- */
    .hero-section {
        height: 100vh;
        width: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        position: relative;
        margin-bottom: 50px;
        /* Subtle grid pattern for light mode */
        background-image: linear-gradient(#e5e8ec 1px, transparent 1px), linear-gradient(90deg, #e5e8ec 1px, transparent 1px);
        background-size: 40px 40px;
    }
    
    .hero-title {
        font-size: 100px;
        font-weight: 900;
        background: linear-gradient(to right, #0052cc, #6b38c7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: 5px;
        /* Soft shadow for depth */
        filter: drop-shadow(0 10px 20px rgba(0, 82, 204, 0.2));
        animation: slideUp 1.2s ease-out;
    }
    
    .hero-subtitle {
        font-size: 20px;
        color: var(--text-gray);
        letter-spacing: 8px;
        text-transform: uppercase;
        font-weight: 600;
        margin-top: 10px;
        margin-bottom: 60px;
        animation: slideUp 1.5s ease-out;
    }

    /* --- LOGO SLIDER --- */
    .slider-container {
        width: 100%;
        overflow: hidden;
        position: relative;
        margin-bottom: 40px;
        /* White fade edges */
        -webkit-mask-image: linear-gradient(to right, transparent, black 10%, black 90%, transparent);
        mask-image: linear-gradient(to right, transparent, black 10%, black 90%, transparent);
        animation: fadeIn 2s ease-in;
    }
    
    .slider-track {
        display: flex;
        width: max-content;
        animation: scroll 40s linear infinite;
    }
    
    .slider-logo {
        height: 80px;
        margin: 0 40px;
        opacity: 0.6;
        /* Adjusted filter for light mode (darker gray) */
        filter: grayscale(100%);
        transition: all 0.3s;
        object-fit: contain;
    }
    
    .slider-logo:hover {
        opacity: 1;
        filter: grayscale(0%);
        transform: scale(1.1);
    }

    /* --- SCROLL INDICATOR --- */
    .scroll-indicator {
        position: absolute;
        bottom: 40px;
        left: 50%;
        transform: translateX(-50%);
        display: flex;
        flex-direction: column;
        align-items: center;
        opacity: 0.6;
        animation: fadeIn 3s ease-in;
    }
    
    .scroll-text {
        font-size: 12px;
        letter-spacing: 2px;
        margin-bottom: 10px;
        color: var(--text-dark);
        font-weight: bold;
        text-transform: uppercase;
    }
    
    .scroll-arrow {
        width: 15px;
        height: 15px;
        border-bottom: 3px solid var(--text-dark);
        border-right: 3px solid var(--text-dark);
        transform: rotate(45deg);
    }

    /* --- MAIN APP INTERFACE --- */
    .main-interface {
        opacity: 0;
        animation: slideUp 1s ease-out forwards;
        animation-delay: 0.2s;
    }

    /* --- WIDGETS (Light Mode) --- */
    .stSelectbox label { 
        color: var(--primary-blue) !important; 
        font-weight: 800; 
        letter-spacing: 1px; 
    }
    .stSelectbox > div > div {
        background-color: #ffffff !important;
        border: 1px solid #dfe1e6 !important;
        border-radius: 12px !important;
        color: var(--text-dark) !important;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
    }
    
    .stButton > button {
        border-radius: 50px;
        background: linear-gradient(90deg, var(--primary-blue), var(--primary-purple));
        border: none;
        color: white;
        font-family: 'Rajdhani', sans-serif;
        font-weight: 700;
        font-size: 18px;
        padding: 15px 30px;
        letter-spacing: 2px;
        box-shadow: 0 10px 25px rgba(107, 56, 199, 0.3);
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 15px 30px rgba(107, 56, 199, 0.5);
    }

    /* --- WARNING ALERT READABILITY --- */
    [data-testid="stAlert"] {
        background-color: #fff3cd !important;
        border: 1px solid #ffe08a !important;
        border-radius: 12px !important;
    }

    [data-testid="stAlert"] * {
        color: #4d3a00 !important;
    }

    /* --- RESULT HUD (FROSTED GLASS LIGHT) --- */
    .result-hud {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        height: 100%;
        min-height: 400px;
        border-radius: 30px;
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(20px);
        border: 1px solid #ffffff;
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.1);
        animation: slideUp 0.8s ease-out;
    }
    </style>
""", unsafe_allow_html=True)


# 2. DATA & LOGIC


if 'prediction' not in st.session_state:
    st.session_state['prediction'] = None
if 'prediction_debug' not in st.session_state:
    st.session_state['prediction_debug'] = None

# --- LOGO MAPPING ---
logo_map = {
    "Arsenal": "https://upload.wikimedia.org/wikipedia/en/thumb/5/53/Arsenal_FC.svg/1200px-Arsenal_FC.svg.png",
    "Aston Villa": "https://upload.wikimedia.org/wikipedia/en/9/9a/Aston_Villa_FC_new_crest.svg",
    "Bournemouth": "https://resources.premierleague.com/premierleague/badges/t91.svg",
    "Brentford": "https://resources.premierleague.com/premierleague/badges/t94.svg",
    "Brighton": "https://resources.premierleague.com/premierleague/badges/t36.svg",
    "Burnley": "https://upload.wikimedia.org/wikipedia/en/6/6d/Burnley_FC_Logo.svg",
    "Chelsea": "https://resources.premierleague.com/premierleague/badges/t8.svg",
    "Crystal Palace": "https://resources.premierleague.com/premierleague/badges/t31.svg",
    "Everton": "https://resources.premierleague.com/premierleague/badges/t11.svg",
    "Fulham": "https://resources.premierleague.com/premierleague/badges/t54.svg",
    "Ipswich": "https://upload.wikimedia.org/wikipedia/en/4/43/Ipswich_Town.svg",
    "Leeds": "https://upload.wikimedia.org/wikipedia/en/5/54/Leeds_United_F.C._logo.svg",
    "Leicester": "https://resources.premierleague.com/premierleague/badges/t13.svg",
    "Liverpool": "https://resources.premierleague.com/premierleague/badges/t14.svg",
    "Luton": "https://resources.premierleague.com/premierleague/badges/t102.svg",
    "Man City": "https://resources.premierleague.com/premierleague/badges/t43.svg",
    "Man United": "https://resources.premierleague.com/premierleague/badges/t1.svg",
    "Newcastle": "https://resources.premierleague.com/premierleague/badges/t4.svg",
    "Nott'm Forest": "https://resources.premierleague.com/premierleague/badges/t17.svg",
    "Norwich": "https://upload.wikimedia.org/wikipedia/en/1/17/Norwich_City_FC_logo.svg",
    "Sheffield United": "https://resources.premierleague.com/premierleague/badges/t49.svg",
    "Southampton": "https://resources.premierleague.com/premierleague/badges/t20.svg",
    "Stoke": "https://upload.wikimedia.org/wikipedia/en/2/29/Stoke_City_FC.svg",
    "Sunderland": "https://upload.wikimedia.org/wikipedia/en/7/77/Logo_Sunderland.svg",
    "Swansea": "https://upload.wikimedia.org/wikipedia/en/f/f9/Swansea_City_AFC_logo.svg",
    "Tottenham": "https://resources.premierleague.com/premierleague/badges/t6.svg",
    "Watford": "https://resources.premierleague.com/premierleague/badges/t57.svg",
    "West Brom": "https://resources.premierleague.com/premierleague/badges/t35.svg",
    "West Ham": "https://resources.premierleague.com/premierleague/badges/t21.svg",
    "Wolves": "https://resources.premierleague.com/premierleague/badges/t39.svg",
    "Middlesbrough": "https://upload.wikimedia.org/wikipedia/en/2/2c/Middlesbrough_FC_crest.svg",
    "Blackburn": "https://upload.wikimedia.org/wikipedia/en/0/0f/Blackburn_Rovers.svg",
    "Bolton": "https://upload.wikimedia.org/wikipedia/en/8/82/Bolton_Wanderers_FC_logo.svg"
}

def get_logo(team):
    if team in logo_map: return logo_map[team]
    for key in logo_map:
        if key in team or team in key: return logo_map[key]
    return "https://resources.premierleague.com/premierleague/badges/t3.svg"

@st.cache_resource
def load_model():
    if os.path.exists("epl_model.keras"):
        return tf.keras.models.load_model('epl_model.keras')
    else:
        class MockModel:
            def predict(self, x):
                x = np.asarray(x, dtype=np.float32)
                if x.ndim == 1:
                    x = x.reshape(1, -1)

                # Deterministic fallback: convert input features to class probabilities.
                home_score = x[:, 0] * 1.4 + x[:, 4] * 0.6
                draw_score = x[:, 1] * 1.2 + x[:, 5] * 0.4
                away_score = x[:, 2] * 1.4 + x[:, 3] * 0.6

                logits = np.stack([home_score, draw_score, away_score], axis=1)
                logits = logits - logits.max(axis=1, keepdims=True)
                exp_logits = np.exp(logits)
                return exp_logits / exp_logits.sum(axis=1, keepdims=True)
        return MockModel()


@st.cache_resource
def load_scaler():
    if os.path.exists("scaler.pkl") and joblib is not None:
        try:
            return joblib.load("scaler.pkl")
        except Exception:
            pass

    class IdentityScaler:
        def transform(self, x):
            return x

    return IdentityScaler()

@st.cache_data
def load_data():
    if os.path.exists("epl_final.csv"):
        df = pd.read_csv("epl_final.csv")
        rename_map = {'FullTimeHomeGoals': 'FTHG', 'FullTimeAwayGoals': 'FTAG',
                      'HomeYellowCards': 'HY', 'AwayYellowCards': 'AY',
                      'HomeRedCards': 'HR', 'AwayRedCards': 'AR'}
        df.rename(columns=rename_map, inplace=True)
        return df
    else:
        data = {
            'Season': ['2023-2024'] * 4, 
            'HomeTeam': ['Arsenal', 'Liverpool', 'Chelsea', 'Man City'],
            'AwayTeam': ['Man United', 'Everton', 'Tottenham', 'Wolves']
        }
        return pd.DataFrame(data)

model = load_model()
scaler = load_scaler()
df = load_data()

MODEL_FEATURE_ORDER = [
    "HomeShots", "AwayShots", "HomeShotsOnTarget", "AwayShotsOnTarget",
    "HomeCorners", "AwayCorners", "HomeFouls", "AwayFouls", "HY", "AY", "HR", "AR",
    "HomeForm", "AwayForm", "FormDiff", "HomeAdvantage", "ShotsDiff",
    "ShotsOnTargetDiff", "FoulsDiff", "YellowDiff", "RedDiff",
    "HomeAvgGoalsScored", "HomeAvgGoalsConceded", "AwayAvgGoalsScored",
    "AwayAvgGoalsConceded", "HomeGoalDiffTrend", "AwayGoalDiffTrend",
    "MatchNumber", "TotalMatches", "SeasonProgress", "SeasonAvgPoints_Home",
    "SeasonAvgPoints_Away", "Points_HomeBefore", "Points_AwayBefore",
    "PointsGap_HomeAway", "RelegationPressure_Home",
]


def _safe_mean(frame, column_name, default_value=0.0):
    if column_name not in frame.columns:
        return float(default_value)
    series = pd.to_numeric(frame[column_name], errors="coerce").dropna()
    if series.empty:
        return float(default_value)
    return float(series.mean())


def _points_from_result(result, side):
    if pd.isna(result):
        return 0
    if side == "home":
        return 3 if result == "H" else 1 if result == "D" else 0
    return 3 if result == "A" else 1 if result == "D" else 0


def _team_total_points(season_data, team):
    points = 0
    if "FullTimeResult" not in season_data.columns:
        return 0.0
    home_rows = season_data[season_data["HomeTeam"] == team]
    away_rows = season_data[season_data["AwayTeam"] == team]
    points += sum(_points_from_result(r, "home") for r in home_rows["FullTimeResult"])
    points += sum(_points_from_result(r, "away") for r in away_rows["FullTimeResult"])
    return float(points)


def _team_total_matches(season_data, team):
    return int((season_data["HomeTeam"] == team).sum() + (season_data["AwayTeam"] == team).sum())


def _team_form_points(matches, side, window=5):
    if matches.empty or "FullTimeResult" not in matches.columns:
        return 1.2
    recent = matches.tail(window)
    pts = sum(_points_from_result(r, side) for r in recent["FullTimeResult"])
    return float(pts / max(len(recent), 1))


def _league_table_rank(season_data, team):
    teams = sorted(set(season_data["HomeTeam"]).union(set(season_data["AwayTeam"])))
    standings = []
    for t in teams:
        points = _team_total_points(season_data, t)
        gf = _safe_mean(season_data[season_data["HomeTeam"] == t], "FTHG", 0.0) * max(len(season_data[season_data["HomeTeam"] == t]), 1)
        gf += _safe_mean(season_data[season_data["AwayTeam"] == t], "FTAG", 0.0) * max(len(season_data[season_data["AwayTeam"] == t]), 1)
        ga = _safe_mean(season_data[season_data["HomeTeam"] == t], "FTAG", 0.0) * max(len(season_data[season_data["HomeTeam"] == t]), 1)
        ga += _safe_mean(season_data[season_data["AwayTeam"] == t], "FTHG", 0.0) * max(len(season_data[season_data["AwayTeam"] == t]), 1)
        standings.append((t, points, gf - ga, gf))

    standings.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)
    for idx, (name, _, _, _) in enumerate(standings, start=1):
        if name == team:
            return idx
    return len(standings)


def build_model_input_from_stats(season_data, home_team, away_team):
    season_df = season_data.copy()
    if "MatchDate" in season_df.columns:
        season_df["MatchDate"] = pd.to_datetime(season_df["MatchDate"], errors="coerce")
        season_df = season_df.sort_values("MatchDate")

    home_rows = season_df[season_df["HomeTeam"] == home_team]
    away_rows = season_df[season_df["AwayTeam"] == away_team]

    home_shots = _safe_mean(home_rows, "HomeShots", _safe_mean(season_df, "HomeShots", 12.0))
    away_shots = _safe_mean(away_rows, "AwayShots", _safe_mean(season_df, "AwayShots", 11.0))
    home_sot = _safe_mean(home_rows, "HomeShotsOnTarget", _safe_mean(season_df, "HomeShotsOnTarget", 4.5))
    away_sot = _safe_mean(away_rows, "AwayShotsOnTarget", _safe_mean(season_df, "AwayShotsOnTarget", 4.0))
    home_corners = _safe_mean(home_rows, "HomeCorners", _safe_mean(season_df, "HomeCorners", 5.0))
    away_corners = _safe_mean(away_rows, "AwayCorners", _safe_mean(season_df, "AwayCorners", 4.5))
    home_fouls = _safe_mean(home_rows, "HomeFouls", _safe_mean(season_df, "HomeFouls", 11.0))
    away_fouls = _safe_mean(away_rows, "AwayFouls", _safe_mean(season_df, "AwayFouls", 11.0))
    home_yellow = _safe_mean(home_rows, "HY", _safe_mean(season_df, "HY", 2.0))
    away_yellow = _safe_mean(away_rows, "AY", _safe_mean(season_df, "AY", 2.0))
    home_red = _safe_mean(home_rows, "HR", _safe_mean(season_df, "HR", 0.08))
    away_red = _safe_mean(away_rows, "AR", _safe_mean(season_df, "AR", 0.08))

    home_form = _team_form_points(home_rows, "home")
    away_form = _team_form_points(away_rows, "away")

    home_avg_scored = _safe_mean(home_rows, "FTHG", _safe_mean(season_df, "FTHG", 1.4))
    home_avg_conceded = _safe_mean(home_rows, "FTAG", _safe_mean(season_df, "FTAG", 1.2))
    away_avg_scored = _safe_mean(away_rows, "FTAG", _safe_mean(season_df, "FTAG", 1.2))
    away_avg_conceded = _safe_mean(away_rows, "FTHG", _safe_mean(season_df, "FTHG", 1.4))

    home_goal_diff_trend = _safe_mean(home_rows.tail(5).assign(diff=lambda x: x["FTHG"] - x["FTAG"]), "diff", 0.0)
    away_goal_diff_trend = _safe_mean(away_rows.tail(5).assign(diff=lambda x: x["FTAG"] - x["FTHG"]), "diff", 0.0)

    total_matches = float(max(len(season_df), 1))
    match_number = float(min(len(home_rows), len(away_rows)) + 1)
    season_progress = min(match_number / total_matches, 1.0)

    if "FullTimeResult" in season_df.columns:
        season_avg_points_home = float(np.mean([
            _points_from_result(r, "home") for r in season_df["FullTimeResult"]
        ]))
        season_avg_points_away = float(np.mean([
            _points_from_result(r, "away") for r in season_df["FullTimeResult"]
        ]))
    else:
        season_avg_points_home = 1.35
        season_avg_points_away = 1.10

    points_home_before = _team_total_points(season_df, home_team)
    points_away_before = _team_total_points(season_df, away_team)
    points_gap = points_home_before - points_away_before

    home_rank = _league_table_rank(season_df, home_team)
    relegation_pressure_home = 1.0 if home_rank >= 16 else 0.0

    feature_values = {
        "HomeShots": home_shots,
        "AwayShots": away_shots,
        "HomeShotsOnTarget": home_sot,
        "AwayShotsOnTarget": away_sot,
        "HomeCorners": home_corners,
        "AwayCorners": away_corners,
        "HomeFouls": home_fouls,
        "AwayFouls": away_fouls,
        "HY": home_yellow,
        "AY": away_yellow,
        "HR": home_red,
        "AR": away_red,
        "HomeForm": home_form,
        "AwayForm": away_form,
        "FormDiff": home_form - away_form,
        "HomeAdvantage": 1.0,
        "ShotsDiff": home_shots - away_shots,
        "ShotsOnTargetDiff": home_sot - away_sot,
        "FoulsDiff": home_fouls - away_fouls,
        "YellowDiff": home_yellow - away_yellow,
        "RedDiff": home_red - away_red,
        "HomeAvgGoalsScored": home_avg_scored,
        "HomeAvgGoalsConceded": home_avg_conceded,
        "AwayAvgGoalsScored": away_avg_scored,
        "AwayAvgGoalsConceded": away_avg_conceded,
        "HomeGoalDiffTrend": home_goal_diff_trend,
        "AwayGoalDiffTrend": away_goal_diff_trend,
        "MatchNumber": match_number,
        "TotalMatches": total_matches,
        "SeasonProgress": season_progress,
        "SeasonAvgPoints_Home": season_avg_points_home,
        "SeasonAvgPoints_Away": season_avg_points_away,
        "Points_HomeBefore": points_home_before,
        "Points_AwayBefore": points_away_before,
        "PointsGap_HomeAway": points_gap,
        "RelegationPressure_Home": relegation_pressure_home,
    }

    if hasattr(scaler, "feature_names_in_"):
        ordered_columns = [str(c) for c in scaler.feature_names_in_]
    else:
        ordered_columns = MODEL_FEATURE_ORDER

    row = {name: float(feature_values.get(name, 0.0)) for name in ordered_columns}
    feature_df = pd.DataFrame([row], columns=ordered_columns)
    model_input = scaler.transform(feature_df)
    return np.asarray(model_input, dtype=np.float32)


def _poisson_pmf(k, lam):
    lam = max(float(lam), 0.05)
    return (math.exp(-lam) * (lam ** k)) / math.factorial(k)


def compute_poisson_probabilities(season_data, home_team, away_team, max_goals=7):
    home_rows = season_data[season_data["HomeTeam"] == home_team]
    away_rows = season_data[season_data["AwayTeam"] == away_team]

    league_home_goals = _safe_mean(season_data, "FTHG", 1.4)
    league_away_goals = _safe_mean(season_data, "FTAG", 1.1)

    home_avg_scored = _safe_mean(home_rows, "FTHG", league_home_goals)
    home_avg_conceded = _safe_mean(home_rows, "FTAG", league_away_goals)
    away_avg_scored = _safe_mean(away_rows, "FTAG", league_away_goals)
    away_avg_conceded = _safe_mean(away_rows, "FTHG", league_home_goals)

    home_attack = home_avg_scored / max(league_home_goals, 0.05)
    home_defense = home_avg_conceded / max(league_away_goals, 0.05)
    away_attack = away_avg_scored / max(league_away_goals, 0.05)
    away_defense = away_avg_conceded / max(league_home_goals, 0.05)

    expected_home_goals = league_home_goals * home_attack * away_defense
    expected_away_goals = league_away_goals * away_attack * home_defense

    home_win = 0.0
    draw = 0.0
    away_win = 0.0

    for hg in range(max_goals + 1):
        p_h = _poisson_pmf(hg, expected_home_goals)
        for ag in range(max_goals + 1):
            p_a = _poisson_pmf(ag, expected_away_goals)
            p = p_h * p_a
            if hg > ag:
                home_win += p
            elif hg == ag:
                draw += p
            else:
                away_win += p

    probs = np.asarray([home_win, draw, away_win], dtype=np.float32)
    probs = probs / max(probs.sum(), 1e-8)
    return probs


def compute_rating_probabilities(season_data, home_team, away_team):
    teams = sorted(set(season_data["HomeTeam"]).union(set(season_data["AwayTeam"])))
    league_avg_ppm = 1.35
    all_ppm = []
    for t in teams:
        m = _team_total_matches(season_data, t)
        if m > 0:
            all_ppm.append(_team_total_points(season_data, t) / m)
    if all_ppm:
        league_avg_ppm = float(np.mean(all_ppm))

    def smoothed_ppm(team):
        m = _team_total_matches(season_data, team)
        p = _team_total_points(season_data, team)
        # Bayesian smoothing to avoid extreme ratings with few matches.
        return float((p + 8 * league_avg_ppm) / max(m + 8, 1))

    home_ppm = smoothed_ppm(home_team)
    away_ppm = smoothed_ppm(away_team)
    strength_diff = home_ppm - away_ppm

    # Add modest home-field advantage in rating space.
    home_advantage = 0.22
    margin = strength_diff + home_advantage

    # Convert margin to non-draw split.
    home_non_draw = 1.0 / (1.0 + np.exp(-2.2 * margin))
    away_non_draw = 1.0 - home_non_draw

    # Draw chance decreases as mismatch grows.
    draw_prob = 0.32 * np.exp(-1.10 * abs(margin))
    draw_prob = float(np.clip(draw_prob, 0.18, 0.34))

    remainder = 1.0 - draw_prob
    home_prob = remainder * home_non_draw
    away_prob = remainder * away_non_draw

    probs = np.asarray([home_prob, draw_prob, away_prob], dtype=np.float32)
    return probs / max(float(probs.sum()), 1e-8)


# 3. BUILD SLIDER HTML

logos_html = ""
slider_teams = list(set(logo_map.values())) 
for url in slider_teams:
    logos_html += f'<img src="{url}" class="slider-logo">'

full_slider_html = logos_html + logos_html

# 4. HERO SECTION

st.markdown(f"""
<div class="hero-section">
<h1 class="hero-title">PREMIERVISION</h1>
<p class="hero-subtitle">NEXT GEN AI MATCH FORECASTING</p>
<div class="slider-container">
<div class="slider-track">
{full_slider_html}
</div>
</div>
<div class="scroll-indicator">
<span class="scroll-text">INITIALIZE SYSTEM</span>
<div class="scroll-arrow"></div>
</div>
</div>
""", unsafe_allow_html=True)



# 5. MAIN APPLICATION


st.markdown('<div class="main-interface">', unsafe_allow_html=True)

# Dataset Selection
season_list = sorted(df['Season'].unique(), reverse=True)
c_sel_1, c_sel_2, c_sel_3 = st.columns([1,2,1])
with c_sel_2:
    st.markdown("<div style='text-align:center; margin-bottom: 5px; color: #5e6c84; letter-spacing: 2px; font-weight:bold;'>SELECT DATA PARAMETERS</div>", unsafe_allow_html=True)
    selected_season = st.selectbox("SELECT SEASON", season_list, label_visibility="collapsed")

# Filter
season_data = df[df['Season'] == selected_season]
all_teams = sorted(pd.concat([season_data['HomeTeam'], season_data['AwayTeam']]).unique())

st.write("---")

# --- MATCHUP ---
c1, c2, c3 = st.columns([1, 0.2, 1])

with c1:
    st.markdown('<div style="color:#0052cc; letter-spacing:2px; font-weight:900; margin-bottom:10px; text-align:center;">HOME TEAM</div>', unsafe_allow_html=True)
    home_team = st.selectbox("Select Home Team", all_teams, key="home", label_visibility="collapsed")
    st.markdown(f"""<div style='display: flex; justify-content: center; margin-top: 20px; filter: drop-shadow(0 5px 15px rgba(0,0,0,0.1));'><img src='{get_logo(home_team)}' width='130'></div>""", unsafe_allow_html=True)

with c2:
    st.markdown("<br><br><br><h1 style='text-align: center; color: #172b4d; font-size: 30px; opacity: 0.5;'>VS</h1>", unsafe_allow_html=True)

with c3:
    st.markdown('<div style="color:#6b38c7; letter-spacing:2px; font-weight:900; margin-bottom:10px; text-align:center;">AWAY TEAM</div>', unsafe_allow_html=True)
    away_team = st.selectbox("Select Away Team", all_teams, key="away", index=min(1, len(all_teams)-1), label_visibility="collapsed")
    st.markdown(f"""<div style='display: flex; justify-content: center; margin-top: 20px; filter: drop-shadow(0 5px 15px rgba(0,0,0,0.1));'><img src='{get_logo(away_team)}' width='130'></div>""", unsafe_allow_html=True)

st.write("")
st.write("")

# Button Logic
col_btn_1, col_btn_2, col_btn_3 = st.columns([5, 2, 5])

with col_btn_2:
    if st.button("RUN PREDICTION ALGORITHM"):
        if home_team == away_team:
            st.warning("CONFLICT: SAME TEAM SELECTED")
        else:
            with st.spinner("PROCESSING NEURAL LAYERS..."):
                time.sleep(1.2)

                model_input = build_model_input_from_stats(season_data, home_team, away_team)
                model_probs = np.asarray(model.predict(model_input)[0], dtype=np.float32)
                if model_probs.shape[0] != 3 or not np.isfinite(model_probs).all():
                    model_probs = np.asarray([1/3, 1/3, 1/3], dtype=np.float32)
                else:
                    model_probs = model_probs / max(model_probs.sum(), 1e-8)

                # Prevent pathological overconfidence from out-of-distribution feature inputs.
                if float(model_probs.max()) > 0.9:
                    model_probs = 0.65 * model_probs + 0.35 * np.asarray([1/3, 1/3, 1/3], dtype=np.float32)

                poisson_probs = compute_poisson_probabilities(season_data, home_team, away_team)
                rating_probs = compute_rating_probabilities(season_data, home_team, away_team)

                # Hybrid forecast balances rating, Poisson, and model signal.
                prediction = 0.55 * rating_probs + 0.35 * poisson_probs + 0.10 * model_probs
                balance = 1.0 - abs(float(rating_probs[0] - rating_probs[2]))
                draw_uplift = 0.08 * max(balance, 0.0)
                prediction[1] = prediction[1] + draw_uplift
                prediction = prediction / max(prediction.sum(), 1e-8)
                st.session_state['prediction'] = prediction
                st.session_state['prediction_debug'] = {
                    'model_probs': model_probs,
                    'poisson_probs': poisson_probs,
                    'rating_probs': rating_probs,
                    'final_probs': prediction,
                    'home_team': home_team,
                    'away_team': away_team,
                    'season': selected_season,
                }

# --- RESULTS SECTION ---
if st.session_state['prediction'] is not None:
    st.write("")
    st.write("")
    st.write("")
    
    prediction = st.session_state['prediction']
    idx = np.argmax(prediction)
    confidence = prediction[idx] * 100
    outcomes = ["HOME WIN", "DRAW", "AWAY WIN"]
    result_text = outcomes[idx]
    
    # Updated Colors for Light Theme
    # Home (Blue), Away (Purple), Draw (Dark Gray)
    res_color = "#0052cc" if idx == 0 else "#6b38c7" if idx == 2 else "#172b4d"

    # Split Layout for Results
    res_c1, res_c2 = st.columns(2)

    with res_c1:
        st.markdown(f"""
            <div class="result-hud">
                <p style="font-size: 14px; letter-spacing: 4px; margin-bottom: 20px; opacity: 0.6; color:#172b4d; font-weight:bold;">ANALYSIS COMPLETE</p>
                <h1 style="font-size: 70px; margin: 0; color: {res_color}; text-shadow: 0 0 40px rgba(255,255,255,0.8); white-space: nowrap;">
                    {result_text}
                </h1>
                <p style="font-size: 20px; margin-top: 20px; color: #5e6c84;">PROBABILITY: <b style="color:#172b4d;">{confidence:.1f}%</b></p>
            </div>
        """, unsafe_allow_html=True)

    with res_c2:
        chart_data = pd.DataFrame({
            'Outcome': ['HOME', 'DRAW', 'AWAY'],
            'Probability': [prediction[0], prediction[1], prediction[2]],
            'Color': ['#0052cc', '#97a0af', '#6b38c7'] # Blue, Gray, Purple
        })

        c = alt.Chart(chart_data).mark_bar(cornerRadius=10).encode(
            x=alt.X('Outcome', sort=None, axis=alt.Axis(labels=True, title=None, labelColor='#172b4d', labelFontSize=14, labelFont='Rajdhani')),
            y=alt.Y('Probability', axis=alt.Axis(format='%', title=None, labelColor='#5e6c84', labelFontSize=12)),
            color=alt.Color('Color', scale=None),
            tooltip=['Outcome', alt.Tooltip('Probability', format='.1%')]
        ).properties(
            height=400 
        ).configure_view(
            stroke='transparent',
            fill='rgba(255,255,255,0.5)' # Semi-transparent white view background
        ).configure_axis(
            gridColor='#ebecf0' # Light gray grid lines
        ).configure(
            background='transparent' # Transparent overall chart background
        )

        st.altair_chart(c, use_container_width=True)

    debug_payload = st.session_state.get('prediction_debug')
    if debug_payload is not None:
        with st.expander("Show Prediction Debug (Model vs Poisson vs Final)"):
            st.caption(
                f"Season: {debug_payload['season']} | Match: {debug_payload['home_team']} vs {debug_payload['away_team']}"
            )
            debug_df = pd.DataFrame({
                'Outcome': ['HOME WIN', 'DRAW', 'AWAY WIN'],
                'Model': np.round(np.asarray(debug_payload['model_probs']) * 100, 2),
                'Rating': np.round(np.asarray(debug_payload['rating_probs']) * 100, 2),
                'Poisson': np.round(np.asarray(debug_payload['poisson_probs']) * 100, 2),
                'Final': np.round(np.asarray(debug_payload['final_probs']) * 100, 2),
            })
            st.dataframe(debug_df, use_container_width=True, hide_index=True)
            st.caption("Final = 55% Rating + 35% Poisson + 10% Model (with confidence damping when model is too sharp).")

st.markdown('</div>', unsafe_allow_html=True)