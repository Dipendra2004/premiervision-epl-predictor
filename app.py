import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import altair as alt

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

                logits = np.stack([away_score, draw_score, home_score], axis=1)
                logits = logits - logits.max(axis=1, keepdims=True)
                exp_logits = np.exp(logits)
                return exp_logits / exp_logits.sum(axis=1, keepdims=True)
        return MockModel()

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
df = load_data()


def _safe_column_mean(frame, column_name, default_value):
    if column_name not in frame.columns:
        return float(default_value)
    values = pd.to_numeric(frame[column_name], errors="coerce").dropna()
    if values.empty:
        return float(default_value)
    return float(values.mean())


def _get_season_baselines(season_data):
    default_goals = 1.2
    return {
        "gf_home": _safe_column_mean(season_data, "FTHG", default_goals),
        "ga_home": _safe_column_mean(season_data, "FTAG", default_goals),
        "sf_home": _safe_column_mean(season_data, "HomeShots", 12.0),
        "sa_home": _safe_column_mean(season_data, "AwayShots", 12.0),
        "sotf_home": _safe_column_mean(season_data, "HomeShotsOnTarget", 4.5),
        "sota_home": _safe_column_mean(season_data, "AwayShotsOnTarget", 4.5),
        "cf_home": _safe_column_mean(season_data, "HomeCorners", 5.0),
        "ca_home": _safe_column_mean(season_data, "AwayCorners", 5.0),
        "ff_home": _safe_column_mean(season_data, "HomeFouls", 11.0),
        "fa_home": _safe_column_mean(season_data, "AwayFouls", 11.0),
        "yc_home": _safe_column_mean(season_data, "HY", 2.0),
        "rc_home": _safe_column_mean(season_data, "HR", 0.08),
        "gf_away": _safe_column_mean(season_data, "FTAG", default_goals),
        "ga_away": _safe_column_mean(season_data, "FTHG", default_goals),
        "sf_away": _safe_column_mean(season_data, "AwayShots", 12.0),
        "sa_away": _safe_column_mean(season_data, "HomeShots", 12.0),
        "sotf_away": _safe_column_mean(season_data, "AwayShotsOnTarget", 4.5),
        "sota_away": _safe_column_mean(season_data, "HomeShotsOnTarget", 4.5),
        "cf_away": _safe_column_mean(season_data, "AwayCorners", 5.0),
        "ca_away": _safe_column_mean(season_data, "HomeCorners", 5.0),
        "ff_away": _safe_column_mean(season_data, "AwayFouls", 11.0),
        "fa_away": _safe_column_mean(season_data, "HomeFouls", 11.0),
        "yc_away": _safe_column_mean(season_data, "AY", 2.0),
        "rc_away": _safe_column_mean(season_data, "AR", 0.08),
        "ppm": 1.33,
    }


def _team_points_per_match(matches, is_home_side):
    if matches.empty or "FullTimeResult" not in matches.columns:
        return 1.33
    if is_home_side:
        wins = (matches["FullTimeResult"] == "H").sum()
        draws = (matches["FullTimeResult"] == "D").sum()
    else:
        wins = (matches["FullTimeResult"] == "A").sum()
        draws = (matches["FullTimeResult"] == "D").sum()
    return float((wins * 3 + draws) / max(len(matches), 1))


def _build_team_profile(season_data, team, side, baselines):
    if side == "home":
        matches = season_data[season_data["HomeTeam"] == team].copy()
        profile = {
            "gf": _safe_column_mean(matches, "FTHG", baselines["gf_home"]),
            "ga": _safe_column_mean(matches, "FTAG", baselines["ga_home"]),
            "sf": _safe_column_mean(matches, "HomeShots", baselines["sf_home"]),
            "sa": _safe_column_mean(matches, "AwayShots", baselines["sa_home"]),
            "sotf": _safe_column_mean(matches, "HomeShotsOnTarget", baselines["sotf_home"]),
            "sota": _safe_column_mean(matches, "AwayShotsOnTarget", baselines["sota_home"]),
            "cf": _safe_column_mean(matches, "HomeCorners", baselines["cf_home"]),
            "ca": _safe_column_mean(matches, "AwayCorners", baselines["ca_home"]),
            "ff": _safe_column_mean(matches, "HomeFouls", baselines["ff_home"]),
            "fa": _safe_column_mean(matches, "AwayFouls", baselines["fa_home"]),
            "yc": _safe_column_mean(matches, "HY", baselines["yc_home"]),
            "rc": _safe_column_mean(matches, "HR", baselines["rc_home"]),
            "ppm": _team_points_per_match(matches, is_home_side=True),
        }
    else:
        matches = season_data[season_data["AwayTeam"] == team].copy()
        profile = {
            "gf": _safe_column_mean(matches, "FTAG", baselines["gf_away"]),
            "ga": _safe_column_mean(matches, "FTHG", baselines["ga_away"]),
            "sf": _safe_column_mean(matches, "AwayShots", baselines["sf_away"]),
            "sa": _safe_column_mean(matches, "HomeShots", baselines["sa_away"]),
            "sotf": _safe_column_mean(matches, "AwayShotsOnTarget", baselines["sotf_away"]),
            "sota": _safe_column_mean(matches, "HomeShotsOnTarget", baselines["sota_away"]),
            "cf": _safe_column_mean(matches, "AwayCorners", baselines["cf_away"]),
            "ca": _safe_column_mean(matches, "HomeCorners", baselines["ca_away"]),
            "ff": _safe_column_mean(matches, "AwayFouls", baselines["ff_away"]),
            "fa": _safe_column_mean(matches, "HomeFouls", baselines["fa_away"]),
            "yc": _safe_column_mean(matches, "AY", baselines["yc_away"]),
            "rc": _safe_column_mean(matches, "AR", baselines["rc_away"]),
            "ppm": _team_points_per_match(matches, is_home_side=False),
        }
    return profile


def build_model_input_from_stats(season_data, home_team, away_team, feature_count=36):
    baselines = _get_season_baselines(season_data)
    home_profile = _build_team_profile(season_data, home_team, "home", baselines)
    away_profile = _build_team_profile(season_data, away_team, "away", baselines)

    home_features = [
        home_profile["gf"], home_profile["ga"], home_profile["sf"], home_profile["sa"],
        home_profile["sotf"], home_profile["sota"], home_profile["cf"], home_profile["ca"],
        home_profile["ff"], home_profile["fa"], home_profile["yc"], home_profile["rc"],
        home_profile["ppm"],
    ]
    away_features = [
        away_profile["gf"], away_profile["ga"], away_profile["sf"], away_profile["sa"],
        away_profile["sotf"], away_profile["sota"], away_profile["cf"], away_profile["ca"],
        away_profile["ff"], away_profile["fa"], away_profile["yc"], away_profile["rc"],
        away_profile["ppm"],
    ]

    matchup_features = [
        home_profile["gf"] - away_profile["ga"],
        away_profile["gf"] - home_profile["ga"],
        home_profile["ppm"] - away_profile["ppm"],
        home_profile["sotf"] - away_profile["sota"],
        away_profile["sotf"] - home_profile["sota"],
        home_profile["sf"] - away_profile["sa"],
        away_profile["sf"] - home_profile["sa"],
        home_profile["cf"] - away_profile["ca"],
        home_profile["yc"] - away_profile["yc"],
        home_profile["rc"] - away_profile["rc"],
    ]

    features = home_features + away_features + matchup_features
    if len(features) < feature_count:
        features.extend([0.0] * (feature_count - len(features)))
    elif len(features) > feature_count:
        features = features[:feature_count]

    return np.asarray([features], dtype=np.float32)


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
                prediction = model.predict(model_input)[0]
                st.session_state['prediction'] = prediction

# --- RESULTS SECTION ---
if st.session_state['prediction'] is not None:
    st.write("")
    st.write("")
    st.write("")
    
    prediction = st.session_state['prediction']
    idx = np.argmax(prediction)
    confidence = prediction[idx] * 100
    outcomes = ["AWAY WIN", "DRAW", "HOME WIN"]
    result_text = outcomes[idx]
    
    # Updated Colors for Light Theme
    # Home (Blue), Away (Purple), Draw (Dark Gray)
    res_color = "#0052cc" if idx == 2 else "#6b38c7" if idx == 0 else "#172b4d"

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
            'Probability': [prediction[2], prediction[1], prediction[0]],
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

st.markdown('</div>', unsafe_allow_html=True)