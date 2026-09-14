import os
import json
import io
from datetime import datetime, timezone
import requests
import pandas as pd
import numpy as np
from scipy.stats import poisson

# Feed statistiche storiche per calcolo forza attacco/difesa
LEAGUE_HISTORICAL = {
    "Serie A (Italia)": "https://www.football-data.co.uk/mmz4281/2425/I1.csv",
    "Serie B (Italia)": "https://www.football-data.co.uk/mmz4281/2425/I2.csv",
    "Premier League (Inghilterra)": "https://www.football-data.co.uk/mmz4281/2425/E0.csv",
    "La Liga (Spagna)": "https://www.football-data.co.uk/mmz4281/2425/SP1.csv",
    "Bundesliga (Germania)": "https://www.football-data.co.uk/mmz4281/2425/D1.csv",
    "Ligue 1 (Francia)": "https://www.football-data.co.uk/mmz4281/2425/F1.csv"
}

# Codici campionato per filtrare le prossime partite dal file fixtures
DIVISION_MAPPING = {
    "I1": "Serie A (Italia)",
    "I2": "Serie B (Italia)",
    "E0": "Premier League (Inghilterra)",
    "SP1": "La Liga (Spagna)",
    "D1": "Bundesliga (Germania)",
    "F1": "Ligue 1 (Francia)"
}

FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

def fetch_csv(url):
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code == 200:
            return pd.read_csv(io.StringIO(resp.text))
    except Exception as e:
        print(f"Errore caricamento {url}: {e}")
    return None

def calculate_bivariate_poisson(lambda_h, lambda_a, max_goals=8):
    p_h = [poisson.pmf(i, lambda_h) for i in range(max_goals + 1)]
    p_a = [poisson.pmf(j, lambda_a) for j in range(max_goals + 1)]
    matrix = np.outer(p_h, p_a)
    
    prob_1 = np.sum(np.tril(matrix, -1))
    prob_x = np.sum(np.diag(matrix))
    prob_2 = np.sum(np.triu(matrix, 1))
    
    prob_under25 = 0.0
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            if h + a < 2.5:
                prob_under25 += matrix[h, a]
    prob_over25 = 1.0 - prob_under25
    
    return {
        "1": prob_1 * 100,
        "X": prob_x * 100,
        "2": prob_2 * 100,
        "1X": (prob_1 + prob_x) * 100,
        "X2": (prob_2 + prob_x) * 100,
        "12": (prob_1 + prob_2) * 100,
        "Under 2.5": prob_under25 * 100,
        "Over 2.5": prob_over25 * 100
    }

def build_league_profiles():
    league_data = {}
    for league_name, url in LEAGUE_HISTORICAL.items():
        df = fetch_csv(url)
        if df is None or len(df) < 10:
            continue
            
        df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"]).copy()
        avg_home_goals = df["FTHG"].mean()
        avg_away_goals = df["FTAG"].mean()
        
        has_corners = "HC" in df.columns and "AC" in df.columns
        has_cards = "HY" in df.columns and "AY" in df.columns
        
        teams = sorted(list(set(df["HomeTeam"].unique()) | set(df["AwayTeam"].unique())))
        team_stats = {}
        
        for t in teams:
            h_games = df[df["HomeTeam"] == t]
            a_games = df[df["AwayTeam"] == t]
            total_games = len(h_games) + len(a_games)
            if total_games < 3:
                continue
                
            h_scored = h_games["FTHG"].sum()
            h_conceded = h_games["FTAG"].sum()
            a_scored = a_games["FTAG"].sum()
            a_conceded = a_games["FTHG"].sum()
            
            h_corners = (h_games["HC"].sum() + h_games["AC"].sum()) if has_corners else 0
            a_corners = (a_games["AC"].sum() + a_games["HC"].sum()) if has_corners else 0
            
            h_cards = (h_games["HY"].sum() + h_games.get("HR", pd.Series(0)).sum() * 2) if has_cards else 0
            a_cards = (a_games["AY"].sum() + a_games.get("AR", pd.Series(0)).sum() * 2) if has_cards else 0
            
            team_stats[t] = {
                "h_att": (h_scored / len(h_games)) / avg_home_goals if len(h_games) > 0 and avg_home_goals > 0 else 1.0,
                "h_def": (h_conceded / len(h_games)) / avg_away_goals if len(h_games) > 0 and avg_away_goals > 0 else 1.0,
                "a_att": (a_scored / len(a_games)) / avg_away_goals if len(a_games) > 0 and avg_away_goals > 0 else 1.0,
                "a_def": (a_conceded / len(a_games)) / avg_home_goals if len(a_games) > 0 and avg_home_goals > 0 else 1.0,
                "avg_corners": ((h_corners + a_corners) / total_games) if has_corners else 9.5,
                "avg_cards": ((h_cards + a_cards) / total_games) if has_cards else 4.2
            }
            
        league_data[league_name] = {
            "teams": team_stats,
            "avg_home_goals": avg_home_goals,
            "avg_away_goals": avg_away_goals
        }
    return league_data

def main():
    print("Elaborazione profili statistici e lettura palinsesto partite future...")
    leagues_profile = build_league_profiles()
    fixtures_df = fetch_csv(FIXTURES_URL)
    
    upcoming_predictions = []
    
    if fixtures_df is not None and not fixtures_df.empty:
        fixtures_df = fixtures_df.dropna(subset=["Div", "HomeTeam", "AwayTeam"])
        
        for _, row in fixtures_df.iterrows():
            div = str(row["Div"]).strip()
            if div not in DIVISION_MAPPING:
                continue
                
            league_name = DIVISION_MAPPING[div]
            h_team = str(row["HomeTeam"]).strip()
            a_team = str(row["AwayTeam"]).strip()
            date_match = str(row.get("Date", "")).strip()
            time_match = str(row.get("Time", "")).strip()
            
            profile = leagues_profile.get(league_name)
            if not profile:
                continue
                
            t_stats = profile["teams"]
            if h_team not in t_stats or a_team not in t_stats:
                continue
                
            h_s = t_stats[h_team]
            a_s = t_stats[a_team]
            
            lambda_h = max(0.2, h_s["h_att"] * a_s["a_def"] * profile["avg_home_goals"])
            lambda_a = max(0.2, a_s["a_att"] * h_s["h_def"] * profile["avg_away_goals"])
            
            poisson_mk = calculate_bivariate_poisson(lambda_h, lambda_a)
            
            exp_corners = (h_s["avg_corners"] + a_s["avg_corners"]) / 2.0
            prob_over85_corners = (1.0 - poisson.cdf(8, exp_corners)) * 100
            prob_under115_corners = (poisson.cdf(11, exp_corners)) * 100
            
            exp_cards = (h_s["avg_cards"] + a_s["avg_cards"]) / 2.0
            prob_over35_cards = (1.0 - poisson.cdf(3, exp_cards)) * 100
            prob_under55_cards = (poisson.cdf(5, exp_cards)) * 100
            
            all_markets = {
                "1": round(poisson_mk["1"], 1),
                "X": round(poisson_mk["X"], 1),
                "2": round(poisson_mk["2"], 1),
                "1X": round(poisson_mk["1X"], 1),
                "X2": round(poisson_mk["X2"], 1),
                "12": round(poisson_mk["12"], 1),
                "Under 2.5": round(poisson_mk["Under 2.5"], 1),
                "Over 2.5": round(poisson_mk["Over 2.5"], 1),
                "Corner Over 8.5": round(prob_over85_corners, 1),
                "Corner Under 11.5": round(prob_under115_corners, 1),
                "Cartellini Over 3.5": round(prob_over35_cards, 1),
                "Cartellini Under 5.5": round(prob_under55_cards, 1)
            }
            
            best_market = max(all_markets.items(), key=lambda x: x[1])
            
            match_str = f"{h_team} - {a_team}"
            if date_match:
                match_str += f" ({date_match} {time_match})".strip()
                
            upcoming_predictions.append({
                "league": league_name,
                "match": match_str,
                "home": h_team,
                "away": a_team,
                "exp_goals": f"{round(lambda_h, 2)} - {round(lambda_a, 2)}",
                "exp_corners": round(exp_corners, 1),
                "exp_cards": round(exp_cards, 1),
                "all_markets": all_markets,
                "top_bet": {
                    "market": best_market[0],
                    "probability": best_market[1]
                }
            })

    # Ordina per probabilità più alta
    upcoming_predictions.sort(key=lambda x: x["top_bet"]["probability"], reverse=True)
    
    output_data = {
        "last_update": datetime.now(timezone.utc).strftime("%d/%m/%Y %H:%M UTC"),
        "total_analyzed": len(upcoming_predictions),
        "high_prob_count": len(upcoming_predictions),
        "matches": upcoming_predictions
    }
    
    os.makedirs("data", exist_ok=True)
    with open("data/predictions.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
        
    print(f"Salvate {len(upcoming_predictions)} partite reali del palinsesto futuro.")

if __name__ == "__main__":
    main()
    
