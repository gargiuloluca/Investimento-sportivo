import os
import json
import io
import math
from datetime import datetime, timezone
import requests
import pandas as pd
import numpy as np
from scipy.stats import poisson

# Campionati supportati (CSV ufficiali football-data.co.uk)
# Includono: Risultati FT, Corner (HC, AC), Cartellini Gialli (HY, AY) e Rossi (HR, AR)
LEAGUES = {
    "Serie A (Italia)": "https://www.football-data.co.uk/mmz4281/2425/I1.csv",
    "Serie B (Italia)": "https://www.football-data.co.uk/mmz4281/2425/I2.csv",
    "Premier League (Inghilterra)": "https://www.football-data.co.uk/mmz4281/2425/E0.csv",
    "La Liga (Spagna)": "https://www.football-data.co.uk/mmz4281/2425/SP1.csv",
    "Bundesliga (Germania)": "https://www.football-data.co.uk/mmz4281/2425/D1.csv",
    "Ligue 1 (Francia)": "https://www.football-data.co.uk/mmz4281/2425/F1.csv"
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

def fetch_league_data(url):
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code == 200:
            df = pd.read_csv(io.StringIO(resp.text))
            return df
    except Exception as e:
        print(f"Errore download {url}: {e}")
    return None

def calculate_bivariate_poisson(lambda_h, lambda_a, max_goals=8):
    """Calcola matrice di probabilità congiunta per il calcio (Poisson bivariata)."""
    p_h = [poisson.pmf(i, lambda_h) for i in range(max_goals + 1)]
    p_a = [poisson.pmf(j, lambda_a) for j in range(max_goals + 1)]
    
    matrix = np.outer(p_h, p_a)
    
    prob_1 = np.sum(np.tril(matrix, -1)) # h > a
    prob_x = np.sum(np.diag(matrix))     # h == a
    prob_2 = np.sum(np.triu(matrix, 1))  # h < a
    
    # Over / Under 2.5
    prob_under25 = 0.0
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            if h + a < 2.5:
                prob_under25 += matrix[h, a]
    prob_over25 = 1.0 - prob_under25
    
    # Doppia Chance
    prob_1x = prob_1 + prob_x
    prob_x2 = prob_2 + prob_x
    prob_12 = prob_1 + prob_2
    
    return {
        "1": prob_1 * 100,
        "X": prob_x * 100,
        "2": prob_2 * 100,
        "1X": prob_1x * 100,
        "X2": prob_x2 * 100,
        "12": prob_12 * 100,
        "Under 2.5": prob_under25 * 100,
        "Over 2.5": prob_over25 * 100
    }

def analyze_stats(df):
    """Estrae statistiche di attacco/difesa per gol, corner e cartellini."""
    required_cols = ["HomeTeam", "AwayTeam", "FTHG", "FTAG"]
    if not all(col in df.columns for col in required_cols):
        return None, None
        
    df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"]).copy()
    
    # Check per corner e cartellini
    has_corners = "HC" in df.columns and "AC" in df.columns
    has_cards = "HY" in df.columns and "AY" in df.columns
    
    avg_home_goals = df["FTHG"].mean()
    avg_away_goals = df["FTAG"].mean()
    avg_total_corners = (df["HC"] + df["AC"]).mean() if has_corners else 9.5
    avg_total_cards = (df["HY"] + df["AY"] + (df.get("HR", 0) + df.get("AR", 0))*2).mean() if has_cards else 4.2
    
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
        
        # Corner
        h_corners_for = h_games["HC"].sum() if has_corners else 0
        h_corners_against = h_games["AC"].sum() if has_corners else 0
        a_corners_for = a_games["AC"].sum() if has_corners else 0
        a_corners_against = a_games["HC"].sum() if has_corners else 0
        
        # Cartellini (Giallo=1 pt, Rosso=2 pt)
        h_cards_for = (h_games["HY"].sum() + h_games.get("HR", pd.Series(0)).sum() * 2) if has_cards else 0
        a_cards_for = (a_games["AY"].sum() + a_games.get("AR", pd.Series(0)).sum() * 2) if has_cards else 0
        
        team_stats[t] = {
            "games": total_games,
            "h_att": (h_scored / len(h_games)) / avg_home_goals if len(h_games) > 0 and avg_home_goals > 0 else 1.0,
            "h_def": (h_conceded / len(h_games)) / avg_away_goals if len(h_games) > 0 and avg_away_goals > 0 else 1.0,
            "a_att": (a_scored / len(a_games)) / avg_away_goals if len(a_games) > 0 and avg_away_goals > 0 else 1.0,
            "a_def": (a_conceded / len(a_games)) / avg_home_goals if len(a_games) > 0 and avg_home_goals > 0 else 1.0,
            "avg_corners_tot": ((h_corners_for + h_corners_against + a_corners_for + a_corners_against) / total_games) if has_corners else 9.5,
            "avg_cards_tot": ((h_cards_for + a_cards_for) / total_games) if has_cards else 4.2
        }
        
    league_meta = {
        "avg_home_goals": avg_home_goals,
        "avg_away_goals": avg_away_goals,
        "avg_total_corners": avg_total_corners,
        "avg_total_cards": avg_total_cards
    }
    
    return team_stats, league_meta

def predict_match(home_team, away_team, stats, meta):
    if home_team not in stats or away_team not in stats:
        return None
        
    h_s = stats[home_team]
    a_s = stats[away_team]
    
    # Lambda Poisson per Gol
    lambda_h = max(0.2, h_s["h_att"] * a_s["a_def"] * meta["avg_home_goals"])
    lambda_a = max(0.2, a_s["a_att"] * h_s["h_def"] * meta["avg_away_goals"])
    
    # Calcolo 1X2, DC, Under/Over
    poisson_markets = calculate_bivariate_poisson(lambda_h, lambda_a)
    
    # Calcolo Corner (media ponderata)
    exp_corners = (h_s["avg_corners_tot"] + a_s["avg_corners_tot"]) / 2.0
    prob_over85_corners = (1.0 - poisson.cdf(8, exp_corners)) * 100
    prob_under115_corners = (poisson.cdf(11, exp_corners)) * 100
    
    # Calcolo Cartellini (media ponderata)
    exp_cards = (h_s["avg_cards_tot"] + a_s["avg_cards_tot"]) / 2.0
    prob_over35_cards = (1.0 - poisson.cdf(3, exp_cards)) * 100
    prob_under55_cards = (poisson.cdf(5, exp_cards)) * 100
    
    all_markets = {
        "1": round(poisson_markets["1"], 1),
        "X": round(poisson_markets["X"], 1),
        "2": round(poisson_markets["2"], 1),
        "1X": round(poisson_markets["1X"], 1),
        "X2": round(poisson_markets["X2"], 1),
        "12": round(poisson_markets["12"], 1),
        "Under 2.5": round(poisson_markets["Under 2.5"], 1),
        "Over 2.5": round(poisson_markets["Over 2.5"], 1),
        "Corner Over 8.5": round(prob_over85_corners, 1),
        "Corner Under 11.5": round(prob_under115_corners, 1),
        "Cartellini Over 3.5": round(prob_over35_cards, 1),
        "Cartellini Under 5.5": round(prob_under55_cards, 1)
    }
    
    # Trova il mercato con la probabilità più alta in assoluto
    best_market = max(all_markets.items(), key=lambda x: x[1])
    
    return {
        "match": f"{home_team} - {away_team}",
        "home": home_team,
        "away": away_team,
        "exp_goals": f"{round(lambda_h, 2)} - {round(lambda_a, 2)}",
        "exp_corners": round(exp_corners, 1),
        "exp_cards": round(exp_cards, 1),
        "all_markets": all_markets,
        "top_bet": {
            "market": best_market[0],
            "probability": best_market[1]
        }
    }

def main():
    print("Inizio scansione campionati e calcolo modelli matematici...")
    full_database = []
    
    for league_name, url in LEAGUES.items():
        print(f"Scaricamento dati: {league_name}")
        df = fetch_league_data(url)
        if df is None or len(df) < 10:
            continue
            
        stats, meta = analyze_stats(df)
        if not stats:
            continue
            
        # Simula/analizza tutti gli accoppiamenti testa a testa del campionato
        teams = list(stats.keys())
        for i in range(len(teams)):
            for j in range(len(teams)):
                if i != j:
                    h_team = teams[i]
                    a_team = teams[j]
                    pred = predict_match(h_team, a_team, stats, meta)
                    if pred:
                        pred["league"] = league_name
                        full_database.append(pred)
                        
    # Ordina l'intero database in base alla probabilità della Top Bet (singola più probabile in assoluto)
    full_database.sort(key=lambda x: x["top_bet"]["probability"], reverse=True)
    
    # Estrai le TOP 50 partite a più alta probabilità in assoluto
    high_probability_picks = [m for m in full_database if m["top_bet"]["probability"] >= 75.0][:50]
    
    output_data = {
        "last_update": datetime.now(timezone.utc).strftime("%d/%m/%Y %H:%M UTC"),
        "total_analyzed": len(full_database),
        "high_prob_count": len(high_probability_picks),
        "matches": high_probability_picks
    }
    
    os.makedirs("data", exist_ok=True)
    with open("data/predictions.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
        
    print(f"Completato con successo! Salvate {len(high_probability_picks)} selezioni ad altissima probabilità.")

if __name__ == "__main__":
    main()
