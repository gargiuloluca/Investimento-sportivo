import os
import json
import io
from datetime import datetime, timezone
import requests
import pandas as pd
import numpy as np
from scipy.stats import poisson

LEAGUE_HISTORICAL = {
    "Serie A (Italia)": "https://www.football-data.co.uk/mmz4281/2425/I1.csv",
    "Serie B (Italia)": "https://www.football-data.co.uk/mmz4281/2425/I2.csv",
    "Premier League (Inghilterra)": "https://www.football-data.co.uk/mmz4281/2425/E0.csv",
    "La Liga (Spagna)": "https://www.football-data.co.uk/mmz4281/2425/SP1.csv",
    "Bundesliga (Germania)": "https://www.football-data.co.uk/mmz4281/2425/D1.csv",
    "Ligue 1 (Francia)": "https://www.football-data.co.uk/mmz4281/2425/F1.csv"
}

DIVISION_MAPPING = {
    "I1": "Serie A (Italia)",
    "I2": "Serie B (Italia)",
    "E0": "Premier League (Inghilterra)",
    "SP1": "La Liga (Spagna)",
    "D1": "Bundesliga (Germania)",
    "F1": "Ligue 1 (Francia)"
}

FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
HEADERS = {"User-Agent": "Mozilla/5.0"}

def fetch_csv(url):
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        if resp.status_code == 200:
            return pd.read_csv(io.StringIO(resp.content.decode('latin-1')))
    except Exception as e:
        print(f"Errore download {url}: {e}")
    return None

def calculate_bivariate_poisson(lambda_h, lambda_a, max_goals=8):
    p_h = [poisson.pmf(i, lambda_h) for i in range(max_goals + 1)]
    p_a = [poisson.pmf(j, lambda_a) for j in range(max_goals + 1)]
    matrix = np.outer(p_h, p_a)
    
    prob_1 = float(np.sum(np.tril(matrix, -1)))
    prob_x = float(np.sum(np.diag(matrix)))
    prob_2 = float(np.sum(np.triu(matrix, 1)))
    
    prob_under25 = sum(float(matrix[h, a]) for h in range(max_goals + 1) for a in range(max_goals + 1) if h + a < 2.5)
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
        req_cols = ["HomeTeam", "AwayTeam", "FTHG", "FTAG"]
        if not all(c in df.columns for c in req_cols):
            continue
            
        df = df.dropna(subset=req_cols).copy()
        avg_h_goals = float(df["FTHG"].mean())
        avg_a_goals = float(df["FTAG"].mean())
        has_corners = "HC" in df.columns and "AC" in df.columns
        has_cards = "HY" in df.columns and "AY" in df.columns
        
        teams = sorted(list(set(df["HomeTeam"].unique()) | set(df["AwayTeam"].unique())))
        team_stats = {}
        for t in teams:
            h_g = df[df["HomeTeam"] == t]
            a_g = df[df["AwayTeam"] == t]
            tot = len(h_g) + len(a_g)
            if tot < 2:
                continue
            h_sc, h_co = h_g["FTHG"].sum(), h_g["FTAG"].sum()
            a_sc, a_co = a_g["FTAG"].sum(), a_g["FTHG"].sum()
            h_crn = (h_g["HC"].sum() + h_g["AC"].sum()) if has_corners else 0
            a_crn = (a_g["AC"].sum() + a_g["HC"].sum()) if has_corners else 0
            h_crd = (h_g["HY"].sum() + h_g.get("HR", pd.Series(0)).sum()*2) if has_cards else 0
            a_crd = (a_g["AY"].sum() + a_g.get("AR", pd.Series(0)).sum()*2) if has_cards else 0
            
            team_stats[t] = {
                "h_att": (h_sc / len(h_g)) / avg_h_goals if len(h_g) > 0 and avg_h_goals > 0 else 1.0,
                "h_def": (h_co / len(h_g)) / avg_a_goals if len(h_g) > 0 and avg_a_goals > 0 else 1.0,
                "a_att": (a_sc / len(a_g)) / avg_a_goals if len(a_g) > 0 and avg_a_goals > 0 else 1.0,
                "a_def": (a_co / len(a_g)) / avg_h_goals if len(a_g) > 0 and avg_h_goals > 0 else 1.0,
                "avg_corners": ((h_crn + a_crn) / tot) if has_corners else 9.5,
                "avg_cards": ((h_crd + a_crd) / tot) if has_cards else 4.2
            }
        league_data[league_name] = {
            "teams": team_stats,
            "avg_h_goals": avg_h_goals,
            "avg_a_goals": avg_a_goals
        }
    return league_data

def evaluate_match(league_name, h_team, a_team, date_str, time_str, profile):
    t_stats = profile["teams"]
    if h_team not in t_stats or a_team not in t_stats:
        return None
    h_s, a_s = t_stats[h_team], t_stats[a_team]
    
    lambda_h = max(0.2, h_s["h_att"] * a_s["a_def"] * profile["avg_h_goals"])
    lambda_a = max(0.2, a_s["a_att"] * h_s["h_def"] * profile["avg_a_goals"])
    poisson_mk = calculate_bivariate_poisson(lambda_h, lambda_a)
    
    exp_corners = (h_s["avg_corners"] + a_s["avg_corners"]) / 2.0
    prob_over85_corners = (1.0 - poisson.cdf(8, exp_corners)) * 100
    prob_under115_corners = (poisson.cdf(11, exp_corners)) * 100
    
    exp_cards = (h_s["avg_cards"] + a_s["avg_cards"]) / 2.0
    prob_over35_cards = (1.0 - poisson.cdf(3, exp_cards)) * 100
    
    # Lista mercati attivi: esclusi volutamente Under Cartellini
    raw_markets = {
        "1": poisson_mk["1"],
        "X": poisson_mk["X"],
        "2": poisson_mk["2"],
        "1X": poisson_mk["1X"],
        "X2": poisson_mk["X2"],
        "12": poisson_mk["12"],
        "Under 2.5": poisson_mk["Under 2.5"],
        "Over 2.5": poisson_mk["Over 2.5"],
        "Corner Over 8.5": prob_over85_corners,
        "Corner Under 11.5": prob_under115_corners,
        "Cartellini Over 3.5": prob_over35_cards
    }
    
    all_markets = {}
    eligible_singles = []
    eligible_multipla = []
    
    for m_name, prob in raw_markets.items():
        if prob <= 0:
            continue
        est_odds = round(100.0 / prob, 2)
        all_markets[m_name] = {"prob": round(prob, 1), "odds": est_odds}
        
        # Filtro Singola: quota >= 1.50
        if est_odds >= 1.50:
            eligible_singles.append((m_name, prob, est_odds))
        # Filtro Multipla: quota >= 1.30
        if est_odds >= 1.30:
            eligible_multipla.append((m_name, prob, est_odds))
            
    # Migliore singola per questa partita (con quota >= 1.50)
    top_single = max(eligible_singles, key=lambda x: x[1]) if eligible_singles else None
    top_multi_component = max(eligible_multipla, key=lambda x: x[1]) if eligible_multipla else None
    
    match_label = f"{h_team} - {a_team}"
    if date_str:
        match_label += f" ({date_str} {time_str})".strip()
        
    return {
        "league": league_name,
        "match": match_label,
        "home": h_team,
        "away": a_team,
        "exp_goals": f"{round(lambda_h, 2)} - {round(lambda_a, 2)}",
        "exp_corners": round(exp_corners, 1),
        "exp_cards": round(exp_cards, 1),
        "all_markets": all_markets,
        "top_single": {
            "market": top_single[0], "prob": round(top_single[1], 1), "odds": top_single[2]
        } if top_single else None,
        "top_multi_component": {
            "market": top_multi_component[0], "prob": round(top_multi_component[1], 1), "odds": top_multi_component[2]
        } if top_multi_component else None
    }

def generate_multipla(matches, target_odds=2.50, max_legs=4):
    """Combina da 2 a 4 selezioni indipendenti con quota >= 1.30 fino a raggiungere circa 2.50."""
    candidates = []
    for m in matches:
        if m.get("top_multi_component"):
            candidates.append({
                "match": m["match"],
                "league": m["league"],
                "market": m["top_multi_component"]["market"],
                "prob": m["top_multi_component"]["prob"],
                "odds": m["top_multi_component"]["odds"]
            })
            
    # Ordina per affidabilità percentuale
    candidates.sort(key=lambda x: x["prob"], reverse=True)
    
    selected_legs = []
    current_multiplier = 1.0
    
    for c in candidates:
        if len(selected_legs) >= max_legs:
            break
        projected = current_multiplier * c["odds"]
        if projected <= (target_odds + 0.35):
            selected_legs.append(c)
            current_multiplier = projected
            if current_multiplier >= (target_odds - 0.20):
                break
                
    if len(selected_legs) < 2 and len(candidates) >= 2:
        selected_legs = candidates[:2]
        current_multiplier = round(selected_legs[0]["odds"] * selected_legs[1]["odds"], 2)
        
    return {
        "legs": selected_legs,
        "total_odds": round(current_multiplier, 2)
    }

def main():
    leagues_profile = build_league_profiles()
    fixtures_df = fetch_csv(FIXTURES_URL)
    matches = []
    
    if fixtures_df is not None and not fixtures_df.empty:
        valid_cols = ["Div", "HomeTeam", "AwayTeam"]
        if all(c in fixtures_df.columns for c in valid_cols):
            for _, row in fixtures_df.iterrows():
                div = str(row["Div"]).strip()
                if div not in DIVISION_MAPPING:
                    continue
                league_name = DIVISION_MAPPING[div]
                h_t = str(row["HomeTeam"]).strip()
                a_t = str(row["AwayTeam"]).strip()
                d_str = str(row.get("Date", "")).replace("nan", "").strip()
                t_str = str(row.get("Time", "")).replace("nan", "").strip()
                
                profile = leagues_profile.get(league_name)
                if profile:
                    pred = evaluate_match(league_name, h_t, a_t, d_str, t_str, profile)
                    if pred:
                        matches.append(pred)
                        
    if not matches:
        for league_name, profile in leagues_profile.items():
            teams = list(profile["teams"].keys())
            for i in range(0, len(teams) - 1, 2):
                pred = evaluate_match(league_name, teams[i], teams[i+1], "Prossimo Turno", "", profile)
                if pred:
                    matches.append(pred)

    # Ordina le partite per probabilità della migliore singola valida (quota >= 1.50)
    matches.sort(key=lambda x: (x["top_single"]["prob"] if x["top_single"] else 0), reverse=True)
    
    # Genera la schedina multipla (quota totale ~ 2.50)
    multipla = generate_multipla(matches, target_odds=2.50, max_legs=4)
    
    output_data = {
        "last_update": datetime.now(timezone.utc).strftime("%d/%m/%Y %H:%M UTC"),
        "multipla_consigliata": multipla,
        "matches": matches
    }
    
    os.makedirs("data", exist_ok=True)
    with open("data/predictions.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print("Elaborazione completata con successo.")

if __name__ == "__main__":
    main()
