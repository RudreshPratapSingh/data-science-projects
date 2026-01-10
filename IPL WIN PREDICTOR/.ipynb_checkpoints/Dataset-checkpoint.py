#!/usr/bin/env python3
"""
build_ipl_live_states_full.py

Builds a ball-by-ball CSV with live-state features for IPL matches (2008-2025).
Outputs: ./ipl_dataset/ipl_live_states_2008_2025.csv

Usage:
    python build_ipl_live_states_full.py
If automatic download fails, manually place the Cricsheet IPL JSON zip at ./ipl_dataset/ipl_json.zip
"""

import os
import json
import zipfile
import requests
from typing import Any, Dict, List
from tqdm import tqdm
import pandas as pd

# ---------------- CONFIG ----------------
OUT_DIR = "./ipl_dataset"
ZIP_LOCAL = os.path.join(OUT_DIR, "ipl_json.zip")
CRICSHEET_ZIP_URL = "https://cricsheet.org/downloads/ipl_json.zip"  # try auto-download
OUT_CSV = os.path.join(OUT_DIR, "ipl_live_states_2008_2025.csv")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------- Helpers: safe coercion ----------------
def as_dict(x: Any) -> Dict:
    """Ensure x is a dict. If x is JSON string, try parse; otherwise return {}."""
    if isinstance(x, dict):
        return x
    if x is None:
        return {}
    if isinstance(x, str):
        try:
            parsed = json.loads(x)
            if isinstance(parsed, dict):
                return parsed
            else:
                return {}
        except Exception:
            return {}
    return {}

def as_list(x: Any) -> List:
    """Ensure x is a list. If x is JSON string, try parse; otherwise return []."""
    if isinstance(x, list):
        return x
    if x is None:
        return []
    if isinstance(x, str):
        try:
            parsed = json.loads(x)
            if isinstance(parsed, list):
                return parsed
            else:
                return []
        except Exception:
            return []
    return []

# ---------------- Parse match info safely ----------------
def parse_match_info_safe(d: Any) -> Dict:
    d = as_dict(d)
    info = as_dict(d.get("info"))
    match = {}
    # season
    season = info.get("season")
    if season:
        try:
            match['season'] = int(season)
        except:
            match['season'] = season
    else:
        dates = as_list(info.get("dates"))
        if dates:
            try:
                match['season'] = int(str(dates[0]).split("-")[0])
            except:
                match['season'] = ""
        else:
            match['season'] = ""
    # basic fields
    dates = as_list(info.get("dates"))
    match['date'] = str(dates[0]) if dates else ""
    match['city'] = info.get("city") or ""
    match['venue'] = info.get("venue") or ""
    teams = as_list(info.get("teams"))
    match['team1'] = teams[0] if len(teams) > 0 else ""
    match['team2'] = teams[1] if len(teams) > 1 else ""
    toss = as_dict(info.get("toss"))
    match['toss_winner'] = toss.get("winner") or ""
    match['toss_decision'] = toss.get("decision") or ""
    outcome = as_dict(info.get("outcome"))
    match['winner'] = outcome.get("winner") or ""
    match['result'] = outcome.get("result") or ("normal" if match['winner'] else "")
    match['won_by_runs'] = outcome.get("runs") or 0
    match['won_by_wickets'] = outcome.get("wickets") or 0
    pom = info.get("player_of_match")
    match['player_of_match'] = ", ".join(pom) if isinstance(pom, list) else (pom or "")
    umps = as_list(info.get("umpires"))
    match['umpire1'] = umps[0] if len(umps) > 0 else ""
    match['umpire2'] = umps[1] if len(umps) > 1 else ""
    mid = info.get("match_id") or info.get("id") or info.get("match_number")
    if not mid:
        mid = f"{match.get('season','')}_{match.get('team1','')}_vs_{match.get('team2','')}_{match.get('date','')}"
    match['match_id'] = str(mid)
    return match

# ---------------- Parse deliveries & compute live-state ----------------
def parse_deliveries_and_states_safe(d: Any, match_meta: Dict) -> List[Dict]:
    d = as_dict(d)
    innings = as_list(d.get("innings"))
    rows = []

    for inning_entry in innings:
        if not isinstance(inning_entry, dict):
            continue
        for inning_name, inning_data_raw in inning_entry.items():
            inning_name = str(inning_name)
            inning_data = as_dict(inning_data_raw)
            inning_no = 1 if "1st" in inning_name.lower() else (2 if "2nd" in inning_name.lower() else 0)
            team_batting = inning_data.get("team") or ""
            deliveries = as_list(inning_data.get("deliveries"))
            cumulative_runs = 0
            wicket_count = 0
            legal_balls_bowled = 0

            # detect declared innings length if present (rain-reduced). Default T20=120 legal balls.
            innings_length_balls = 6 * 20
            if inning_data.get("overs") is not None:
                # some JSONs may store overs (float/str). If present, convert to balls: overs * 6
                try:
                    overs_val = float(inning_data.get("overs"))
                    innings_length_balls = int(round(overs_val * 6))
                except:
                    pass

            for ball in deliveries:
                if not isinstance(ball, dict):
                    continue
                for k, v_raw in ball.items():
                    k = str(k)
                    v = as_dict(v_raw)
                    # parse over.ball
                    over, ball_in_over = 0, 0
                    try:
                        a, b = k.split(".")
                        over = int(a)
                        ball_in_over = int(b)
                    except Exception:
                        try:
                            of = float(k)
                            over = int(of)
                            ball_in_over = int(round((of - over) * 10))
                        except:
                            over, ball_in_over = 0, 0

                    runs_info = as_dict(v.get("runs"))
                    # batsman runs fallback
                    batsman_runs = runs_info.get("batsman")
                    if batsman_runs is None:
                        batsman_runs = (runs_info.get("total") or 0) - (runs_info.get("extras") or 0)
                    try:
                        batsman_runs = int(batsman_runs or 0)
                    except:
                        batsman_runs = 0
                    extras = runs_info.get("extras") or 0
                    try:
                        extras = int(extras or 0)
                    except:
                        extras = 0
                    total_runs = runs_info.get("total")
                    if total_runs is None:
                        total_runs = batsman_runs + extras
                    try:
                        total_runs = int(total_runs or (batsman_runs + extras))
                    except:
                        total_runs = batsman_runs + extras

                    wicket_info = as_dict(v.get("wicket"))
                    dismissal_kind = wicket_info.get("kind") or ""
                    player_dismissed = wicket_info.get("player_out") or ""

                    # detect illegal delivery (wides/noballs) conservatively
                    illegal = False
                    extras_breakdown = v.get("extras")
                    if isinstance(extras_breakdown, dict):
                        if any(k2 in extras_breakdown for k2 in ("wides", "wide", "noballs", "noball", "nb")):
                            illegal = True
                    if extras > 0 and batsman_runs == 0 and not illegal:
                        # heuristic fallback
                        illegal = True

                    if not illegal:
                        legal_balls_bowled += 1
                    if dismissal_kind:
                        wicket_count += 1
                    cumulative_runs += total_runs

                    balls_bowled_in_innings = legal_balls_bowled
                    balls_remaining = max(innings_length_balls - balls_bowled_in_innings, 0)
                    overs_completed = (balls_bowled_in_innings // 6) + ((balls_bowled_in_innings % 6) / 6.0)
                    crr = (cumulative_runs / overs_completed) if overs_completed > 0 else 0.0

                    row = {
                        "match_id": match_meta.get("match_id",""),
                        "season": match_meta.get("season",""),
                        "date": match_meta.get("date",""),
                        "city": match_meta.get("city",""),
                        "venue": match_meta.get("venue",""),
                        "inning": inning_no,
                        "over": over,
                        "ball_in_over": ball_in_over,
                        "batting_team": team_batting,
                        "bowling_team": "",  # infer later
                        "striker": v.get("batsman") or "",
                        "non_striker": v.get("non_striker") or "",
                        "bowler": v.get("bowling") or "",
                        "runs_off_bat": batsman_runs,
                        "extras": extras,
                        "total_runs": total_runs,
                        "dismissal_kind": dismissal_kind,
                        "player_dismissed": player_dismissed,
                        "inning_runs_to_date": cumulative_runs,
                        "inning_wickets_down": wicket_count,
                        "balls_bowled_in_innings": balls_bowled_in_innings,
                        "balls_remaining": balls_remaining,
                        "runs_required": None,
                        "runs_left": None,
                        "wickets_left": max(10 - wicket_count, 0),
                        "crr": round(crr, 3),
                        "rrr": None,
                        "target": None,
                        "is_super_over": ("super over" in inning_name.lower()),
                        "winner": match_meta.get("winner",""),
                        "result": match_meta.get("result","")
                    }
                    rows.append(row)

    # Post-process: infer bowling_team, fill target/runs_left/rrr for inning 2 rows
    if not rows:
        return rows

    inning1_final = None
    for r in reversed(rows):
        if r.get("inning") == 1:
            inning1_final = r.get("inning_runs_to_date")
            break

    teams = [match_meta.get("team1") or "", match_meta.get("team2") or ""]

    for r in rows:
        batting = r.get("batting_team") or ""
        if batting and any(t for t in teams if t == batting):
            other = [t for t in teams if t and t != batting]
            r["bowling_team"] = other[0] if other else ""
        else:
            r["bowling_team"] = ""

        if r.get("inning") == 2 and inning1_final is not None:
            target = inning1_final + 1
            r["target"] = target
            runs_required = target - r.get("inning_runs_to_date", 0)
            runs_required = runs_required if runs_required > 0 else 0
            r["runs_required"] = runs_required
            r["runs_left"] = runs_required
            overs_remaining = (r["balls_remaining"] / 6.0) if r["balls_remaining"] > 0 else 0
            r["rrr"] = round((r["runs_left"] / overs_remaining), 3) if overs_remaining > 0 and r["runs_left"] is not None else None
        else:
            r["target"] = None
            r["runs_required"] = None
            r["runs_left"] = None
            r["rrr"] = None

    return rows

# ---------------- Download helper ----------------
def download_cricsheet_zip(url: str, outpath: str) -> str:
    if os.path.exists(outpath):
        print("Found existing zip:", outpath)
        return outpath
    print("Downloading Cricsheet IPL JSON zip (may be large)...")
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    with open(outpath, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024*1024):
            if chunk:
                f.write(chunk)
    print("Saved to", outpath)
    return outpath

# ---------------- Build CSV ----------------
def build_from_cricsheet(zip_path: str, out_csv: str):
    all_rows = []
    with zipfile.ZipFile(zip_path, "r") as z:
        json_files = [name for name in z.namelist() if name.endswith(".json")]
        print("Matches to parse:", len(json_files))
        for jf in tqdm(json_files):
            try:
                with z.open(jf) as f:
                    d = json.load(f)
            except Exception as e:
                print("Error reading", jf, e)
                continue
            match_meta = parse_match_info_safe(d)
            rows = parse_deliveries_and_states_safe(d, match_meta)
            # ensure some match-level fields populated
            for rr in rows:
                rr['season'] = match_meta.get('season')
                rr['date'] = match_meta.get('date')
                rr['city'] = match_meta.get('city')
                rr['venue'] = match_meta.get('venue')
            all_rows.extend(rows)

    if not all_rows:
        print("No rows parsed. Exiting.")
        return

    df = pd.DataFrame(all_rows)

    # Choose column order (alias some names for the user's requested names)
    # Provide requested names: batting team, bowling team, city, balls_left, run_left, wicketsleft, total_runs, crr, rrr, result
    df = df.rename(columns={
        "balls_remaining": "balls_left",
        "runs_left": "run_left",
        "wickets_left": "wicketsleft",
        "total_runs": "total_runs_off_delivery"  # keep original delivered runs as total_runs_off_delivery
    })

    # But keep also inning_runs_to_date and other columns; compute cumulative total_runs column for the innings if user wants 'total_runs' as inning total:
    # We'll add 'inning_total_runs' representing inning_runs_to_date for clarity.
    if "inning_runs_to_date" in df.columns:
        df["inning_total_runs"] = df["inning_runs_to_date"]

    # Reorder to a helpful order
    preferred_cols = [
        "match_id", "season", "date", "city", "venue",
        "inning", "over", "ball_in_over",
        "batting_team", "bowling_team",
        "striker", "non_striker", "bowler",
        "runs_off_bat", "total_runs_off_delivery", "extras",
        "inning_total_runs",
        "balls_bowled_in_innings", "balls_left",
        "target", "run_left", "runs_required",
        "wicketsleft",
        "crr", "rrr",
        "dismissal_kind", "player_dismissed",
        "is_super_over", "winner", "result"
    ]
    # keep only existing in df
    cols = [c for c in preferred_cols if c in df.columns] + [c for c in df.columns if c not in preferred_cols]
    df = df[cols]

    df.to_csv(out_csv, index=False)
    print("Wrote", out_csv, "rows:", len(df))

# ---------------- Main ----------------
def main():
    try:
        zip_local = download_cricsheet_zip(CRICSHEET_ZIP_URL, ZIP_LOCAL)
    except Exception as e:
        print("Auto-download failed:", e)
        print("If you have the Cricsheet IPL JSON zip, place it at:", ZIP_LOCAL, "and re-run.")
        return
    build_from_cricsheet(zip_local, OUT_CSV)

if __name__ == "__main__":
    main()
