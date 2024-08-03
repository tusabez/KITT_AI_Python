import pandas as pd
from nba_api.stats.static import teams
from nba_api.stats.endpoints import leaguegamefinder, boxscoretraditionalv2
from datetime import datetime, timedelta
import random

# Fetch all NBA teams
nba_teams = teams.get_teams()

# Select the dictionary for the Lakers, which contains their team ID
lakers = [team for team in nba_teams if team['abbreviation'] == 'LAL'][0]
lakers_id = lakers['id']

# Query for games where the Lakers were playing
gamefinder = leaguegamefinder.LeagueGameFinder(team_id_nullable=lakers_id)

# The first DataFrame of those returned is what we want
games = gamefinder.get_data_frames()[0]

# Set the date range for the last 5 days
start_date = (datetime.today() - timedelta(5)).strftime('%Y-%m-%d')
end_date = datetime.today().strftime('%Y-%m-%d')

# Filter for games in the last 5 days and sort by date
games_last_5_days = games[(games['GAME_DATE'] >= start_date) & (games['GAME_DATE'] <= end_date)].sort_values(by='GAME_DATE', ascending=False)

# Ensure we only process the most recent game
if not games_last_5_days.empty:
    most_recent_game = games_last_5_days.iloc[0]
    game_id = most_recent_game['GAME_ID']
    game_date = most_recent_game['GAME_DATE']
    matchup = most_recent_game['MATCHUP']
    home_team_name = "Los Angeles Lakers" if "vs." in matchup else next(team['full_name'] for team in nba_teams if team['abbreviation'] == matchup.split(' ')[2])
    away_team_name = next(team['full_name'] for team in nba_teams if team['abbreviation'] == matchup.split(' ')[2]) if "vs." in matchup else "Los Angeles Lakers"
    home_team_id = most_recent_game['TEAM_ID'] if home_team_name == "Los Angeles Lakers" else next(team['id'] for team in nba_teams if team['full_name'] == home_team_name)
    
    # Determine if the Lakers were playing at home or on the road
    location = "at home" if "vs." in matchup else "on the road"
    
    # Fetch the box score for the game
    box_score = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id).get_data_frames()[0]
    team_scores = box_score.groupby('TEAM_ID')['PTS'].sum().to_dict()
    
    # Identify the away team ID
    away_team_id = next(tid for tid in team_scores.keys() if tid != home_team_id)
    
    home_score = round(team_scores[home_team_id])
    away_score = round(team_scores[away_team_id])
    
    # Determine if the Lakers won or lost
    win_phrases = ["won against", "beat"]
    lose_phrases = ["lost to", "were defeated by"]
    blowout_phrases = ["blew out", "were blown out by"]
    
    score_diff = abs(home_score - away_score)
    if home_team_name == "Los Angeles Lakers":
        if home_score > away_score:
            outcome = blowout_phrases[0] if score_diff >= 20 else random.choice(win_phrases)
            mp3_files = ['lakers1.mp3', 'lakers2.mp3', 'lakers3.mp3', 'lakers4.mp3', 'lakers5.mp3']
        else:
            outcome = blowout_phrases[1] if score_diff >= 20 else random.choice(lose_phrases)
            mp3_files = ['lakers6.mp3', 'lakers7.mp3', 'lakers8.mp3', 'lakers9.mp3', 'lakers10.mp3']
    else:
        if away_score > home_score:
            outcome = blowout_phrases[0] if score_diff >= 20 else random.choice(win_phrases)
            mp3_files = ['lakers1.mp3', 'lakers2.mp3', 'lakers3.mp3', 'lakers4.mp3', 'lakers5.mp3']
        else:
            outcome = blowout_phrases[1] if score_diff >= 20 else random.choice(lose_phrases)
            mp3_files = ['lakers6.mp3', 'lakers7.mp3', 'lakers8.mp3', 'lakers9.mp3', 'lakers10.mp3']
    
    # Convert date to natural English format
    game_date_obj = datetime.strptime(game_date, '%Y-%m-%d')
    game_date_natural = game_date_obj.strftime('%A')
    
    # Find the Lakers' player with the most points
    lakers_box_score = box_score[box_score['TEAM_ID'] == lakers_id]
    top_lakers_scorer = lakers_box_score.loc[lakers_box_score['PTS'].idxmax()]
    
    player_name = top_lakers_scorer['PLAYER_NAME']
    points = round(top_lakers_scorer['PTS'])
    rebounds = round(top_lakers_scorer['REB'])
    assists = round(top_lakers_scorer['AST'])
    blocks = round(top_lakers_scorer['BLK'])
    
    # Find the top scorer of the game
    top_game_scorer = box_score.loc[box_score['PTS'].idxmax()]
    if top_game_scorer['TEAM_ID'] == lakers_id:
        scorer_type = "game-high"
    else:
        scorer_type = "team-high"
    
    # Format player stats
    player_stats = [f"{player_name} scored a {scorer_type} {points} point{'s' if points != 1 else ''}"]
    if rebounds > 0:
        player_stats.append(f"grabbed {rebounds} rebound{'s' if rebounds != 1 else ''}")
    if assists > 0:
        player_stats.append(f"dished out {assists} assist{'s' if assists != 1 else ''}")
    if blocks > 0:
        player_stats.append(f"blocked {blocks} shot{'s' if blocks != 1 else ''}")
    
    player_stats_str = ", ".join(player_stats[:-1])
    if len(player_stats) > 1:
        player_stats_str += f", and {player_stats[-1]}"
    else:
        player_stats_str = player_stats[0]
    
    line = f"On {game_date_natural}, the Los Angeles Lakers {outcome} the {away_team_name if home_team_name == 'Los Angeles Lakers' else home_team_name} {location}. The final score was {home_team_name} {home_score} - {away_team_name} {away_score}. {player_stats_str}."
    
    # Print the output to the console
    print(line)
    # Print the selected MP3 files for the outcome
    print("Selected MP3 files:", mp3_files)
else:
    print("No games found in the last 5 days.")
