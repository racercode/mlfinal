import pandas as pd
import random

def clean_data(path, std_out_threshold, std_dev_threshold, delete_rows = True):

    data = pd.read_csv(path)
    std_out_threshold = (int)(std_out_threshold * len(data.columns))

    if 'date' in data.columns:
        data.drop(columns=['date'], inplace=True)

    data.drop(columns=['id', 'home_team_abbr', 'away_team_abbr', 'home_pitcher', 'away_pitcher', 'season', 'home_team_season', 'away_team_season'], inplace=True)

    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    means = data[numeric_columns].mean()
    stds = data[numeric_columns].std()

    def within_std_range(row):
        cnt = 0
        if 'home_team_win' in data.columns and pd.isna(row['home_team_win']):
            return False
        for col in numeric_columns:
            if pd.isna(row[col]):
                cnt += 1
                if cnt > std_out_threshold:
                    return False
        else:
            return True

    if delete_rows:
        data = data[data.apply(within_std_range, axis=1)]
    for row in data.columns:
        if row == 'is_night_game':
            c = random.choices([0, 1], weights=(1, 2), k=1)[0]
            data[row].fillna(c, inplace=True)
        elif row != 'home_team_win':
            if 'rest' in row:
                data[row].fillna(means[row], inplace=True)
            else:
                data[row].fillna(0, inplace=True)

    data['is_night_game'] = data['is_night_game'].astype(int)
    if 'home_team_win' in data.columns:
        data['home_team_win'] = data['home_team_win'].astype(int)

    return data
