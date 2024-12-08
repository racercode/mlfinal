from xgboost import XGBClassifier
import pandas as pd

test_data_path = 'test_data_ver1.csv'
train_and_validation_data_path = 'train_data_ver3.csv'

dataset = pd.read_csv(train_and_validation_data_path)
chosed_features = [
    'home_batting_onbase_plus_slugging_10RA', 'away_batting_onbase_plus_slugging_10RA', 
    'home_team_wins_mean', 'away_team_wins_mean', 
    'home_team_wins_skew', 'away_team_wins_skew',
    'home_batting_onbase_plus_slugging_mean', 'away_batting_onbase_plus_slugging_mean',
    'home_batting_onbase_plus_slugging_skew', 'away_batting_onbase_plus_slugging_skew', 
    'home_pitching_earned_run_avg_mean', 'away_pitching_earned_run_avg_mean', 
    'home_pitching_earned_run_avg_skew', 'away_pitching_earned_run_avg_skew', 
    'home_pitcher_earned_run_avg_10RA', 'away_pitcher_earned_run_avg_10RA',
    'home_batting_wpa_bat_mean', 'away_batting_wpa_bat_mean',
    'home_batting_wpa_bat_skew', 'away_batting_wpa_bat_skew',
    'home_batting_onbase_perc_mean', 'away_batting_onbase_perc_mean',
    'home_batting_onbase_perc_skew', 'away_batting_onbase_perc_skew',
    'home_pitching_H_batters_faced_10RA', 'away_pitching_H_batters_faced_10RA',
]

drop_features = [
    'id', 'home_team_abbr',
    'away_team_abbr', 'is_night_game',
    'home_pitcher', 'away_pitcher',
    'home_team_rest', 'away_team_rest',
    'home_pitcher_rest','away_pitcher_rest', 
    'season'
]
dataset = dataset[chosed_features + ['home_team_win']]
#dataset.drop(drop_features, axis=1, inplace=True)
num_chunks = 10
chunk_size = len(dataset) // num_chunks
chunks = [dataset.iloc[i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks)]
validation_dataset = chunks[0]
train_dataset = dataset.drop(chunks[0].index)
X_train = train_dataset.drop('home_team_win', axis=1)
y_train = train_dataset['home_team_win']
X_validation = validation_dataset.drop('home_team_win', axis=1)
y_validation = validation_dataset['home_team_win']

from sklearn import ensemble, preprocessing, metrics

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

# Load or create your dataset
# Example: df = pd.read_csv('your_dataset.csv')
# Assume 'features' are the independent variables and 'target' is the binary dependent variable

# For demonstration, let's create a synthetic datase

# Initialize the XGBoost classifier
forest = ensemble.RandomForestClassifier(n_estimators = 100)
forest_fit = forest.fit(X_train, y_train)

# 預測
test_y_predicted = forest.predict(X_validation)

# 績效
accuracy = metrics.accuracy_score(y_validation, test_y_predicted)
print(accuracy)