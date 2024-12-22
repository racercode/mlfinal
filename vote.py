import pandas as pd

file1path = "submission/BAGResult.csv"
file2path = "submission/predictions_with_ids.csv"
file3path = "submission/submission (1).csv"

df1 = pd.read_csv(file1path)
df2 = pd.read_csv(file2path)
df3 = pd.read_csv(file3path)

weight = [0.57682, 0.58618, 0.58263]

df1["home_team_win"] = ((weight[0] * df1["home_team_win"] + weight[1] * df2["home_team_win"] + weight[2] * df3["home_team_win"]) / sum(weight))>0.5
df1.to_csv("final.csv", index=False)