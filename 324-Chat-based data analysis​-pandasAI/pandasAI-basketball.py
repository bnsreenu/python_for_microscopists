# https://youtu.be/9JWsiY_KdVY
"""
A very short introduction to pandasAI

Dataset from: 
https://www.kaggle.com/datasets/vivovinco/nba-player-stats


"""

from dotenv import load_dotenv

load_dotenv()

import pandas as pd
from pandasai import PandasAI


########################################################################

df = pd.read_csv("2021-2022 NBA Player Stats - Regular.csv", 
                 encoding="Windows-1252", delimiter=";")
print(df.columns)

df = df[df['Tm'] != 'TOT']

# Instantiate a LLM
from pandasai.llm.openai import OpenAI
llm = OpenAI()

pandas_ai = PandasAI(llm, conversational=True)

pandas_ai.run(df, prompt='Who was the oldest player in the league?')

pandas_ai.run(df, prompt='Who are the top 3 players that had the most \
              defensive rebounds? How many each?')

pandas_ai.run(df,"Plot the 3 point per game versus total score for each \
              player on a scatter plot with small dots. Do not label with \
                  player name, just the score on y-axis.")

pandas_ai.run(df,"Please group all players for each of these positions \
              SG, SF PG, PF, C. then plot 3P per game as a function of \
                  the position. I want to understand if certain position \
                      tend to score more 3-pointers. ")

pandas_ai.run(df,"Plot a bar plot of all teams showing the average age of \
              players. The plot shall show teams with average age in \
                  decreasing order from oldest to youngest. ")
