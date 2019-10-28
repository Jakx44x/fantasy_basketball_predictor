from bs4 import BeautifulSoup
from requests import get, Session
import numpy as np 
import pandas as pd
from json import loads
from pprint import pprint

def main():
	all_players, season = getPlayers()
	playerDF = parsePlayers(all_players)
	print(playerDF)
	playerDF.to_csv(f'{season}_Player_stats.csv', index=False)

def getPlayers():
	#gives your the JSON FILE with all Statistcs for all NBA Players of the particular season
	season = '2014-15'
	url = f'https://stats.nba.com/stats/leagueLeaders?LeagueID=00&PerMode=PerGame&Scope=S&Season={season}&SeasonType=Regular+Season&StatCategory=PTS'
	session = Session()
	page = session.get(url, timeout=(2,5))



	JSON = loads(page.text)

	all_players = JSON['resultSet']['rowSet']

	return all_players, season


def parsePlayers(all_players):
	'''all_players is json format for everysingle player with there stats in the draft. This function will get each player arranged by ranking
	into a df
	'''

	cols = ["PLAYER","TEAM","GP","MIN","FGM","FGA","FG_PCT","FG3M","FG3A","FG3_PCT","FTM","FTA","FT_PCT","OREB","DREB","REB","AST","STL","BLK","TOV","PTS","EFF"]
	#indexed all_players
	all_player_stats = []	
	col_num = 0
	for player in all_players:
		player_stats = []
		#grabs all the statistics from its position `
		player_stats = player[2:]
		#places all the statistics in the dictionary
		all_player_stats.append(player_stats)
	
	playerDF = pd.DataFrame(all_player_stats, columns=cols)

	return playerDF

if __name__ == '__main__':
	main()