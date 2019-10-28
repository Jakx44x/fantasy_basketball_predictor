# fantasy_basketball_predictor
This Repository will scrape statistics from a given NBA season, and then train a model to predict the next years fantasy basketball season

This is specifically to be used for standard week-by-week leagues which use a scoring system based on the stats. 

The order in which to use these files is as follows:
1. statScraper.py
2. fantasyTrainer.py
3. fantasyPredictor.py

1.statScraper.py 
This file will be used to scrape statistics off of NBA.com for a particular NBA season which you will specify as a variable.
Once the data is in the program, it will parse it into a Pandas DataFrame and export it to csv file.
-I gave 2 example files that were scraped using this program, one for the 2017-18 and the 2019-19 season

2. fantasyTrainer.py
Once you have the csv Files you will use the data to train a model using TensorFlows' Keras library. 
You will use one season, say 2016-17, to try and predict the next years season, 2017-18. 
The training uses Mean Squared Error as its loss function. 
Once trained, the program will save the model to a .h5 file

Few Notes:
One will need to optimize how scoring is calculated for your individual league. These changes can be made in the get_fantasy() function
You will also need to specify whether you want this model try and predict a players avg pts-per-game, or his total pts-per-season. This
can be changed from a varaible in main() called prediction_type

3.fantasyPredictor.py
Finally, you will use this file to take the .h5 model and predict the fantasy scores. I use this to predict 

Final Notes:
I used this to predict total fantasy scores for players this current season (2019-20). I personally believe my model needs a lot of work
I think one improvement I should implement is to change the inputs to be 3 years worth of statistics instead of just one. I think this
could help account for potential growth in players to find some players who are about to have a breakout year. 
