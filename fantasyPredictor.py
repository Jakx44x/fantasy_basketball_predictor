import numpy as np 
import tensorflow.keras as keras
import pandas as pd 
#avg to get avg predictions.
#totals to get total score predictions
prediction_type = 'avg'

prediction_data_year = '2018-19'

season_data = pd.read_csv(f'{prediction_data_year}_Player_stats.csv').set_index("PLAYER")

season_data_no_teams = season_data.drop(['TEAM'], axis=1)

print(season_data_no_teams)

prediction_arrays = np.array(season_data_no_teams)

model = keras.models.load_model(f'{prediction_type}_model.h5')

predictions = model.predict(prediction_arrays)

print(predictions)

season_data['fantasy_predictions'] = predictions

season_data.to_csv(f'{prediction_data_year}_{prediction_type}_predictions.csv')

print(season_data)

# predictions = model.predict()