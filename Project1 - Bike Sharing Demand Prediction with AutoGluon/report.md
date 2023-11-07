# Report: Predict Bike Sharing Demand with AutoGluon Solution
#### Lucindah Fry-Nartey

## Initial Training
### What did you realize when you tried to submit your predictions? What changes were needed to the output of the predictor to submit your results?

I needed to remove the negative values from the predictions, setting them to zero. 

### What was the top ranked model that performed?

The top-ranked model was WeightedEnsemble_L2 with a score of -84.125.

## Exploratory data analysis and feature creation
### What did the exploratory analysis find and how did you add additional features?

The exploratory data analysis showed the distributions of features such as temperature, wind speed, humidity and the count of bikes rented. It also showed the categories present in features such as the season and weather. Generally, the exploratory data analysis gave a broad overview of what the data looks like. The data was fairly clean.

I targeted the 'datetime' column in the data to provide additional features. I first converted the 'datetime' column to a datetime object and then extracted the year, month, day and hour as additional features using the pandas datetime functionality. I also converted the season and weather columns into categorical data types. 

### How much better did your model preform after adding additional features and why do you think that is?

The model performed much better with the new features, with the error reducing by about half of the initial training error. I think this was so because the new features gave the model much more useful information to be able to make more correct predictions. 

## Hyper parameter tuning
### How much better did your model perform after trying different hyper parameters?

The error increased by about 0.5, but the kaggle score went from 0.617 to 0.491. 

### If you were given more time with this dataset, where do you think you would spend more time?

I would spend more time with the hyperparameter optimization, specifically for the top models on the Autogluon leaderboard. 

### Create a table with the models you ran, the hyperparameters modified, and the kaggle score.
|model|hpo1|hpo2|hpo3|score|
|--|--|--|--|--|
|initial|time_limit|presets|N/A|1.847|
|add_features|time_limit|presets|N/A|0.617|
|hpo|num_trials|searcher|scheduler|0.491|

### Create a line plot showing the top model score for the three (or more) training runs during the project.


![model_train_score.png](img/model_train_score.png)

### Create a line plot showing the top kaggle score for the three (or more) prediction submissions during the project.


![model_test_score.png](img/model_test_score.png)

## Summary
This project trained models to predict bike sharing demand using Autogluon. The models were trained first with the original data, with additional features and with hyperparameter optimization before training. The performance of the models in each instance were compared. The models trained with addtiional features performed better (having a smaller loss and kaggle score) than that trained on the original data. The models trained after hyperparameter optimization in addition to the new features generated had the best kaggle score. 
