# Data_Science_AirBnB

The aim of this project is to build a framework for the evaluation of various machine learning models which can then be applied to different datasets. It involves an initial step of exploratory data analysis on the first dataset - AirBnB data, which is cleaned and seperated into text, tabular and image data. Regression and classification models are then trained, tuned and evaluated to discover the best performing model which are saved as a .joblib file. Finally a configurable neural network was created.

## Data Preparation
- The tabular dataset was read in from a csv file and functions were defined to remove/edit the dataframe based on the column in its parameters.
    - The tabular data had rows removed if certain columns were NaN or some NaN values were set to a specific int.
    - The text data within the dataset had its several string descriptions combined into 1 string in its specific column.
    - These functions accumulated in a load_airbnb() function which outputs the features and labels based on the input parameters.
- The image folder and files were downloaded and resized to the same height and width based on the smallest images height, whilst maintaining the aspect ratio. 

## Regression 
- Regression models were chosen, trained on the same dataset and evaluated with gridsearchcv with custom hyperparameters to determine the optimal model. 
