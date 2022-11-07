# Data_Science_AirBnB

The aim of this project is to build a framework for the evaluation of various machine learning models which can then be applied to different datasets. It involves an initial step of exploratory data analysis on the first dataset - AirBnB data, which is cleaned and seperated into text, tabular and image data. Regression and classification models are then trained, tuned and evaluated to discover the best performing model which are saved as a .joblib file. Finally a configurable neural network was created.

## Data Preparation
- The tabular dataset was read in from a csv file and functions were defined to remove/edit the dataframe based on the column in its parameters.
    - The tabular data had rows removed if certain columns were NaN or some NaN values were set to a specific int.
    - The text data within the dataset had its several string descriptions combined into 1 string in its specific column.
    - These functions accumulated in a load_airbnb() function which outputs the features and labels based on the input parameters.
    ```python
    def load_airbnb(label: str = "Price_Night", str_cols: list = ["ID", "Category", "Title", "Description", "Amenities", "Location", "url"]):
    ```
    - Here the label parameter isolates the specified column, the str_cols parameter columns are removed from the dataframe which leaves the rest as features.
- The image folder and files were downloaded and resized to the same height and width based on the smallest images height, whilst maintaining the aspect ratio. 
- Once the data is cleaned it is split into 3 subsets: train, validation and test.
```python
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3)
xvalidation, xtest, yvalidation, ytest = train_test_split(xtest, ytest, test_size=0.5)
```
-   The train set(70%) is used for model optimisation
-   the validation set(15%) is used to make decisions about the model(e.g. which is the best model)
-   The test set(15%) is used to show how the model will perform on unseen data

## Regression 
- Regression models were chosen, trained on the same dataset and evaluated with gridsearchcv with custom hyperparameters to determine the optimal model. 
### Evaluation
- Loss functions provide a measurement on the models performance.
- The root mean squared error was the loss function chosen to determine the best model. More can be found about it here: https://www.brainstobytes.com/mean-absolute-error-vs-root-mean-square-error/
- Several other metrics such as Rsquared and mean absolute error can also be tracked. (https://www.mygreatlearning.com/blog/r-square/#:~:text=R-square%20is%20a%20goodness-of-fit%20measure%20for%20linear%20regression,variable%20on%20a%20convenient%200%20%E2%80%93%20100%25%20scale.)

## Models
### Stochastic Gradient Descent
- Usually a Gradient Descent optimisation technique (seen in image below) would be very computationally expensive with a large dataset, but Stochastic Gradient Descent selects a random data sample to calculate the derivitives instead of the whole dataset.
- ![image](https://user-images.githubusercontent.com/108297203/200419171-2dd31e1a-1b87-44fa-a1e2-b716df9cff64.png)
- o
### Regularization
- This parameter is present in SGDRegressor and Logstic Regression.
- It reduces the overfitting/generalsation error (difference between training and validation sets) by discouraging a learning a more complex/flexible model.
- More can be found at: 
    - https://www.pluralsight.com/guides/linear-lasso-ridge-regression-scikit-learn
    - https://towardsdatascience.com/regularization-in-machine-learning-76441ddcf99a


## Neural Networks
- These are computing systems which are comprised of many layers of interconnected units. 
![image](https://user-images.githubusercontent.com/108297203/200390046-30515704-46c2-41e2-a751-d84341f99ae1.png)
- Neural networks can have many layers, the increase in layers allows the network to perfrom more complex calculations.
- Each node in the network 
