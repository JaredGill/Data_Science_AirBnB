from pyexpat import model
from pandasgui import show
from sklearn.linear_model import SGDRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import scale
from tabular_data import clean_tabular_data
from tabular_data import load_airbnb
import inspect
import matplotlib.pyplot as plt
import numpy as np
import os

def data_sets():
    #df = ()
    airbnb = load_airbnb()
    #feature
    x = airbnb[0]
    #labels
    y = airbnb[1]
    # show(x)
    # show(y)

    #standardise a dataset across any axis
    #scale features not labels
    x = scale(x)
    #y = scale(y)

    #create validation and test samples made of 15% of the data each, and 70% for training
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3)
    xvalidation, xtest, yvalidation, ytest = train_test_split(xtest, ytest, test_size=0.5)
    return xtrain, xtest, xvalidation, ytrain, ytest, yvalidation

#dont use test set until optimised model
#score on validation set
#tune with validation for hyperparameters
#test set is at end of training data

def simple_model(xtrain, xvalidation, ytrain, yvalidation):
    #instantiate linear model
    #sgdr = SGDRegressor(alpha=0.0001, epsilon=0.01, eta0=0.1,penalty='elasticnet')
    sgdr = SGDRegressor(alpha=0.0001, epsilon=0.01, eta0=0.1,penalty='elasticnet')
    #sgdr.score()
    #train data and check the model accuracy score.
    sgdr.fit(xtrain, ytrain)
    score = sgdr.score(xtrain, ytrain)
    print("Train R-squared:", score)
    test_score = sgdr.score(xvalidation, yvalidation)
    print("Test R-squared:", test_score)

    #get predictions with feature test data(ypred)
    ypred = sgdr.predict(xvalidation)
    #print("Predictions:\n", ypred[:10])

    #train_mse = mean_squared_error()
    mse = mean_squared_error(yvalidation, ypred)
    rmse = mean_squared_error(yvalidation, ypred,squared=False)
    mae = mean_absolute_error(yvalidation, ypred)
    print("MSE: ", mse)
    #root mean squared error
    #print("RMSE: ", mse**(1/2.0))
    print("RMSE:", rmse)
    print("MAE:", mae)

    #notebook ver
    samples = len(ypred)
    plt.figure()
    plt.scatter(np.arange(samples), ypred, c='r', label='predictions')
    plt.scatter(np.arange(samples), yvalidation, c='b', label='true labels', marker='x')
    plt.legend()
    plt.title("AirBnB test and predicted data")
    plt.xlabel('X-axis')
    plt.ylabel('Price per night')
    plt.show()

    # visualize the original and predicted data in a plot.
    x_ax = range(len(yvalidation))
    plt.plot(x_ax, yvalidation, label="original")
    plt.plot(x_ax, ypred, label="predicted")
    plt.title("AirBnB test and predicted data")
    plt.xlabel('X-axis')
    plt.ylabel('Price per night')
    plt.legend(loc='best',fancybox=True, shadow=True)
    plt.grid(True)
    plt.show() 
    return sgdr
    
def custom_tune_regression_model_hyperparameters(model, train_features, train_labels, 
                                                        val_features, val_labels,
                                                        test_features, test_labels, 
                                                        hyperparameters_dict: dict):
    # iterate through every possible variation and get rmse from all, the best model & hyperparameters will be from rmse 
    sgdr = model()
    perf_metrics = {"validation_RMSE": [], "r2_score": []}
    grid = list(ParameterGrid(hyperparameters_dict))
    models = []
    for i in grid:
        mydict = i
        models.append(i)

        #check each key is present in parameters for method SGDRegressor
        filtered_mydict = {k: v for k, v in mydict.items() if k in [p.name for p in inspect.signature(SGDRegressor).parameters.values()]}
        
        #pass in dict as parameters
        sgdr = SGDRegressor(**filtered_mydict)
        sgdr.fit(train_features, train_labels)
        score = sgdr.score(train_features, train_labels)
        ypred = sgdr.predict(val_features)
        rmse = mean_squared_error(val_labels, ypred, squared=False)
        perf_metrics['validation_RMSE'].append(rmse)
        perf_metrics['r2_score'].append(score)

    #print(perf_metrics)
    best_rmse = min(perf_metrics["validation_RMSE"])
    print("best rmse", best_rmse)
    for item, value in perf_metrics.items():
        if best_rmse in value:
            best_rmse_pos = value.index(best_rmse)
            print(value.index(best_rmse))
    #print(models)
    print(models[best_rmse_pos])
    best_hyperparameters = models[best_rmse_pos]

    best_model = SGDRegressor(**filtered_mydict)
    return best_model, best_hyperparameters, perf_metrics, best_rmse

def tune_regression_model_hyperparameters(model, xtrain, ytrain, hyperparameters_dict: dict):
    #
    sgdr = model()

    search = GridSearchCV(sgdr, param_grid=hyperparameters_dict, cv=5, scoring='neg_root_mean_squared_error')
    result = search.fit(xtrain, ytrain)
    best_hyperparameters = result.best_params_
    best_score = result.best_score_
    print('Best Score: %s' % result.best_score_)
    print('Best Hyperparameters: %s' % result.best_params_)
    return best_hyperparameters, best_score

def save_model(foldername: str = "//models//regression//linear_regression"):
    curr_dir = os.getcwd()
    target_path = os.path.join(curr_dir, foldername)
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    


def model_data():
    #more descriptive names 
    data = data_sets()
    #ensure it only works when data has the same length as parameters
    xtrain, xtest, xvalidation, ytrain, ytest, yvalidation = data
    # xtrain = data[0]
    # xtest = data[1]
    # xvalidation = data[2]
    # ytrain = data[3]
    # ytest = [4]
    # yvalidation = data[5]
    simple_model(xtrain, xvalidation, ytrain, yvalidation)
    #model = (xtrain, xvalidation, ytrain, yvalidation)
    param_grid = {"learning_rate": ["constant", "adaptive", "optimal"], 
                "eta0": [0.001, 0.005, 0.01, 0.05, 0.1, 0.5], 
                "alpha": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1],
                "penalty": ["l2", "l1", "elasticnet"],
                #"loss": ["squared_error"]
                }
    custom = custom_tune_regression_model_hyperparameters(SGDRegressor, xtrain, ytrain, xvalidation, yvalidation, xtest, ytest, param_grid)
    prebuilt = tune_regression_model_hyperparameters(SGDRegressor, xtrain, ytrain, param_grid)
    #print(custom[0])
    # 
    print("best custom hyperparameters:", custom[1])
    print("best custom rmse:", custom[3])
    #print(prebuilt)
    save_model()
if __name__ == "__main__":
    model_data()
