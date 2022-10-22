from pyexpat import model
from pandasgui import show
from sklearn.linear_model import SGDRegressor
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import scale
from sklearn.tree import DecisionTreeRegressor
from tabular_data import clean_tabular_data
from tabular_data import load_airbnb
import inspect
import joblib
import json
import matplotlib.pyplot as plt
import numpy as np
import os

def load_and_split_data():
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

def simple_model(model, xtrain, xvalidation, ytrain, yvalidation):
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
    perf_metrics = {"hyperparameters": [], "validation_RMSE": [], "r2_score": []}
    grid = list(ParameterGrid(hyperparameters_dict))
    models = []
    for params in grid:
        mydict = params
        models.append(params)

        #check each key is present in parameters for method SGDRegressor
        filtered_mydict = {k: v for k, v in mydict.items() if k in [p.name for p in inspect.signature(SGDRegressor).parameters.values()]}
        
        #pass in dict as parameters
        sgdr = SGDRegressor(**filtered_mydict)
        sgdr.fit(train_features, train_labels)
        score = sgdr.score(train_features, train_labels)
        ypred = sgdr.predict(val_features)
        rmse = mean_squared_error(val_labels, ypred, squared=False)
        perf_metrics['hyperparameters'].append(params)
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
    print(best_hyperparameters)
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
    cv_results = search.cv_results_
    perf_metrics = {
        #'hyperparameters': [], 
        'neg_root_mean_squared_error_score': []}
    perf_metrics['neg_root_mean_squared_error_score'].append(result.best_score_)
    # perf_metrics['hyperparameters'].extend(cv_results.get('params'))
    # perf_metrics['neg_root_mean_squared_error_score'].extend(cv_results.get('mean_test_score'))

    best_model = model(**best_hyperparameters)
    return best_model, best_hyperparameters, perf_metrics, best_score


def save_model(filename: str, hyperparameters, model, metrics, foldername: str = "models/regression/linear_regression"):
    curr_dir = os.getcwd()
    target_path = os.path.join(curr_dir, foldername)
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    model_filename = os.path.join(target_path, f'{filename}_model.joblib')
    hyperparameters_filename = os.path.join(target_path, f'{filename}_hyperparameters.json')
    metrics_filename = os.path.join(target_path, f'{filename}_metrics.json')
    print(model_filename)
    joblib.dump(model, model_filename)
    with open(hyperparameters_filename, 'w') as f:
        json.dump(hyperparameters, f)
    with open(metrics_filename, 'w') as f:
        json.dump(metrics, f)
    


def model_data(model, param_grid, model_name, folder_name, xtrain, xtest, xvalidation, ytrain, ytest, yvalidation):
    #more descriptive names 
    # data = load_and_split_data()
    # #ensure it only works when data has the same length as parameters
    # xtrain, xtest, xvalidation, ytrain, ytest, yvalidation = data
    #model = (xtrain, xvalidation, ytrain, yvalidation)
    
    gridsearch = tune_regression_model_hyperparameters(model, xtrain, ytrain, param_grid)
    gridsearch_best_model, gridsearch_best_hyperparameters, gridsearch_perf_metrics, gridsearch_best_rmse = gridsearch
    
    #simple_model(gridsearch_best_model, xtest, xvalidation, ytest, yvalidation)

    save_model(f"{model_name}", gridsearch_best_hyperparameters, gridsearch_best_model, gridsearch_perf_metrics, folder_name)
    return gridsearch_best_rmse

def evaluate_all_models():
    data = load_and_split_data()
    #ensure it only works when data has the same length as parameters
    xtrain, xtest, xvalidation, ytrain, ytest, yvalidation = data
    hyperparameters_sgdr =  {"learning_rate": ["constant", "adaptive", "optimal"], 
                            "eta0": [0.001, 0.005, 0.01, 0.05, 0.1, 0.5], 
                            "alpha": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1],
                            "penalty": ["l2", "l1", "elasticnet"],
                            #"loss": ["squared_error"]
                            } 
                                    # GradientBoostingRegressor
    hyperparameters_gbr = {'learning_rate': [0.001,0.005,0.01,0.05, 0.1],
                            'subsample': [0.9, 0.5, 0.2, 0.1],
                            'n_estimators': [100, 250, 500],
                            'max_depth': [1,2,4,6]
                            }
                                    # # RandomForestRegressor
    hyperparameters_rfr = {'criterion': ['squared_error'],
                            'min_samples_split' : [2,4,8],
                            'n_estimators': [100,500,1000, 1500],
                            'max_depth': [4,6,8,10]
                            },
                                    # # DecisionTreeRegressor
    hyperparameters_dfr =  {'criterion': ['squared_error'],
                            'min_samples_split' : [2,4,8],
                            "max_features": ["log2","sqrt"],
                            'max_depth': [2,4,6,8,10], 
                            "min_weight_fraction_leaf":[0.1,0.2,0.3,0.4,0.5]
                            }
    
    model_data(SGDRegressor, hyperparameters_sgdr, 'SGDRegressor', "models/regression/linear_regression", xtrain, xtest, xvalidation, ytrain, ytest, yvalidation)
    model_data(GradientBoostingRegressor, hyperparameters_gbr, 'GradientBoostingRegressor', "models/regression/gradient_boosted_regression", xtrain, xtest, xvalidation, ytrain, ytest, yvalidation)
    model_data(RandomForestRegressor, hyperparameters_rfr, 'RandomForestRegressor', "models/regression/random_forest_regression", xtrain, xtest, xvalidation, ytrain, ytest, yvalidation)
    model_data(DecisionTreeRegressor, hyperparameters_dfr, 'DecisionTreeRegressor', "models/regression/decision_tree_regression", xtrain, xtest, xvalidation, ytrain, ytest, yvalidation)

#def find_best_model():

if __name__ == "__main__":
    evaluate_all_models()

# %%
