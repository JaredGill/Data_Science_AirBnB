import datetime
import inspect
import joblib
import json
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDRegressor, LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, accuracy_score, confusion_matrix 
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from tabular_data import load_airbnb



def load_and_split_data(label: str, str_cols: list):
    '''
    Function loads and splits data cleaned from tabular_data.py, standardises the features and splits data into subsets.
    Split is: 70% training, 15% validation, 15% test

    Returns:
    --------
    xtrain: numpy.ndarray
        Feature subset for training model
    ytrain: numpy.ndarray
        Label subset for training model
    xvalidation: numpy.ndarray
        Feature subset for testing different models and hyperparameters
    yvalidation: numpy.ndarray
        Label subset for testing different models and hyperparameters
    xtest: numpy.ndarray
        Feature subset to use on optimised model
    ytest: numpy.ndarray
        Label subset to use on optimised model
    '''
    airbnb = load_airbnb(label, str_cols)
    x = airbnb[0]
    y = airbnb[1]
    x = scale(x)
    #np.random.seed(42)
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3)
    xvalidation, xtest, yvalidation, ytest = train_test_split(xtest, ytest, test_size=0.5)
    return xtrain, xtest, xvalidation, ytrain, ytest, yvalidation

def simple_regression_model(model, xtrain, ytrain, xset, yset):
    '''
    Function loads in model and data, then trains model and computes R^2 and mse.
    Uses matplotlib to show models in scatter and line graph.

    Parameters:
    -----------
    model: regression model
        Input regression model with hyperparameters
    xtrain: numpy.ndarray
        Feature subset for training model
    ytrain: numpy.ndarray
        Label subset for training model
    xset: numpy.ndarray
        Feature subset for testing model(could use either validation or test set)      
    yset: numpy.ndarray
        Label subset for testing model(could use either validation or test set)
    '''

    #sgdr = SGDRegressor(alpha=0.0001, epsilon=0.01, eta0=0.1,penalty='elasticnet')
    #sgdr = model
    model.fit(xtrain, ytrain)
    score = model.score(xtrain, ytrain)
    print("Train R-squared:", score)
    test_score = model.score(xset, yset)
    print("Test R-squared:", test_score)

    #get predictions with feature test data(ypred)
    ypred = model.predict(xset)

    mse = mean_squared_error(yset, ypred)
    rmse = mean_squared_error(yset, ypred,squared=False)
    mae = mean_absolute_error(yset, ypred)
    print("MSE: ", mse)
    #print("RMSE: ", mse**(1/2.0))
    print("RMSE:", rmse)
    print("MAE:", mae)

    #notebook ver
    samples = len(ypred)
    plt.figure()
    plt.scatter(np.arange(samples), ypred, c='r', label='predictions')
    plt.scatter(np.arange(samples), yset, c='b', label='true labels', marker='x')
    plt.legend()
    plt.title("AirBnB validation set original and predicted data")
    plt.xlabel('Length of validation set')
    plt.ylabel('Number of Bedrooms')
    plt.show()

    # visualize the original and predicted data in a plot.
    x_ax = range(len(yset))
    plt.plot(x_ax, yset, label="original")
    plt.plot(x_ax, ypred, label="predicted")
    plt.title("AirBnB validation set original and predicted data")
    plt.xlabel('Length of validation set')
    plt.ylabel('Number of Bedrooms')
    plt.legend(loc='best',fancybox=True, shadow=True)
    plt.grid(True)
    plt.show() 
    
def custom_tune_regression_model_hyperparameters(model, train_features, train_labels, 
                                                        val_features, val_labels,
                                                        test_features, test_labels, 
                                                        hyperparameters_dict: dict):
    '''
    Custom function to go through every possible combination of hyperparameters and find the best scoring.
    Loads model and uses ParameterGrid to go through hyperparameters and return a list.
    Iterates over list:
        Appending the hyperparameters and checking that they are present in documentation for the SGDRegressor.
        Once checked it passes the filtered params in and trains model.
        Gets the scores and append to per_metrics dict
    
    Parameters:
    -----------
    model: regression model
        Input regression model with hyperparameters
    '''
    # iterate through every possible variation and get rmse from all, the best model & hyperparameters will be from rmse 
    sgdr = model()
    perf_metrics = {"hyperparameters": [], "validation_RMSE": [], "r2_score": []}
    grid = list(ParameterGrid(hyperparameters_dict))
    models = []
    for params in grid:
        mydict = params
        models.append(params)

        #check each key is present in parameters for method SGDRegressor
        filtered_params = {key: value for key, value in mydict.items() if key in [parameter.name for parameter in inspect.signature(SGDRegressor).parameters.values()]}
        
        #pass in dict as parameters
        sgdr = SGDRegressor(**filtered_params)
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
    best_model = SGDRegressor(**filtered_params)
    return best_model, best_hyperparameters, perf_metrics, best_rmse

def tune_regression_model_hyperparameters(model, xtrain, xtest, xvalidation, ytrain, ytest, yvalidation, hyperparameters_dict: dict):
    '''
    Function loads in model training data and hyperparameters.
    Then uses Gridsearchcv and fits the training data to return the best performing model and score

    Parameters:
    -----------
    model: regression model
        Input regression model with hyperparameters
    xtrain: numpy.ndarray
        Feature subset for training model
    ytrain: numpy.ndarray
        Label subset for training model
    hyperparameters_dict: dict
        Dict of hyperparameters for model      
    
    Returns:
    --------
    best_model: model
        The model with best hyperparameters passed into it
    best_hyperparameters: dict
        The best hyperparameters based on the score
    perf_metrics: dict
        The best score
    best_score: int
        The best score (regression: r-squared, classification: accuracy)
    '''
    regression_model = model()

    search = GridSearchCV(regression_model, param_grid=hyperparameters_dict, cv=5)
    result = search.fit(xtrain, ytrain)
    best_hyperparameters = result.best_params_
    #best_score = result.best_score_
    print('Best Score:', result.best_score_)
    print('Best Hyperparameters: ', result.best_params_)
    val_ypred = search.best_estimator_.predict(xvalidation)
    val_rmse = mean_squared_error(yvalidation, val_ypred,squared=False)
    val_r2 = r2_score(yvalidation, val_ypred)
    test_ypred = search.best_estimator_.predict(xtest)
    test_rmse = mean_squared_error(ytest, test_ypred, squared=False)
    test_r2 = r2_score(ytest, test_ypred)
    cv_results = search.cv_results_
    perf_metrics = {
        'train_r2_score': [], 
        'val_rmse_score': [],
        'val_r2_score': [],
        'test_rmse_score': [],
        'test_r2_score': [],
        }
    perf_metrics['train_r2_score'].append(result.best_score_)
    perf_metrics['val_rmse_score'].append(val_rmse)
    perf_metrics['val_r2_score'].append(val_r2)
    perf_metrics['test_rmse_score'].append(test_rmse)
    perf_metrics['test_r2_score'].append(test_r2)
    evaluation_score = perf_metrics['val_rmse_score']
    # perf_metrics['hyperparameters'].extend(cv_results.get('params'))
    # perf_metrics['neg_root_mean_squared_error_score'].extend(cv_results.get('mean_test_score'))
    
    best_model = model(**best_hyperparameters)
    return best_model, best_hyperparameters, perf_metrics, evaluation_score

def save_model(model_name: str, hyperparameters, model, metrics, foldername: str = "models/regression/linear_regression"):
    '''
    Function checks whether model is from pytorch or not.
    If it is a pytorch model the foldername is set to 'models/regression/neural_networks/'and saved using torch.save()
    If not the foldername is as unaltered and the function saves the model in a .joblib
    The hyperparameters and the metrics are saved in a .json

    Parameters:
    -----------
    model_name: str
        Name of the model
    hyperparameters: dict
        hyperparameters for model
    model: regression model
        Input regression model with hyperparameters
    metrics: dict
        Feature subset for training model
    foldername: str
        Name of folder files are to be saved into    
    '''
    curr_dir = os.getcwd()
    pytorch_check = bool
    try: 
        model.state_dict()
        pytorch_check = True
    except: 
        pytorch_check = False

    if pytorch_check == True:
        curr_datetime = datetime.datetime.now()
        # For files ':' is not an acceptable character
        #save as timestamp for folder
        datetime_str = curr_datetime.strftime('%Y') + '-' + curr_datetime.strftime('%m') + '-' + curr_datetime.strftime('%d') + '_' + curr_datetime.strftime('%H') + ';' + curr_datetime.strftime('%M') + ';' + curr_datetime.strftime('%S')
        foldername = 'models/regression/neural_networks/' + datetime_str
        target_path = os.path.join(curr_dir, foldername)
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        model_name = 'model.pt'
        save_path = os.path.join(target_path, model_name)
        torch.save(model.state_dict(), save_path)
        hyperparameters_filename = os.path.join(target_path, 'hyperparameters.json')
        metrics_filename = os.path.join(target_path, 'metrics.json')
        with open(hyperparameters_filename, 'w') as f:
            json.dump(hyperparameters, f)
        with open(metrics_filename, 'w') as f:
            # numpy.float32 wont convert to json so make them str instead
            json.dump(metrics, f)

    elif pytorch_check == False:
        target_path = os.path.join(curr_dir, foldername)
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        model_filename = os.path.join(target_path, f'{model_name}_model.joblib')
        joblib.dump(model, model_filename)
        hyperparameters_filename = os.path.join(target_path, f'{model_name}_hyperparameters.json')
        metrics_filename = os.path.join(target_path, f'{model_name}_metrics.json')
        with open(hyperparameters_filename, 'w') as f:
            json.dump(hyperparameters, f)
        with open(metrics_filename, 'w') as f:
            json.dump(metrics, f)
    
def tune_and_save_model(model_type, model, param_grid, model_name, folder_name, xtrain, xtest, xvalidation, ytrain, ytest, yvalidation):
    '''
    Calls the tune_regression_model_hyperparameters and save_model depending on the type of model passed as a string.

    Parameters:
    ----------
    model_type: str
        Name of the type of model either regression or classification
    model: sklearn model
        Input regression model with hyperparameters
    param_grid: dict
        Dict of list of hyperparameters for model
    model_name: str
        Name of the chosen sklearn model 
    folder_name: str
        Name of the folder to save model and metrics to
    
    Returns:
    --------
    gridsearch_evaluation_score: int
        The best score of all gridsearch combinations(val_rmse for regression, accuracy for classification)
    gridsearch_best_model: sklearn model
        The model with hyperparameters
    gridsearch_best_hyperparameters: dict
        The best hyperparameters based on score
    gridsearch_perf_metrics: dict
        The metrics for the best scoring combination
    '''
    if model_type == 'regression':
        gridsearch = tune_regression_model_hyperparameters(model, xtrain, xtest, xvalidation, ytrain, ytest, yvalidation, param_grid)
    elif model_type == 'classification':
        gridsearch = tune_classification_model_hyperparameters(model, xtrain, xtest, xvalidation, ytrain, ytest, yvalidation, param_grid)
    gridsearch_best_model, gridsearch_best_hyperparameters, gridsearch_perf_metrics, gridsearch_evaluation_score = gridsearch
    
    save_model(f"{model_name}", gridsearch_best_hyperparameters, gridsearch_best_model, gridsearch_perf_metrics, folder_name)
    return gridsearch_evaluation_score, gridsearch_best_model, gridsearch_best_hyperparameters, gridsearch_perf_metrics

def evaluate_all_regression_models(xtrain, xtest, xvalidation, ytrain, ytest, yvalidation):
    '''
    Function loads the data with load_and_split_data() and has custom dicts of hyperparameters based on the model type.
    Then it calls tune_and_save_model() on each model and returns its outputs.

    Returns:
    --------
    sgdr: tune_and_save_model() return values
        The SGDRegressor gridsearch_best_score, gridsearch_best_model, gridsearch_best_hyperparameters, gridsearch_perf_metrics
    gbr: tune_and_save_model() return values
        The GradientBoostingRegressor gridsearch_best_score, gridsearch_best_model, gridsearch_best_hyperparameters, gridsearch_perf_metrics
    rfr:tune_and_save_model() return values
        The RandomForestRegressor gridsearch_best_score, gridsearch_best_model, gridsearch_best_hyperparameters, gridsearch_perf_metrics
    dfr: tune_and_save_model() return values
        The DecisionTreeRegressor gridsearch_best_score, gridsearch_best_model, gridsearch_best_hyperparameters, gridsearch_perf_metrics
    '''
    # data = load_and_split_data("Price_Night", ["ID", "Category", "Title", "Description", "Amenities", "Location", "url"])
    # #ensure it only works when data has the same length as parameters
    # xtrain, xtest, xvalidation, ytrain, ytest, yvalidation = data
    #dict of dict with key = name of model, value = another dict
    # 2nd dict = model & hyperparameters
    # put hyperparams into yaml file (config)
    hyperparameters_sgdr =  {"learning_rate": ["constant", "adaptive", "optimal"], 
                            "eta0": [0.001, 0.005, 0.01, 0.05, 0.1, 0.5], 
                            "alpha": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1],
                            "penalty": ["l2", "l1", "elasticnet"],
                            #"loss": ["squared_error"]
                            } 
                                    # GradientBoostingRegressor
    hyperparameters_gbr = {'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1],
                            'subsample': [0.9, 0.5, 0.2, 0.1],
                            'n_estimators': [100, 250, 500],
                            'max_depth': [1, 2, 4, 6]
                            }
                                    # # RandomForestRegressor
    hyperparameters_rfr = {'criterion': ['squared_error'],
                            'min_samples_split' : [2, 4, 8],
                            'n_estimators': [100, 500, 1000, 1500],
                            'max_depth': [4, 6, 8, 10]
                            },
                                    # # DecisionTreeRegressor
    hyperparameters_dfr =  {'criterion': ['squared_error'],
                            'min_samples_split' : [2, 4, 8],
                            "max_features": ["log2","sqrt"],
                            'max_depth': [2, 4, 6, 8, 10], 
                            "min_weight_fraction_leaf":[0.1, 0.2, 0.3, 0.4, 0.5]
                            }
    
    sgdr = tune_and_save_model('regression', SGDRegressor, hyperparameters_sgdr, 'SGDRegressor', "models/regression/linear_regression", xtrain, xtest, xvalidation, ytrain, ytest, yvalidation)
    gbr = tune_and_save_model('regression', GradientBoostingRegressor, hyperparameters_gbr, 'GradientBoostingRegressor', "models/regression/gradient_boosted_regression", xtrain, xtest, xvalidation, ytrain, ytest, yvalidation)
    rfr = tune_and_save_model('regression', RandomForestRegressor, hyperparameters_rfr, 'RandomForestRegressor', "models/regression/random_forest_regression", xtrain, xtest, xvalidation, ytrain, ytest, yvalidation)
    dfr = tune_and_save_model('regression', DecisionTreeRegressor, hyperparameters_dfr, 'DecisionTreeRegressor', "models/regression/decision_tree_regression", xtrain, xtest, xvalidation, ytrain, ytest, yvalidation)
    #print(sgdr, gbr, rfr, dfr)
    return sgdr, gbr, rfr, dfr

def find_best_model(model_type: str, sgd, gb, rf, df):
    '''
    Function takes in the four models and finds the best model
    This is found via minimum option from the list of val_rmse is chosen as for regression, and the maximum for accuracy in classification.
    both of which indicate a better model the higher the value
    '''
    models = [sgd, gb, rf, df]
    model_evaluation_scores = []
    loaded_models = []
    hyperparameters = []
    perf_metrics = []
    for i in models:
        model_evaluation_scores.append(i[0])
        loaded_models.append(i[1])
        hyperparameters.append(i[2])
        perf_metrics.append(i[3])
    if model_type == 'Regression':
        best_value = min(model_evaluation_scores)
    elif model_type == 'Classification':
        best_value = max(model_evaluation_scores)
    else: 
        print('Not a valid type of ML model')
    best_value_pos = model_evaluation_scores.index(best_value)
    best_model = loaded_models[best_value_pos]
    best_hyperparameters = hyperparameters[best_value_pos]
    best_metrics = perf_metrics[best_value_pos]
    print('best values = ', model_evaluation_scores)
    print('loaded_models = ', loaded_models)
    print('hyperparameters = ', hyperparameters)
    print('perf_metrics = ', perf_metrics)
    print('---------------------------------------------')
    print(best_value)
    print(best_model, best_hyperparameters, best_metrics)
    return best_model, best_hyperparameters, best_metrics


def simple_classification():
    xtrain, xtest, xvalidation, ytrain, ytest, yvalidation = load_and_split_data("Category", ["ID", "Title", "Description", "Amenities", "Location", "url"])
    scaler = StandardScaler()
    lr = LogisticRegression()
    #should Principal Component Analysis(PCA) be used????
    model1 = Pipeline(steps=[('standardize', scaler),
                        ('log_reg', lr)])
    model1.fit(xtrain, ytrain)
    ypred = model1.predict(xvalidation)
    precision = model1.score(xtrain, ytrain)
    print("The precision of the model:", precision)
    test_precision = model1.score(xvalidation, yvalidation)
    print("The precision of the test data model:", test_precision)
    cf = confusion_matrix(yvalidation, ypred)
    cf_normalised = cf / cf.sum()
    visualise_confusion_matrix(cf_normalised)

def classification_eval_metrics(ytest, ypred, average_option: str = 'weighted'):
    '''
    Function calculates the performance of classification model.

    Parameters:
    ----------
    ytest: numpy.ndarray
        Label subset for test data
    ypred: numpy.ndarray
        Label prediction data
    average_option: str
        Option for scoring performance (weight, micro, or macro)
    '''
    prec_score = precision_score(ytest, ypred, average=average_option)
    rec_score = recall_score(ytest, ypred, average=average_option)
    f1score = f1_score(ytest, ypred, average=average_option)
    accuracyscore = accuracy_score(ytest, ypred)
    return prec_score, rec_score, f1score, accuracyscore

def visualise_confusion_matrix(confusion_matrix):
    display = ConfusionMatrixDisplay(confusion_matrix)
    display.plot()
    plt.show()

def tune_classification_model_hyperparameters(model, xtrain, xtest, xvalidation, ytrain, ytest, yvalidation, hyperparameters_dict: dict):
    '''
    Function loads in model training data and hyperparameters.
    Then uses Gridsearchcv and fits the training data to return the best performing model and score

    Parameters:
    -----------
    model: classification model
        Input regression model with hyperparameters
    xtrain: numpy.ndarray
        Feature subset for training model
    ytrain: numpy.ndarray
        Label subset for training model
    hyperparameters_dict: dict
        Dict of hyperparameters for model      
    
    Returns:
    --------
    best_model: model
        The model with best hyperparameters passed into it
    best_hyperparameters: dict
        The best hyperparameters based on the score
    perf_metrics: dict
        The best score
    best_score: int
        The best score (regression: r-squared, classification: accuracy)
    '''
    classification_model = model()

    search = GridSearchCV(classification_model, param_grid=hyperparameters_dict, cv=5)
    result = search.fit(xtrain, ytrain)
    best_hyperparameters = result.best_params_
    best_score = result.best_score_
    print('Best Score:', result.best_score_)
    print('Best Hyperparameters: ', result.best_params_)
    val_ypred = search.best_estimator_.predict(xvalidation)
    val_prec_score, val_rec_score, val_f1score, val_accuracyscore = classification_eval_metrics(yvalidation, val_ypred)
    test_ypred = search.best_estimator_.predict(xtest)
    test_prec_score, test_rec_score, test_f1score, test_accuracyscore = classification_eval_metrics(ytest, test_ypred)

    cv_results = search.cv_results_
    perf_metrics = {'train_accuracy': [], 'val_f1_score': [],'val_precision_score': [],'val_recall_score': [],'val_accuracy': [],
                    'test_f1_score': [],'test_precision_score': [],'test_recall_score': [],'test_accuracy': [],
                    }
    perf_metrics['train_accuracy'].append(result.best_score_)
    perf_metrics['val_f1_score'].append(val_f1score)
    perf_metrics['val_precision_score'].append(val_prec_score)
    perf_metrics['val_recall_score'].append(val_rec_score)
    perf_metrics['val_accuracy'].append(val_accuracyscore)
    perf_metrics['test_f1_score'].append(test_f1score)
    perf_metrics['test_precision_score'].append(test_prec_score)
    perf_metrics['test_recall_score'].append(test_rec_score)
    perf_metrics['test_accuracy'].append(test_accuracyscore)
    
    best_model = model(**best_hyperparameters)
    return best_model, best_hyperparameters, perf_metrics, best_score

def evaluate_all_classification_models(xtrain, xtest, xvalidation, ytrain, ytest, yvalidation):
    '''
    Function loads the data with load_and_split_data() and has custom dicts of hyperparameters based on the model type.
    Then it calls tune_and_save_model() on each model and returns its outputs.

    Returns:
    --------
    log_reg: tune_and_save_model() return values
        The LogisticRegression gridsearch_best_score, gridsearch_best_model, gridsearch_best_hyperparameters, gridsearch_perf_metrics
    gbc: tune_and_save_model() return values
        The GradientBoostingClassifier gridsearch_best_score, gridsearch_best_model, gridsearch_best_hyperparameters, gridsearch_perf_metrics
    rfc:tune_and_save_model() return values
        The RandomForestClassifier gridsearch_best_score, gridsearch_best_model, gridsearch_best_hyperparameters, gridsearch_perf_metrics
    dfc: tune_and_save_model() return values
        The DecisionTreeClassifier gridsearch_best_score, gridsearch_best_model, gridsearch_best_hyperparameters, gridsearch_perf_metrics
    '''
    # data = load_and_split_data("Category", ["ID", "Title", "Description", "Amenities", "Location", "url"])
    # #ensure it only works when data has the same length as parameters
    # xtrain, xtest, xvalidation, ytrain, ytest, yvalidation = data
    hyperparameters_log_reg =  {"solver": ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                                "C": [1, 5, 10],   
                                "penalty": ["l2", "l1", "elasticnet"],
                                "max_iter": [10000, 500000, 10000000, 1500000]
                                }
                                    # GradientBoostingRegressor
    hyperparameters_gbc = {'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1],
                            'subsample': [0.9, 0.5, 0.2, 0.1],
                            'n_estimators': [100, 250, 500],
                            'max_depth': [1, 2, 4, 6],
                            #'loss': ['log_loss']
                            }
                                    # RandomForestRegressor
    hyperparameters_rfc = {'criterion': ['log_loss'],
                            'min_samples_split' : [2, 4, 8],
                            'n_estimators': [100, 500, 1000, 1500],
                            'max_depth': [4, 6, 8, 10]
                            },
                                    # DecisionTreeRegressor
    hyperparameters_dfc =  {'criterion': ['log_loss'],
                            'min_samples_split' : [2, 4, 8],
                            "max_features": ["log2","sqrt"],
                            'max_depth': [2, 4, 6, 8, 10], 
                            "min_weight_fraction_leaf":[0.1, 0.2, 0.3, 0.4, 0.5]
                            }
    
    log_reg = tune_and_save_model('classification', LogisticRegression, hyperparameters_log_reg, 'LogisticRegression', "models/classification/logistic_regression", xtrain, xtest, xvalidation, ytrain, ytest, yvalidation)
    gbc = tune_and_save_model('classification', GradientBoostingClassifier, hyperparameters_gbc, 'GradientBoostingClassifier', "models/classification/gradient_boosted_classifier", xtrain, xtest, xvalidation, ytrain, ytest, yvalidation)
    rfc = tune_and_save_model('classification', RandomForestClassifier, hyperparameters_rfc, 'RandomForestClassifier', "models/classification/random_forest_classifier", xtrain, xtest, xvalidation, ytrain, ytest, yvalidation)
    dfc = tune_and_save_model('classification', DecisionTreeClassifier, hyperparameters_dfc, 'DecisionTreeClassifier', "models/classification/decision_tree_classifier", xtrain, xtest, xvalidation, ytrain, ytest, yvalidation)
    #print(sgdr, gbr, rfr, dfr)
    return log_reg, gbc, rfc, dfc


if __name__ == "__main__":
    data = load_and_split_data("Price_Night", ["ID", "Category", "Title", "Description", "Amenities", "Location", "url"])
    xtrain, xtest, xvalidation, ytrain, ytest, yvalidation = data
    sgdr, gbr, rfr, dfr = evaluate_all_regression_models(xtrain, xtest, xvalidation, ytrain, ytest, yvalidation)
    find_best_model('Regression', sgdr, gbr, rfr, dfr)

    data = load_and_split_data("Category", ["ID", "Title", "Description", "Amenities", "Location", "url"])
    xtrain, xtest, xvalidation, ytrain, ytest, yvalidation = data
    log_reg, gbc, rfc, dfc = evaluate_all_classification_models(xtrain, xtest, xvalidation, ytrain, ytest, yvalidation)
    find_best_model('Classification', log_reg, gbc, rfc, dfc)

    # data = load_and_split_data("bedrooms", ["ID", "Category", "Title", "Description", "Amenities", "Location", "url"])
    # xtrain, xtest, xvalidation, ytrain, ytest, yvalidation = data
    # model = SGDRegressor(alpha=0.01, eta0=0.001, learning_rate='adaptive', penalty='l1')
    # simple_regression_model(model, xtrain, ytrain, xvalidation, yvalidation)