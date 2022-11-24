from custom_dataset import AirbnbNightlyPriceImageDataset, find_best_nn
from modelling import load_and_split_data, evaluate_all_classification_models, evaluate_all_regression_models, find_best_model, save_model
from torch.utils.data import DataLoader, random_split


data = load_and_split_data("bedrooms", ["ID", "Category", "Title", "Description", "Amenities", "Location", "url"])
xtrain, xtest, xvalidation, ytrain, ytest, yvalidation = data

model_type = 'Regression' 

dataset = AirbnbNightlyPriceImageDataset('bedrooms')
train_set_size = int(len(dataset) * 0.7)
val_set_size = int(len(dataset) * 0.15)
test_set_size = len(dataset) - train_set_size - val_set_size
train_set, val_set, test_set = random_split(dataset=dataset, lengths=[train_set_size, val_set_size, test_set_size])
train_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(dataset=train_set, batch_size=32)
test_loader = DataLoader(dataset=test_set, batch_size=32)


if __name__ == "__main__":
    if model_type == 'Regression':
        sgdr, gbr, rfr, dfr = evaluate_all_regression_models(xtrain, xtest, xvalidation, ytrain, ytest, yvalidation)
        regression_model, regression_hyperparameters, regression_metrics = find_best_model(model_type, sgdr, gbr, rfr, dfr)
    if model_type == 'Classification':
        log_reg, gbc, rfc, dfc = evaluate_all_classification_models(xtrain, xtest, xvalidation, ytrain, ytest, yvalidation)
        find_best_model(model_type, log_reg, gbc, rfc, dfc)

    best_model, best_metrics, best_hyperparameters = find_best_nn(train_loader, val_loader, test_loader)

    save_model('test', best_hyperparameters, best_model, best_metrics, "models/regression/neural_networks/")
    print('From the individual regression models the optimal models: ', regression_model)
    print('Its metrics were: ', regression_metrics)
    print('From the linear regression neural network the optimal models metrics was: ', best_metrics)