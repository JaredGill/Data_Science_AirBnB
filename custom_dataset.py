import os
import time
import torch
import yaml
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from modelling import save_model
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import r2_score
from torch import save, load
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tabular_data import load_airbnb


class AirbnbNightlyPriceImageDataset(Dataset):
    '''
    Class inherits a parent class All datasets that represent a map from keys to data samples should subclass it.
    '''
    def __init__(self, label):
        super().__init__()
        self.tab_data = load_airbnb(label)
        self.cleaned_features = self.tab_data[0]
        self.cleaned_label = self.tab_data[1]
        # show(self.cleaned_features)
        # show(self.cleaned_label)
        #self.csv = pd.read_csv(csv_file)
        #transformations in init not get_item

    def __getitem__(self, idx):
        '''
        Returns an item of the dataset at the index provided. 

        Parameters:
        -----------
        idx: int
            The desired Index of the dataset
        
        Returns:
        --------
        features: torch.tensor
            The features of the dataset
        label: torch.tensor
            The labels of the dataset    
        '''
        features = torch.tensor(self.cleaned_features.iloc[idx]).float()
        label = torch.tensor(self.cleaned_label.iloc[idx]).float()
        return (features, label)
    
    def __len__(self):
        '''
        Returns the length of the dataset.

        Returns:
        --------
        len(self.cleaned_features): int
            The number of rows in dataset
        '''
        return len(self.cleaned_features)

class LinearRegression(torch.nn.Module):#
    
    def __init__(self, 
                config
                ):
        '''
        # The __init__() is used to define any network layers that the model will use.
        '''
        super().__init__()
        self.layers = nn.Sequential(
            config
        )
    def forward(self, features):
        '''
        Builds the model by stacking all the layers together.

        Parameters: 
        -----------
        features: torch.tensor
            Features of the dataset
        
        Returns:
        --------
        self.layers(features):
            The built model

        '''
        return self.layers(features)

def train(model, 
        train_loader, 
        val_loader,
        test_loader, 
        num_epochs: int, 
        optimiser_name, 
        learning_rate,
        loss_name
        ):
    '''
    Trains the neural network model with dataloaders and hyperparameters given.
    The model is trained in batches of training dataloader and then looped over by a number of epochs.
    The validation set is evaluated after every epoch using eval(), and the test set is evaluated once at the end.
    For every batch of the training set, the loss is calcualted and written to the SummaryWriter() to be viewed on TensorBoard.
    Similarly the validation set loss is also but on a seperate graph in Tensorboard after every epoch.
    To save the metrics each is datasets rmse and r2 are appended to a metrics dict. 
    Some have to be in a list from whih the average is calclated then passed to dict

    Parameters:
    -----------
    model: Class
        torch.nn class for building the model
    train_loader: DataLoader
        Train dataset
    val_loader: DataLoader
        Validation dataset
    test_loader: DataLoader
        Testing dataset
    num_epochs: int
        Number of loops to go through the train and val datasets
    optimiser_name: str
        The name of the optimiser function
    learning_rate: float
        The learning rate of model
    loss_name: str
        The name of the loss function
    '''

    if optimiser_name == "torch.optim.SGD":
        optimiser= torch.optim.SGD(model.parameters(), learning_rate)
    #model.parameters() passes the model parameters from LinearRegression class init into the first arg for optimiser
    #optimiser = torch.optim.SGD(model.parameters(), lr=0.00001) # 0.00001

    # loss should go down when training model
    # stores each run of data in seperate folder called runs in start directory
    writer = SummaryWriter()
    batch_idx = 0
    train_start_time = time.time()

    metrics = {'RMSE_loss_train': 0, 'RMSE_loss_val': 0,'RMSE_loss_test': 0,'R_squared_train': 0, 'R_squared_val': 0,'R_squared_test': 0,
                'training_duration': 0, 'inference_latency': 0}
    train_r2_batch = []
    train_rmse_batch = []
    inf_latency_batch = []
    val_r2_list = []
    val_rmse_list = []

    for epoch in range(num_epochs):
        inf_latency_start_time = time.time()
        for batch in train_loader:
            features, labels = batch
            prediction = model(features)
            # compare prediction to labels to obtain loss
            # mse for regression, cross entropy for multiclass classification
            # cross-entropy formula describes how closely the predicted distribution is to the true distribution as a probability
            if loss_name == "mse_loss":
                loss = F.mse_loss(prediction, labels)
            elif loss_name == "cross_entropy":
                loss = F.cross_entropy(prediction, labels.long())
            else:
                print(f'{loss_name} is not a valid option. ')
            
            r2 = r2_score(labels.detach().numpy(), prediction.detach().numpy())
            train_r2_batch.append(r2)
            rmse = loss**(1/2.0) #- mse to the power of 0.5
            train_rmse_batch.append(rmse.detach())
            # backpropagation aims to minimize the loss function by adjusting network’s weights and biases.
            # It reduce error rates and make the model reliable by increasing its generalization to prevent overfitting
            loss.backward()

            # optimisers like stochastic gradient descent help prevent the model stopping at a local minima 
            # by giving weight to prev accumulated gradient over current gradient when at the local minima
            optimiser.step()

            #zeros the gradients so the gradient isnt a combo of old gradient and new
            optimiser.zero_grad()

            #writes each loss to the tensorboard with the batch index as x-axis
            writer.add_scalar('loss', loss.item(), batch_idx)
            batch_idx +=1
            print(f'Epoch:{epoch} loss is {loss.item()}')

        inf_latency_end_time = time.time()
        inf_latency_batch.append(inf_latency_end_time - inf_latency_start_time)

        val_loss, val_r2, val_rmse = eval(model, val_loader)#eval model on validation set
        val_r2_list.append(val_r2)
        val_rmse_list.append(val_rmse)
        writer.add_scalar('loss/Val', val_loss, batch_idx)

    metrics['inference_latency'] = np.average(inf_latency_batch)

    train_end_time = time.time()
    metrics['training_duration'] = train_end_time - train_start_time
    # numpy.float32 wont convert to json so make them normal floats instead
    metrics['R_squared_train'] = float(np.average(train_r2_batch))
    metrics['RMSE_loss_train'] = float(np.average(train_rmse_batch))
    metrics['R_squared_val'] = float(np.average(val_r2_list))
    metrics['RMSE_loss_val'] = float(np.average(val_rmse_list))
    test_loss, test_r2, test_rmse = eval(model, test_loader)
    metrics['R_squared_test'] = float(np.average(test_r2))
    metrics['RMSE_loss_test'] = float(np.average(test_rmse))    
    #print(metrics)
    model.test_loss = test_loss
    return model, metrics

def eval(model, dataset):
    losses=[]
    r2_list = []
    rmse_list = []
    for batch in dataset:

        features, labels = batch

        prediction = model(features)
        loss = F.mse_loss(prediction, labels)
        losses.append(loss.detach())

        r2 = r2_score(labels.detach().numpy(), prediction.detach().numpy())
        r2_list.append(r2)
        rmse = loss**(1/2.0)
        rmse_list.append(rmse.detach())
    avg_loss = np.average(losses)
    avg_r2 = np.average(r2_list)
    avg_rmse = np.average(rmse_list)
    return avg_loss, avg_r2, avg_rmse

class LogisticRegression(torch.nn.Module):
    '''
    To be finished/unused
    '''
    def __init__(self, possible_classes):
        super().__init__()
        self.linear_layer = torch.nn.Linear(3, possible_classes)

    def forward(self, features):
        #F.sigmoid for binary, F.softmax for multiclass
        return F.softmax(self.linear_layer(features))

class ImageClassifier(nn.Module):
    '''
    To be finished/unused
    '''
    def __init__(self):
        # conv2d or convolutional neural network that performs convolution on the image is able to 
        # outperform a regular neural network in which you would feed the image by flattening it

        # They apply a filter kernal on the image at the first positoin(top left) and calculates output value by elements-wise multiplication of 2d tensors.
        # For RGB images the convolution operation is done individually for the three color channels, and their results are added together for the final output
        # The filter then moves to a stride value (1 pixel over)
        # Result may be smaller size as kernals may not perfectly fit in corners of image
        self.model = torch.nn.Sequential(
            # Convolutional layer which has 3 input channels(rgb) each channel is a seperate tensor, 32 filters of shape

            nn.Conv2d(1, 32, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            # Flattens 3d tensor to 1d vector for linear function
            nn.Flatten(),
            # Pass input shape (64 channels from last conv layer), multiplied by (the images heigh -2 pixels for each conv2d layer, and width -2) output layer is equal to labels
            torch.nn.Linear(64*(28-6)*(28-6), 10)
        )

    def forward(self, features):
        return self.model(features)

def get_nn_config():
    '''
    Reads in configuration .yaml file as a dict to be used for neural network

    Returns:
    -------
    databaseConfig: dict
        Configuration details
    '''
    with open('nn_config.yaml') as file:
        try:
            databaseConfig = yaml.safe_load(file)   
            #print(databaseConfig)
        except yaml.YAMLError as exc:
            print(exc)
        return databaseConfig

def generate_nn_configs():
    '''
    Generates all possible combinations of hyperparameters from a dict.

    Returns:
    --------
    configs: list
        List of all combinations
    '''

    config_dict = {'optimiser': ['torch.optim.SGD'],
                'learning_rate': [0.01, 0.001, 0.0001, 0.00001],
                'hidden_layer_width': [16, 32],
                'depth': [1, 2],
                'loss_func': ['mse_loss']}
    grid = list(ParameterGrid(config_dict))
    configs = []
    for params in grid:
        configs.append(params)
    #print(configs)
    return configs

def find_best_nn(train_loader, val_loader, test_loader):
    '''
    Calls on generate_nn_configs() to retrieve all possible hyperparameter combinations.
    For every combination of hyperparameters it creates an ordered dict to pass into model class and trains it.
    Model, metrics, hyperparams and 'RMSE_loss_test' metric are appended to individual lists.
    The index of the best 'RMSE_loss_test', is used to get the rest of details for the best model

    Parameters:
    -----------
    train_loader: DataLoader
        Train dataset
    val_loader: DataLoader
        Validation dataset
    test_loader: DataLoader
        Testing dataset
    
    Returns:
    --------
    best_model: torch.nn.class
        The best performing trained model
    best_metrics:
        The metrics of the best performing model
    best_hyperparameters:
        The hyperparameters of the best performing model
    '''
    configs_list = generate_nn_configs()
    metrics_list = []
    trained_model_list = []
    rmse_metric = []
    hyperparameters = []
    for config in configs_list:
        hyperparameters.append(config)

        #Parse the details from dict key
        optimiser_name = config['optimiser']
        learning_rate = config['learning_rate']
        loss_func = config['loss_func']
        hidden_layer = config['hidden_layer_width']
        linear_depth = config['depth']

        #create ordered dict
        config_dict = OrderedDict()
        config_dict['input'] = nn.Linear(11, hidden_layer)
        for idx in range(linear_depth):
            rel_idx = f'relu{idx}'
            config_dict[rel_idx] = nn.ReLU()
            idx += 1
            od_idx = f'layer{idx}'
            config_dict[od_idx] = nn.Linear(hidden_layer, hidden_layer)
            idx +=1
        config_dict[f'layer{linear_depth}'] = nn.Linear(hidden_layer, 10)
        linear_depth +=1 
        config_dict[f'relu{linear_depth}'] = nn.ReLU()
        config_dict['output'] = nn.Linear(10, 1)
        #neurons for class last layer == to input classes
        #softmax for ouptput
        model = LinearRegression(config_dict)
       
        try:
            trained_model, metrics = train(model, train_loader, val_loader, test_loader, 200, optimiser_name, learning_rate, loss_func)
            trained_model_list.append(trained_model)
            metrics_list.append(metrics)
            rmse_metric.append(metrics['RMSE_loss_val'])
        except:
            print('Error in neural network.')
            # trained_model_list.append('Error in neural network.')
            # metrics_list.append('Error in neural network.')
            # rmse_metric.append('Error in neural network.')
    best_rmse = min(rmse_metric)
    print('rmse_metrics: ', rmse_metric)
    print('metrics_list: ', metrics_list)
    print('hyperparameters: ', hyperparameters)
    idx = rmse_metric.index(best_rmse)
    best_model = trained_model_list[idx]
    best_metrics = metrics_list[idx]
    best_hyperparameters = hyperparameters[idx]
    print('########################################')
    print(best_model, best_metrics, best_hyperparameters)

    return best_model, best_metrics, best_hyperparameters

if __name__ == "__main__":
    dataset = AirbnbNightlyPriceImageDataset('Price_Night')
    print(dataset[10])
    print(len(dataset))
    train_set_size = int(len(dataset) * 0.7)
    val_set_size = int(len(dataset) * 0.15)
    test_set_size = len(dataset) - train_set_size - val_set_size
    train_set, val_set, test_set = random_split(dataset=dataset, lengths=[train_set_size, val_set_size, test_set_size])
    train_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset=train_set, batch_size=32)
    test_loader = DataLoader(dataset=test_set, batch_size=32)
    best_model, best_metrics, best_hyperparameters = find_best_nn(train_loader, val_loader, test_loader)
    save_model('test', best_hyperparameters, best_model, best_metrics, "models/regression/neural_networks/")
