import os
import numpy as np
import pandas as pd
import time
import torch
import yaml
import torch.nn as nn
import torch.nn.functional as F
from modelling import save_model
from pandasgui import show
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from torch import save, load
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tabular_data import load_airbnb


class AirbnbNightlyPriceImageDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.tab_data = load_airbnb()
        self.cleaned_features = self.tab_data[0]
        self.cleaned_label = self.tab_data[1]
        #self.csv = pd.read_csv(csv_file)
        #transformations in init not get_item

    def __getitem__(self, idx):
        features = torch.tensor(self.cleaned_features.iloc[idx]).float()
        label = torch.tensor(self.cleaned_label.iloc[idx]).float()
        # print(features.dtype)
        # print('--------------')
        # print(label.dtype)
        #label = self.cleaned_label.iloc[idx]

        return (features, label)
    
    def __len__(self):
        return len(self.cleaned_features)

dataset = AirbnbNightlyPriceImageDataset()
# print(dataset[10])
# print(len(dataset))
# train set is 70%, test set is 30%
train_set_size = int(len(dataset) * 0.7)
val_set_size = int(len(dataset) * 0.15)
test_set_size = len(dataset) - train_set_size - val_set_size
train_set, val_set, test_set = random_split(dataset=dataset, lengths=[train_set_size, val_set_size, test_set_size])
train_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(dataset=train_set, batch_size=32)
test_loader = DataLoader(dataset=test_set, batch_size=32)

class LinearRegression(torch.nn.Module):#
    
    # The __init__() is used to define any network layers that the model will use.
    def __init__(self, 
                hidden_layer: int,
                linear_depth: int = 1
                ):
        super().__init__()
        #list comprehension for 
        #lin_layers = [nn.linear(hidden_layer, hidden_layer) for idx in range(linear_depth)]
        linear_layers = []
        self.linear_depth = linear_depth
        self.hidden_layer = hidden_layer
        for idx in range(self.linear_depth):
            linear_layers.append(nn.Linear(self.hidden_layer, self.hidden_layer))
            linear_layers.append(nn.ReLU())

        self.layers = nn.Sequential(
            nn.Linear(11, self.hidden_layer),
            #*linear_layers, ##(for list comprehension)
            # Use the rectified-linear activation function over features.
            # It compute the weighted sum of inputs and biases, which is in turn used to decide whether a neuron can be activated or not. 
            # It manipulates the presented data and produces an output for the neural network that contains the parameters in the data. 
            # helps to convert linear function to non-linear and converts complex data into simple functions so that it can be solved easily
            nn.ReLU(),
            linear_layers,
            nn.Linear(self.hidden_layer, 1)
        )
    # The forward() function is where the model is set up by stacking all the layers together.
    def forward(self, features):
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

    if optimiser_name == "torch.optim.SGD":
        optimiser= torch.optim.SGD(model.parameters(), learning_rate)
    #model.parameters() passes the model parameters from LinearRegression class init into the first arg for optimiser
    #optimiser = torch.optim.SGD(model.parameters(), lr=0.00001) # 0.00001

    # loss should go down when training model
    # stores each run of data in seperate folder called runs in start directory
    writer = SummaryWriter()
    batch_idx = 0
    train_start_time = time.time()

    metrics = {'RMSE_loss': [], 'R_squared': [], 'training_duration': [], 'inference_latency': []}
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
                loss = F.cross_entropy(prediction, labels)
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

    metrics['inference_latency'].append(np.average(inf_latency_batch))

    train_end_time = time.time()
    metrics['training_duration'].append(train_end_time - train_start_time)
    metrics['R_squared'].append(np.average(train_r2_batch))
    metrics['RMSE_loss'].append(np.average(train_rmse_batch))
    metrics['R_squared'].append(np.average(val_r2_list))
    metrics['RMSE_loss'].append(np.average(val_rmse_list))
    test_loss, test_r2, test_rmse = eval(model, test_loader)
    metrics['R_squared'].append(np.average(test_r2))
    metrics['RMSE_loss'].append(np.average(test_rmse))    
    print(metrics)
    model.test_loss = test_loss
    return model, metrics

# tqdm use for pytorch to visualize something
# use sklearns rsqaured
# length of time model takes for pred
# custom for inf_lat 
# Time.now() at start of batch then at end, take diff thats for one batch is infrence 
# time for training is all batchs

def eval(model, validation_set):
    #convert into numpy array
    losses=[]
    r2_list = []
    rmse_list = []
    for batch in validation_set:

        features, labels = batch
        #np_labels = labels.detach().numpy()
        #calls model on features
        prediction = model(features)
        #np_prediction = prediction.detach().numpy()
        loss = F.mse_loss(prediction, labels)
        losses.append(loss.detach())

        r2 = r2_score(labels.detach().numpy(), prediction.detach().numpy())
        r2_list.append(r2)
        rmse = loss**(1/2.0) #- mse to the power of 0.5
        rmse_list.append(rmse.detach())
    avg_loss = np.average(losses)
    avg_r2 = np.average(r2_list)
    avg_rmse = np.average(rmse_list)
    # print(avg_loss)
    # print(avg_r2)
    # print(avg_rmse)
    return avg_loss, avg_r2, avg_rmse

class LogisticRegression(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = torch.nn.Linear(3, 1)

    def forward(self, features):
        return F.sigmoid(self.linear_layer(features))

class ImageClassifier(nn.Module):
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
    with open('nn_config.yaml') as file:
        try:
            databaseConfig = yaml.safe_load(file)   
            #print(databaseConfig)
        except yaml.YAMLError as exc:
            print(exc)
        return databaseConfig

def generate_nn_configs():
    
    pass

config_details = get_nn_config()
optimiser_name = config_details['optimiser']
print(optimiser_name)
learning_rate = config_details['learning_rate']
print(learning_rate)
loss_func = config_details['loss_func']
print(loss_func)
hidden_layer = config_details['hidden_layer_width']
print(hidden_layer)
linear_depth = config_details['depth']

model = LinearRegression(hidden_layer, linear_depth)
train(model, train_loader, val_loader, test_loader, 200, optimiser_name, learning_rate, loss_func)

#save_model('linear_regression', hyperparameters, model, metrics, "models/regression/neural_networks/")


#if __name__ == "__main__":

# To save the model to disk to use later, just use the torch.save() function and voila!
### torch.save(model.state_dict(), 'model.ckpt'#)
