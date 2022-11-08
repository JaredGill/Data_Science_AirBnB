import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pandasgui import show
from sklearn.model_selection import train_test_split
from torch import save, load
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tabular_data import load_airbnb

#data = pd.read_csv('AirbnbDataSci/structured/AirBnbData.csv')
#print(data)
# t = clean_tabular_data()
# show(t)
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
print(dataset[10])
print(len(dataset))
# train set is 70%, test set is 30%
train_set_size = int(len(dataset) * 0.7)
valid_set_size = len(dataset) - train_set_size
print(train_set_size)
#validatoin set here
train_set, test_set = random_split(dataset=dataset, lengths=[train_set_size, valid_set_size])
train_loader = DataLoader(dataset=train_set, batch_size=16, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=16, shuffle=True)

class LinearRegression(torch.nn.Module):#
    
    # The __init__() is used to define any network layers that the model will use.
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(11, 121),
            # Use the rectified-linear activation function over features
            nn.ReLU(),
            nn.Linear(121, 1)
        )
    # The forward() function is where the model is set up by stacking all the layers together.
    def forward(self, features):
        return self.layers(features)

def train(model, data_loader, num_epochs: int = 10):

    #model.parameters() passes the model parameters from LinearRegression class init into the first arg for optimiser
    optimiser = torch.optim.SGD(model.parameters(), lr=0.001)
    writer = SummaryWriter()
    batch_idx = 0
   # loss_values = []

    for epoch in range(num_epochs):
        for batch in data_loader:
            features, labels = batch
            # print(features.shape)
            # print(labels.shape)
            prediction = model(features)
            loss = F.mse_loss(prediction, labels)
            #loss_values.append(loss.item())
            loss.backward()
            #print(loss.item())

            # optimisation step
            optimiser.step()
            #zeros the gradients so the gradient isnt a combo of old gradient and new
            optimiser.zero_grad()
            writer.add_scalar('loss', loss.item(), batch_idx)
            batch_idx +=1
        print(f'Epoch:{epoch} loss is {loss.item()}')
        #eval func here
        #eval model on validation set
        #eval(model, valset, batch_idx)
    
    return loss_values

class LogisticRegression(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = torch.nn.Linear(3, 1)

    def forward(self, features):
        return F.sigmoid(self.linear_layer(features))

class ImageClassifier(nn.Module):
    def __init__(self):
        self.model = torch.nn.Sequential(
            # Convolutional layer which has 1 input(for black and white), 32 filters of shape
            torch.nn.Conv2d(1, 32, (3, 3)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, (3, 3)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, (3, 3)),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            # Pass input shape (64 channels from last conv layer), multiplied by (the images shape -2 pixels for each conv2d layer)squared, output layer is equal to labels
            torch.nn.Linear(64*(28-6)*(28-6), 10)
        )

    def forward(self, features):
        return self.model(features)


model = LinearRegression()
train(model, train_loader, 10)
train(model, test_loader, 10)
#if __name__ == "__main__":



# To save the model to disk to use later, just use the torch.save() function and voila!
### torch.save(model.state_dict(), 'model.ckpt'#)
