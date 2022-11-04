import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from pandasgui import show
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader, random_split
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

    def __getitem__(self, idx):
        # feature_data = self.cleaned_features.iloc[idx]
        # features = torch.tensor(feature_data)
        # label_data = self.cleaned_label.iloc[idx]
        # label = label_data

        price_pop = self.cleaned_label.pop('Price_Night')
        total_data = self.cleaned_features
        total_data['Price_Night'] = price_pop
        show(total_data)
        indexed_data = total_data.iloc[idx]
        features = torch.tensor(indexed_data[:11])
        label = indexed_data[-1]
        print(type(label))
        return (features, label)
    
    def __len__(self):
        return len(self.cleaned_features)

dataset = AirbnbNightlyPriceImageDataset()
print(dataset[10])
print(len(dataset))
train_set_size = int(len(dataset) * 0.7)
valid_set_size = len(dataset) - train_set_size
train_set, test_set = random_split(dataset=dataset, lengths=[train_set_size, valid_set_size])
train_loader = DataLoader(dataset=train_set, batch_size=4, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=4, shuffle=True)
print(train_loader)
# example = train_loader[0]
# print(example)

class LinearRegression(torch.nn.Module):#
    def __init__(self):
        super().__init__()
        self.linear_layer = torch.nn.Linear(830, 1)
    
    def forward(self, features):
        return self.linear_layer(features)

def train(model, data_loader, num_epochs: int):
    for epoch in range(num_epochs):
        for batch in train_loader:
            features, labels = batch
            prediction = model(features)
            loss = F.mse_loss(prediction, labels)
            loss.backward()
            print(loss)
            # optimisation step


#if __name__ == "__main__":
