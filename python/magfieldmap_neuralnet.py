import pandas as pd
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import csv

# read the data
def get_device():
    device = 'cpu'
    return device
  
device = get_device()
df_all = pd.read_csv('magfieldmap3.csv', dtype=float).dropna()

df_test = df_all.iloc[0:100, :]
df_train = df_all.iloc[100:, :]

# seperate labels from data
train_labels = df_train.iloc[:, 3:] #Bx, By, Bz
train_images = df_train.iloc[:, 0:3] #x, y, z
test_labels = df_test.iloc[:, 3:]
test_images = df_test.iloc[:, 0:3]

print(train_images)
print(train_labels)

train_images = torch.from_numpy(train_images.to_numpy()).float()
train_labels = torch.squeeze(torch.from_numpy(train_labels.to_numpy()).float())
test_images = torch.from_numpy(test_images.to_numpy()).float()
test_labels = torch.squeeze(torch.from_numpy(test_labels.to_numpy()).float())
print(train_images.shape, train_labels.shape)
print(test_images.shape, test_labels.shape)

class Mag_net(nn.Module):

  #2 hidden layers
    def __init__(self,input_features=3,hidden1=3, hidden2=20,out_features=3):
        super().__init__()
        self.f_connected1 = nn.Linear(input_features,hidden1)
        self.f_connected2 = nn.Linear(hidden1,hidden2)
        self.out = nn.Linear(hidden2,out_features)

        
    #forward function
    def forward(self,x):
        x = F.relu(self.f_connected1(x))
        x = F.relu(self.f_connected2(x))
        x = self.out(x)
        return x

model = Mag_net()

# loss function (Mean Squared Error Loss)
loss_function = nn.MSELoss()   
# optimization
optimizer = torch.optim.Adam(model.parameters(),lr = 0.01)  


#training
epochs=100

final_losses=[]

for i in range(epochs):

    i= i+1
   
    y_pred=model.forward(train_images)

    loss=loss_function(y_pred,train_labels)

    final_losses.append(loss)
    #loss printed every 10 epochs
    if i % 10 == 0:

        print("Epoch number: {} and the loss : {}".format(i,loss.item()))

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()


#prediction function
def predict(data):
  
    predict_data = data

    #converting to float
    if type(predict_data[0]) == int:
      predict_data[0] = float(predict_data[0])
    if type(predict_data[1]) == int:
      predict_data[1] = float(predict_data[1])
    if type(predict_data[2]) == int:
      predict_data[2] = float(predict_data[2])
      
    #printing input values
    print("INPUT")
    print()
    print("x: " + str(predict_data[0]))
    print("y: " + str(predict_data[1]))
    print("z: " + str(predict_data[2]))
    print()
    
    #convert to tensor
    predict_data_tensor = torch.tensor(predict_data)
    
    prediction_value    = model(predict_data_tensor)
    
    #converting to np array
    prediction = prediction_value.cpu().detach().numpy()
    
    #printing the prediction
    print("PREDICTED OUTPUT")
    print()
    print("Bx: " + str(prediction[0]))
    print("By:" + str(prediction[1])) 
    print("Bz: " + str(prediction[2])) 
    
