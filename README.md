# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Explain the problem statement

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name: Lokesh M

### Register Number:212224230142

```
import torch
```

```
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
```

```
df = pd.read_csv("C:\\Users\\admin\\DL\\Exp-1.csv")
df
```

```
x = df[["Input"]].values
y = df[["Output"]].values
xt,xst,yt,yst = train_test_split(x,y,test_size=0.2,random_state=42)
```

```
scale1 = MinMaxScaler()
scale2=MinMaxScaler()
xt = scale1.fit_transform(xt)
xst = scale2.fit_transform(xst)
```

```
xt = torch.FloatTensor(xt)
xst = torch.FloatTensor(xst)
yt = torch.FloatTensor(yt)
yst = torch.FloatTensor(yst)
```

```
# Tesing for new input
new = scale1.transform([[16]])
new = torch.FloatTensor(new)

pred = model(new)
print(pred.item())
```

```
# Evaluating loss for testing data
with torch.no_grad():
    pred=model(xst)
    loss_test=criterion(pred,yst)
    print(loss_test)
```

```
# Plot the loss curve

plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during Training")
plt.show()
```

### Dataset Information
Include screenshot of the generated data

<img width="201" height="568" alt="Screenshot 2026-07-24 202940" src="https://github.com/user-attachments/assets/4ef89c07-1411-4712-aec5-3b3016fa71f3" />


### OUTPUT

<img width="281" height="532" alt="Screenshot 2026-07-24 202950" src="https://github.com/user-attachments/assets/385f6374-c06e-4eab-934e-46c4b59f8a7f" />

### Training Loss Vs Iteration Plot
Include your plot here

<img width="1057" height="623" alt="Screenshot 2026-07-24 203336" src="https://github.com/user-attachments/assets/a26959ff-676a-40cc-b01a-1e02c5feb7cb" />


### New Sample Data Prediction
Include your sample input and output here

<img width="217" height="72" alt="Screenshot 2026-07-24 203408" src="https://github.com/user-attachments/assets/ff824670-2aed-4a5b-abec-3f9098d2a958" />


## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
