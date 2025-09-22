# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

<img width="1002" height="817" alt="image" src="https://github.com/user-attachments/assets/08836fbd-3598-4a8d-a31b-892329c5d97a" />

## DESIGN STEPS

### STEP 1
Load and preprocess the dataset (handle missing values, encode categorical features, scale numeric data).

### STEP 2
Split the dataset into training and testing sets, convert to tensors, and create DataLoader objects.

### STEP 3
Build the neural network model, train it with CrossEntropyLoss and Adam optimizer, then evaluate with confusion matrix and classification report.

## PROGRAM
### Name: HAREESH R
### Register Number: 212223230068

```python
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size,32)
        self.fc2 = nn.Linear(32,16)
        self.fc3 = nn.Linear(16,8)
        self.fc4 = nn.Linear(8,4)


    def forward(self,x):
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      x = F.relu(self.fc3(x))
      x = self.fc4(x)
      return x

```
```python
model = PeopleClassifier(input_size=X_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.01)
```
```python
def train_model(model, train_loader,criterion,optimizer,epochs):
  for epoch in range(epochs):
    model.train()
    for X_batch, y_batch in train_loader:
      optimizer.zero_grad()
      output = model(X_batch)
      loss = criterion(output,y_batch)
      loss.backward()
      optimizer.step()

    if (epoch + 1) % 10 == 0:
      print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
```

## Dataset Information

<img width="1212" height="764" alt="image" src="https://github.com/user-attachments/assets/fc80fdc6-460c-4e61-84ca-1f1e0df08764" />


## OUTPUT

### Confusion Matrix

<img width="1382" height="680" alt="image" src="https://github.com/user-attachments/assets/e7d21261-ecf6-4824-8fd6-951de6534faf" />



### Classification Report

<img width="1086" height="594" alt="image" src="https://github.com/user-attachments/assets/f82d05a3-cf56-47cb-8ed9-602514cfe92b" />



### New Sample Data Prediction

<img width="960" height="333" alt="image" src="https://github.com/user-attachments/assets/851548a7-159c-4724-9e70-207ad368b8c1" />



## RESULT
The neural network model was successfully built and trained to handle classification tasks.
