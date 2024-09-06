import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# loading and preparing data
path = "C:/Users/kezie/OneDrive/Skrivebord/fys-2021/FYS-2021/Assignment1/SpotifyFeatures.csv/data.csv"
data = pd.read_csv(path, delimiter=",", encoding='utf-8')
print("the dimension of the dataset is", np.shape(data), "samples and features")

# filtering dataset
data_set = data[(data["genre"] == "Pop") | (data["genre"] == "Classical")]
amount_pop = data[data['genre'] == 'Pop']
amount_class = data[data["genre"] == "Classical"]
print("Pop genre contains", len(amount_pop), "songs, and classical genre contains", len(amount_class), "songs")

# creating matrices with the specified features, replacing genre name with 1 or 0 
dataset1 = data_set[["genre", "liveness", "loudness"]]
data_set1 = dataset1.replace({"Pop": 1, "Classical": 0})

# shuffling the dataset so that it doesnt appear in a set order

data_set1 = data_set1.sample(frac=1, random_state=42).reset_index(drop=True)
# now splitting the dataset into 80% training set and 20% test set by introducing split index

# implementing split index and converting tuple to numpy array
split_index = int(0.8 * len(data_set1))
train_set_x = data_set1.iloc[:split_index, [1, 2]].to_numpy()
train_set_y = data_set1.iloc[:split_index, [0]].to_numpy()
test_set_x = data_set1.iloc[split_index:, [1, 2]].to_numpy()
test_set_y = data_set1.iloc[split_index:, [0]].to_numpy()

# Transpose
trsx_t = np.transpose(train_set_x)
trsy_t = np.transpose(train_set_y).flatten()        # in order to heck that it is one dimensional (vector)
tesx_t = np.transpose(test_set_x)
tesy_t = np.transpose(test_set_y).flatten()

# comparing the songs according to loudness and liveness
pop_data = data[data['genre'] == 'Pop']
classical_data = data[data["genre"] == "Classical"]

pop_liveness = pop_data['liveness']
pop_loudness = pop_data['loudness']

classical_liveness = classical_data['liveness']
classical_loudness = classical_data['loudness']

plt.scatter(pop_liveness, pop_loudness, color = "blue", s = 0.2)
plt.scatter(classical_liveness, classical_loudness, color = "red", s = 0.2)
plt.xlabel("liveness")
plt.ylabel("loudness")
plt.title("Liveness vs loudness")
plt.show()

# from the plot i see that some songs are overlapping and will be harder
# for the machine to correctly seperate into correct categories. 


# Initialize weights and bias
# had to make bias numpy array in order for the code to run
np.random.seed(42)
weight = 0.1 * np.random.randn(trsx_t.shape[0], 1)
bias = np.array([0.0])  

# defining functions

# struggled with the sigmoid function, 
# implemented clip to prevent overflow
def sigmoid(z):
    z = np.clip(z, -500, 500)  
    return 1 / (1 + np.exp(-z))    

def y_hat(trsx_t, weight, bias):
    return np.dot(weight.T, trsx_t) + bias      

# creating a small epsillon to avoid log 0
# cross entropy function
def cost_function(trsy_t, y_hat):
    eps = 1e-10                  
    y_hat = np.clip(y_hat, eps, 1 - eps)
    loss = -np.mean(trsy_t * np.log(y_hat) + (1 - trsy_t) * np.log(1 - y_hat))  
    return loss

# sending the weights and b forward into the sigmoid function
def forward(trsx_t, weight, bias):      
    z = np.dot(weight.T, trsx_t) + bias
    A = sigmoid(z) 
    return A

# defining the derivatives of w and b (use the formula in report)
# evaluating the 
def backward(trsx_t, trsy_t, y_hat):       
    m = trsx_t.shape[1]                    
    error = y_hat - trsy_t
    d_weight = (1/m) * np.dot(trsx_t, error.T)
    d_bias = (1/m) * np.sum(error)
    return d_weight, d_bias

# updating w and b
def update(weight, bias, d_weight, d_bias, learning_rate = 0.01):     
    weight -= learning_rate * d_weight
    bias -= learning_rate * d_bias
    return weight, bias

# defining the treshold
def roundValue(A):
    return np.uint8(A > 0.5)           
 
def accuracy(y_hat, trsy_t):
    correct_pred = np.sum(y_hat == trsy_t)
    accur = (correct_pred / len(trsy_t)) * 100
    return accur

# training the model
epochs = 1000
learning_rate = 0.01
losses, accuracies = [], []

for e in range(epochs):
    A = forward(trsx_t, weight, bias)
    y_hat = roundValue(A)
    acc = accuracy(y_hat, trsy_t)

    if acc > 100:
        print(f"Warning: Accuracy > 100% at iteration {e}")
        print("y_hat:", y_hat)
        print("trsy_t:", trsy_t)
        break 

    loss = cost_function(trsy_t, A)
    losses.append(loss)

    d_weight, d_bias = backward(trsx_t, trsy_t, A)
    weight, bias = update(weight, bias, d_weight, d_bias, learning_rate)
    accuracies.append(acc)

    if e % 100 == 0:
        print(f'Iteration {e}: Accuracy = {acc}%')


# plotting the losses (error) over epochs
plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.title("Cost over Epochs")
plt.grid()
plt.show()

# testing the model
test_predictions = forward(tesx_t, weight, bias)
test_predictions_rounded = roundValue(test_predictions)
test_accuracy = accuracy(test_predictions_rounded, tesy_t)
print(f"Model accuracy on the test set: {test_accuracy:.2f}%")

# implementing confusion matrix 

true_labels = tesy_t                      # defining the setes i want to compare 
tpr = test_predictions_rounded    

# Compute confusion matrix values
TP = np.sum((tpr == 1) & (true_labels == 1))  # acount for true positives
TN = np.sum((tpr == 0) & (true_labels == 0))  # true negatives
FP = np.sum((tpr == 1) & (true_labels == 0))  # false positives
FN = np.sum((tpr == 0) & (true_labels == 1))  # false negatives

# creating the confusion matrix
conf_matrix = np.array([[TP, FN],
                        [FP, TN]])

print("Confusion Matrix:")
print(conf_matrix)

