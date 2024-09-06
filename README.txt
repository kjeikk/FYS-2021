I struggled with github, and therefore only have these files outside an actual folder. Sorry about that.
The files linked to assignment 1 are: 
-  "README.txt" - this file
- "ass1_fullcode.py" - (the code behind the report delivered in canvas)

Requirements for use of the model "ass1_fullcode.py"
- Python 
- Libraries: Pandas, Numpy and Matplotlib

How to use: 
- The dataset is loaded from a csv file, and only pop and classical songs are kept. Then the selected features "liveness" and "loudness" are filtered out. The dataset is split into training set (80%) and test set (20%). 
- Logistic regression is performed using SDG. The models weights and biases are updated over 1000 epochs, with accuracy being tracked every 100th iteration (epoch)
- After training, the model is evaluated on the test set, and a confusion matrix is generated to analyse performance
- The code plots the cost (error) over epochs to visualize the training process

Notes:
- ensure that the correct path is provided
