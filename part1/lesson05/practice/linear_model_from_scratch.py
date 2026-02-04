'''
Data Preparation:
1. Check for N/A values and replace with mode (works for both category and continious) -> df.mode().iloc[0]
2. Undesrtand the numeric data -> df.describe()
3. using df.hist check long tail dist as some models especially linear dont like it then use log -> np.logp1
4. Check if numeric col are categorical, 
5. Convert categorical to dummy as we cannot mult coeff with string -> pd.get_dummies(df, col_name)
6. convert dataframe into tensor (for pytorch) while defining dep and indep column
Note: tensor.shape ->row * col, len(tensor.shape) -> rank -> vector is rank 1, matrix rank 2

Setting up linear model:
1. Initialize random coeff -> one coeff for each col -> torch.rand(t_indep.shape[1])
2. Multiply matrix(data) * vector(coeff) using broadcasting because we did element wise multi. So its like vector is broadcasted no of ro times
3. After multiplication you add the rows together (basic idea of linear model) -> before sum and mult check if any of the col has higher magnitude compared to others(if yes then mormalize the col before mult)
4. This is out pred -> pred = (t_indep*coeff).sum(axis=1)
5. Calculate Loss: torch.abs(pred-t_dep).mean()

Gradient descent steps:
1. Tell pytorch what do you need derivates for -> coeffs.requires_grad_() -> gradient/derivative for coeff and its in-place
2. Perform the Backward Pass After calculating your loss, you need to trigger the math.
    2.1 loss.backward() -> This calculates the gradient (the slope) for every tensor where requires_grad=True.
    2.2 The results are stored in coeffs.grad.
3. Update the Coefficients (The Step) Once you have the gradients, you move the coefficients in the opposite direction of the slope to reduce the loss.
    3.1 Use a Learning Rate (lr) to control how big of a step you take.
    3.2 coeffs.data.sub_(coeffs.grad * lr) -> This is the manual way to update weights without PyTorch tracking the update itself as part of the gradient.
4. Zero the Gradients PyTorch adds new gradients to old ones (accumulation), so you must empty the "bucket" before the next loop.
    4.1 coeffs.grad.zero_() -> Resets the gradients to zero so they don't add up across iterations.

Adding Sigmoid:
* why? To make our prediction range between 0 and 1 which makes it easier to optimize
* torch.sigmoid((indeps*coeffs).sum(axis=1)) -> using while calculating prediciton
* If you have binary dependent variable always chuck it through sigmoid

Neural Net from scratch:
1. everything has to be in matrix form so that matrix mult can be done
2. deffine rand_var as -> layer_1 = torch.rand(n_coeff, n_hidden); layer_2 = torch.rand(n_hidden, 1) -> in this only one hidden layer
3. Add relu after every layer to add non-linearity in it
** For deep learning you just add n-hidden layers

Activation function - when and where to use them
* ReLU is for Learning: Put it between layers so the model can learn complex, non-linear shapes. It's the "non-linearity" tool.
* Sigmoid is for Deciding: Put it at the very end if you need your output to be a probability between 0 and 1 (like in the Titanic survival model Jeremy Howard uses).
*Final Layer Rule:
    **Regression (predicting a number): Usually no activation at the end.
    **Binary Classification: Use Sigmoid at the end.
    **Multi-class Classification: Use Softmax at the end.
Note: A common mistake is putting a ReLU after the final layer when you want to predict a range. If you do that, your model can never "correct" itself for being too low if the output is already stuck at zero!


Notes:
* Try not to use manual_seed as it helps to understand how your data is behaving
* Normalization technique: divide by max , sub by mean and divide by std dev 
* Underscore in dunciotn is in place operation so requires_grad_() will make the change permanently
* torch.no_grad(): When you are updating your coeffs or calculating accuracy, use this context manager. It saves memory because you don't need to calculate gradients for the update part of the code—only for the model part.
* Why sub_?: We subtract because the gradient points "uphill." To get to the bottom of the valley (minimum loss), we have to go the opposite way!

Element-wise vas matmul:
    Element-wise (*): Multiplies matching positions ($A_{1,1} \times B_{1,1}$); requires identical shapes or broadcasting.
    Matrix Mult (@): Does the "multiply then sum" (Dot Product) in one step; the columns of the first matrix must match the rows of the second.
    (t_indep*coeff).sum(axis=1) = val_indep@coeffs
'''

import os
from pathlib import Path
import numpy as np, pandas as pd
from torch import tensor
import torch

#---Configurations---
DATA_PATH = Path('data')
LEARNING_RATE = 0.01
EPOCHS = 1

#---Log transformation logic to make col normally distributed---
def apply_log_transform(df, threshold=0.75):
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        current_skew = df[col].skew()

        if current_skew > threshold:
            df[col] = np.log1p(df[col])

    return df

def normalize_col(df):
    numeric_col = df.select_dtypes(include=[np.number]).columns

    for col in numeric_col:
        col_max = df[col].max()
        if col_max != 0:
            df[col] = df[col] / col_max
    return df


#---Data Loading and Preprocessing---
def preprocess_data(df, modes):

    #1. Cleaning the df by checking n/a and relacing with mode
    df.fillna(modes, inplace=True)

    #2.Checking for lefttail or righttail and using log to restructure it
    df = apply_log_transform(df)

    #3. Creating dummies for categorical value
    df = pd.get_dummies(df, drop_first=True, dtype=float)

    #4. Normalizing values
    df = normalize_col(df)

    #5. Converting dataframe to tensors
    if 'Survived' in df.columns:
        y = torch.tensor(df['Survived'].values).float()
        x = torch.tensor(df.drop('Survived', axis=1).values).float()
    else:
        # If it's the test set without labels
        y = None
        x = torch.tensor(df.values).float()
    
    return x, y

#---Intializing weights---
def init_weights(n_in):
    weights = (torch.rand(n_in) - 0.5) * 0.1
    return weights.requires_grad_()


#---Calculate Prediction/Forward pass--
def calc_pred(weights, inputs):
    return inputs @ weights

#---Calculate loss---
def calc_loss(preds, targets):
    return torch.abs(preds.squeeze() - targets.squeeze()).mean()

#---Calculate gradient descent
def one_epoch(weights, train_x, train_y, lr):
    
    preds = calc_pred(weights, train_x)

    loss = calc_loss(preds, train_y)

    loss.backward()

    with torch.no_grad():
        weights.sub_(weights.grad * lr)

        weights.grad.zero()

    return loss.item()



def main():
    train_df = pd.read_csv(DATA_PATH/ 'train.csv')
    test_df = pd.read_csv(DATA_PATH/ 'test.csv')

    train_modes = train_df.mode().iloc[0]

    train_x, train_y = preprocess_data(train_df, modes=train_modes)

    n_features = train_x.shape[1]

    weights = init_weights(n_features, 0)

    # 6. The Training Loop
    print("Starting Training...")
    for epoch in range(EPOCHS):
        # Call your 'one_epoch' function which handles the forward/backward/update
        loss = one_epoch(weights, train_x, train_y, LEARNING_RATE)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss {loss:.4f}")

    # 7. Final Evaluation
    # Use your final weights to see how you did on the test set
    final_test_loss = calc_loss(calc_pred(weights, test_x), test_y)
    print(f"Final Test Loss: {final_test_loss:.4f}")

#Defining main function
if __name__ == "__main__":
    main()