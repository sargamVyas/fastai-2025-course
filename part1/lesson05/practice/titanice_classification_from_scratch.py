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
1. Tell pytirch what do you need derivates for -> coeffs.requires_grad_() -> gradient for coeff

Notes:
Try not to use manual_seed as it helps to understand how your data is behaving
Normalization technique: divide by max , sub by mean and divide by std dev 
Underscore in dunciotn is in place operation so requires_grad_() will make the change permanently
'''

import os
from pathlib import Path
import numpy as np, pandas as pd
from torch import tensor



path = Path('data')
train_data = 'train.csv'