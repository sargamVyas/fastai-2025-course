'''
Data Preparation:
1. Check for N/A values and replace with mode (works for both category and continious) -> df.mode().iloc[0]
2. Undesrtand the numeric data -> df.describe()
3. using df.hist check lon tail dist as some mdoels especially linear dont like it then use log -> np.logp1
4. Check if numeric col are categorical, 
5. Convert categorical to dummy as we can mult coeff with string -> pd.get_dummies(df, col_name)
6. convert dataframe into tensor (for pytorch) while defining dep and indep column
Note: tensor.shape ->row * col, len(tensor.shape) -> rank -> vector is rank 1 matrix rank 2





Linear model from scratch steps:
1. After data cleaning convert data into 
1. Multiply coefficient with each column
'''

import os
from pathlib import Path
import numpy as np, pandas as pd
from torch import tensor



path = Path('data')
train_data = 'train.csv'