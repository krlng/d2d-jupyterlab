import os
from datetime import datetime as dt

import numpy as np
import pandas as pd
import sklearn as skl

# Pandas display options
pd.set_option('display.float_format', lambda x: '%.3f' % x)
import IPython.display as ipd

# Set random seed 
RSEED = 42

# Visualizations
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (25, 5)
%matplotlib inline
#plt.style.use('fivethirtyeight')
plt.rcParams['font.size'] = 18

import seaborn as sns
cm = sns.light_palette("green", as_cmap=True)
