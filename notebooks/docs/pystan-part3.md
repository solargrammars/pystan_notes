

```python
%load_ext autoreload
%autoreload 2
import pystan
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import arviz

sns.set()
np.random.seed(101)
```

    The autoreload extension is already loaded. To reload it, use:
      %reload_ext autoreload


# logistic regression


```python
# likelihood to be male based on height and weight
df = pd.read_csv('../data/HtWt.csv')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>male</th>
      <th>height</th>
      <th>weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0</td>
      <td>63.2</td>
      <td>168.7</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0</td>
      <td>68.7</td>
      <td>169.8</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0</td>
      <td>64.8</td>
      <td>176.6</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0</td>
      <td>67.9</td>
      <td>246.8</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1</td>
      <td>68.9</td>
      <td>151.6</td>
    </tr>
  </tbody>
</table>
</div>




```python
log_reg_dat = {
             'n': len(df),
             'male': df.male,
             'height': df.height,
             'weight': df.weight
            }
```


```python
fit = pystan.stan(file="../stan_scripts/logistic.stan", 
                  data=log_reg_dat, 
                  iter=2000, 
                  chains=1)
```

    WARNING:pystan:DeprecationWarning: pystan.stan was deprecated in version 2.17 and will be removed in version 3.0. Compile and use a Stan program in separate steps.
    INFO:pystan:COMPILING THE C++ CODE FOR MODEL anon_model_8a6cdeb95a51e84955a773dea19dead7 NOW.



```python
df_trace = pd.DataFrame(fit.extract(['c', 'b', 'a']))
```


```python
fit.extract().keys()
```




    odict_keys(['a', 'b', 'c', 'lp__'])




```python
arviz.plot_trace(fit)
```




    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x7f6ef0822e50>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f6edd03fdd0>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x7f6edcffbb10>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f6edcfb6dd0>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x7f6edcf73b90>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f6edcf2fe50>]],
          dtype=object)




![png](docs/pystan-part3_files/docs/pystan-part3_7_1.png)



```python

```
