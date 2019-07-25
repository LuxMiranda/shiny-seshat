from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
import numpy as np

# x_0 = x_1 + x_2 + x_3
train = np.array([
    [ 5, 2, 2, 1],
    [10, 1, 2, 7],
    [ 3, 1, 1, 1],
    [ 8, 4, 2, 2]
])

test = np.array([
    [np.nan, 2, 4, 5],
    [np.nan, 4, 1, 2],
    [np.nan, 1, 10, 1]
])

y_actual = np.array([11, 7, 12])

## Prediction using IterativeImputer and  the 1-step fit_transform()
imputer     = IterativeImputer(estimator = LinearRegression())
fullData    = np.concatenate([train,test])
imputedData = imputer.fit_transform(fullData)
y_1imputed  = imputedData[-3:,0]

## Prediction using IterativeImputer and the 2-step fit() / transform()
imputer2     = IterativeImputer(estimator = LinearRegression())
imputer2     = imputer2.fit(train)
imputedTest  = imputer2.transform(test)
y_2imputed   = imputedTest[:,0]


print('Actual y:             {}'.format(y_actual))
print('1-step Imputed y:     {}'.format(y_1imputed))
print('2-step Imputed y:     {}'.format(y_2imputed))



