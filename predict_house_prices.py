from __future__ import division
import numpy as np
import pandas as pd
from pandas.tools.plotting import scatter_matrix
from sklearn import linear_model 
from sklearn import cross_validation
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.learning_curve import validation_curve
import matplotlib.pyplot as plt


'''
*********************************************************************************************************************
                                 Functions
*********************************************************************************************************************
'''

def eval_performance(model, X_trn, y_trn, X_tst, y_tst):
  
  print("\nEvaluating model: %s" % model)
  
  predictions_train = model.predict(X_trn)
  residuals_train = predictions_train - y_trn
  predictions_test = model.predict(X_tst)
  residuals_test = predictions_test - y_tst
  
  # RSS 
  RSS_train = (residuals_train ** 2).sum()
  RSS_test = (residuals_test ** 2).sum()
  print("Residual sum of squares (train): %d" % RSS_train)
  print("Residual sum of squares (test): %d" % RSS_test)
  
  # Mean squared error
  print('MSE (train): %.2f' % mean_squared_error(y_trn, predictions_train))
  print('MSE (test): %.2f' % mean_squared_error(y_tst, predictions_test))
  
  # R^2 = 1 - (residual sum of squares / total sum of squares) = 1 - MSE/Var(y)
  # R_squared_test = 1 - RSS_test / ((y_test - y_test.mean()) **2).sum()
  print('R^2 (train): %.2f' % model.score(X_trn, y_trn))
  print('R^2 (test): %.2f\n' % model.score(X_tst, y_tst))

def coefficient_path(model, alphas, X, y):
  model = model
  coefs = []
  for a in alphas:
    model.set_params(alpha=a)
    model.fit(X, y)
    coefs.append(model.coef_)
  plt.figure()
  ax = plt.gca()
  ax.set_color_cycle(['b', 'r', 'g', 'c', 'k', 'y', 'm'])
  ax.plot(alphas, coefs)
  ax.set_xscale('log')
  ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
  plt.xlabel('alpha')
  plt.ylabel('weights')
  plt.title('Coefficients as a function of the regularization')
  plt.axis('tight')
  plt.show()


def plot_validation_curve(estimator, X, y, param_name, param_range):
  
    train_scores, test_scores = validation_curve(estimator, X, y, param_name, param_range)
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.figure()
    plt.plot(param_range, train_mean, 
         color='blue', marker='o', 
         markersize=5, label='training accuracy')

    plt.fill_between(param_range, train_mean + train_std,
                 train_mean - train_std, alpha=0.15,
                 color='blue')

    plt.plot(param_range, test_mean, 
         color='green', linestyle='--', 
         marker='s', markersize=5, 
         label='validation accuracy')

    plt.fill_between(param_range, 
                 test_mean + test_std,
                 test_mean - test_std, 
                 alpha=0.15, color='green')

    plt.grid()
    plt.xscale('log')
    plt.legend(loc='lower right')
    plt.xlabel('Parameter C')
    plt.ylabel('Accuracy')
    plt.ylim([0., 1.0])
    plt.tight_layout()
    plt.show()

  
'''
*********************************************************************************************************************
                                 Load data
*********************************************************************************************************************
'''

df = pd.read_csv('../data/kc_house_data_small.csv')

X_cols = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade', 'yr_built', 'yr_renovated', 'zipcode']
X_cols_sqm = ['bedrooms', 'bathrooms', 'sqm', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade', 'yr_built', 'yr_renovated', 'zipcode']

df['sqm'] = df['sqft_living'] * 0.092903

X_df = df[X_cols_sqm].values

y_df = df['price'].values
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_df, y_df, test_size=0.2, random_state=0)



# keep these for plotting
X_train_orig = X_train
X_test_orig = X_test


scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X_mean = scaler.mean_
print X_mean
X_scale = scaler.scale_
print X_scale



print '''
*********************************************************************************************************************
                                 Linear regression
*********************************************************************************************************************
'''

lreg_model = linear_model.LinearRegression()
lreg_model.fit(X_train, y_train)

print('Coefficients: \n', lreg_model.coef_)
print('Intercept: \n', lreg_model.intercept_)

eval_performance(lreg_model, X_train, y_train, X_test, y_test)


print '''
*********************************************************************************************************************
                                 Lasso regression with CV
*********************************************************************************************************************
'''


alphas = [0.01, 0.1, 1.0, 10.0, 10.0, 100.0, 100.0, 1000.0, 10000.0, 100000.0]

lasso_model_cv = linear_model.LassoCV(alphas=alphas)
lasso_model_cv.fit(X_train, y_train)  

print('Coefficients: \n', lasso_model_cv.coef_)
print('Intercept: \n', lasso_model_cv.intercept_)
eval_performance(lasso_model_cv, X_train, y_train, X_test, y_test)
print('Best Lasso alpha: {}\n'.format(lasso_model_cv.alpha_)) 

#coefficient_path(linear_model.Lasso(), alphas, X_train, y_train)
#plot_validation_curve(linear_model.Lasso(), X_train, y_train, 'alpha', alphas)


print '''
*********************************************************************************************************************
                                 Lasso regression - alpha = 1e5
*********************************************************************************************************************
'''

lasso_model = linear_model.Lasso(alpha=1e5)
lasso_model.fit(X_train, y_train)  

print('Coefficients: \n', lasso_model.coef_)
print('Intercept: \n', lasso_model.intercept_)
eval_performance(lasso_model, X_train, y_train, X_test, y_test)





print '''
*********************************************************************************************************************
                                 Linear regression sqm only
*********************************************************************************************************************
'''

X_train_sqm = X_train[:,2:3]
X_test_sqm = X_test[:,2:3]

lreg_model_sqm = linear_model.LinearRegression()
lreg_model_sqm.fit(X_train_sqm, y_train)

print('Coefficients: \n', lreg_model_sqm.coef_)
print('Intercept: \n', lreg_model_sqm.intercept_)

eval_performance(lreg_model_sqm, X_train_sqm, y_train, X_test_sqm, y_test)

plt.figure()
plt.xkcd()
plt.scatter(X_test_orig[:,2:3], y_test,  color='cyan', label='actual price')
plt.plot(X_test_orig[:,2:3], lreg_model_sqm.predict(X_test_sqm), color='red', linewidth=3, label='predicted price')
plt.legend()
plt.xlabel('square metres')
plt.ylabel('price')
plt.ylim(0)
plt.title('Predicting house prices from square metres')
plt.show()



print '''
*********************************************************************************************************************
                                 Polynomial features, using just sqm
*********************************************************************************************************************
'''


poly_2 = PolynomialFeatures(degree=2, include_bias=False)

X_train_sqm_poly_2 = poly_2.fit_transform(X_train_sqm)
X_test_sqm_poly_2 = poly_2.fit_transform(X_test_sqm)

lreg_model_poly = linear_model.LinearRegression()
lreg_model_poly.fit(X_train_sqm_poly_2, y_train)
print('Coefficients: \n', lreg_model_poly.coef_)
print('Intercept: \n', lreg_model_poly.intercept_)

eval_performance(lreg_model_poly, X_train_sqm_poly_2, y_train, X_test_sqm_poly_2, y_test)

plt.figure()
plt.xkcd()
plt.scatter(X_test_orig[:,2:3], y_test,  color='cyan', label='actual price')
plt.plot(X_test_orig[:,2:3], lreg_model_poly.predict(X_test_sqm_poly_2), color='red', linewidth=3, label='predicted price')
plt.legend()
plt.xlabel('square metres')
plt.ylabel('price')
plt.ylim(0)
plt.title('Predicting house prices from square metres')
plt.show()


print '''
*********************************************************************************************************************
                                 Get back the original coefficients for unstandardized data
*********************************************************************************************************************
'''


'''

y = a + b*((x - mu) / sigma)

y = a + b/sigma * (x - mu)

y = a + (b/sigma) * x  - (b/sigma) * mu

y = a - (b/sigma) * mu + b/sigma * x

new_intercept = a - (b/sigma) * mu
new_coef = b/sigma

'''

new_intercept_lreg_model_sqm = lreg_model_sqm.intercept_ - (lreg_model_sqm.coef_ / scaler.scale_[2]) * scaler.mean_[2]
new_coef_lreg_model_sqm = lreg_model_sqm.coef_ / scaler.scale_[2] 

print('sqm only regression, intercept and coefficient: {}, {}\n'.format(new_intercept_lreg_model_sqm, new_coef_lreg_model_sqm))

new_intercept_lasso = lasso_model.intercept_ -(lreg_model.coef_[2] / scaler.scale_[2]) * scaler.mean_[2] - (lreg_model.coef_[6] / scaler.scale_[6]) * scaler.mean_[6] - (lreg_model.coef_[8] / scaler.scale_[8]) * scaler.mean_[8]
new_coef_lasso = lasso_model.coef_[[2,6,8]] / scaler.scale_[[2,6,8]] 

print('lasso, intercept and coefficients: {}, {}\n'.format(new_intercept_lasso, new_coef_lasso))


new_intercept_poly = lreg_model_poly.intercept_ - (lreg_model_poly.coef_[0] / scaler.scale_[2]) * scaler.mean_[2] - (lreg_model_poly.coef_[1] / scaler.scale_[2]) * scaler.mean_[2]
new_coef_poly = lreg_model_poly.coef_ / scaler.scale_[2] 

print('sqm squared only regression, intercept and coefficients: {}, {}\n'.format(new_intercept_poly, new_coef_poly))

new_intercept_lreg_all = lreg_model.intercept_ -(lreg_model.coef_[0] / scaler.scale_[0]) * scaler.mean_[0] - (lreg_model.coef_[1] / scaler.scale_[1]) * scaler.mean_[1] - (lreg_model.coef_[2] / scaler.scale_[2]) * scaler.mean_[2] -(lreg_model.coef_[3] / scaler.scale_[3]) * scaler.mean_[3] - (lreg_model.coef_[4] / scaler.scale_[4]) * scaler.mean_[4] - (lreg_model.coef_[5] / scaler.scale_[5]) * scaler.mean_[5] -(lreg_model.coef_[6] / scaler.scale_[6]) * scaler.mean_[6] - (lreg_model.coef_[7] / scaler.scale_[7]) * scaler.mean_[7] - (lreg_model.coef_[8] / scaler.scale_[8]) * scaler.mean_[8] -(lreg_model.coef_[9] / scaler.scale_[9]) * scaler.mean_[9] - (lreg_model.coef_[10] / scaler.scale_[10]) * scaler.mean_[10] - (lreg_model.coef_[11] / scaler.scale_[11]) * scaler.mean_[11]

new_coef_lreg_all = lreg_model.coef_ / scaler.scale_ 

print('linear regression all variables, intercept and coefficients: {}, {}\n'.format(new_intercept_lreg_all, new_coef_lreg_all))


'''
import numpy as np
from sklearn import preprocessing
from sklearn import linear_model

X = np.array(np.random.normal(100,10,1000)).reshape(-1,1)
y = 100 + X * 2

scaler = preprocessing.StandardScaler().fit(X)
X_std = scaler.transform(X)
scaler.mean_
scaler.scale_

lreg_model = linear_model.LinearRegression()
lreg_model.fit(X_std,y)
lreg_model.coef_
lreg_model.intercept_

X[0:5]
X_std[0:5]
y[0:5]


'''


