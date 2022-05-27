# https://youtu.be/fk7bzKFDmk8
"""
MAE vs MSE for machine learning metrics and loss functions
"""
#Understanding MAE / MSE metrics
import numpy as np

true = [234,285,324,248,423,345, 422, 345, 367, 285]
predicted = [248,310,340,265,403,325, 400, 322, 387, 300]

#Let us define our own MAE function
def mae(y_true, predictions):
    y_true, predictions = np.array(y_true), np.array(predictions)
    return np.mean(np.abs(y_true - predictions))


print(mae(true, predicted))

#Using pre-built function from sklearn
from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(true, predicted))


#MSE
mse = np.square(np.subtract(true, predicted)).mean()
print(mse)

from sklearn.metrics import mean_squared_error
print(mean_squared_error(true, predicted))


############################
#Let us add a couple of outliers
###################################
#First data point is an outlier, with a large difference between true and predicted
true = [234,285,324,248,423,345, 422, 345, 367, 285]
predicted = [450,310,340,265,403,325, 400, 322, 387, 300]

print(mae(true, predicted))
print(mean_squared_error(true, predicted))

#########################################
#Using MAE and MSE as loss functions
################################
from matplotlib import pyplot as plt
#MAE
x = np.arange(-100, 100, 1)
y_mae = np.abs(x)

plt.plot(x, y_mae, "blue")
plt.grid(True, which="both")
plt.show()

#MSE
y_mse = np.square(x)

plt.plot(x, y_mae, "blue", label="MAE")
plt.plot(x, y_mse, "red", label="MSE")
plt.legend(loc="lower right")
plt.grid(True, which="both")
plt.axis([-20, 20, 0, 100])
plt.show()


#Huber loss
#For loss values below come number use MSE and for loss above 
#some number, use MAE
delta1 = 1.5  #if delta=1 Huber loss = MAE. 
delta2 = 4
mse_huber = 0.5 * np.square(x)
mae_huber1 = delta1 * (np.abs(x) - 0.5 * delta1)
mae_huber2 = delta2 * (np.abs(x) - 0.5 * delta2)
y_huber1 = np.where(np.abs(x) <= delta1, mse_huber, mae_huber1)
y_huber2 = np.where(np.abs(x) <= delta2, mse_huber, mae_huber2)

plt.plot(x, y_mae, "blue", label="MAE")
plt.plot(x, y_mse, "red", label="MSE")
plt.plot(x, y_huber1, "green", label="Huber d=1.5")
plt.plot(x, y_huber2, "limegreen", label="Huber d=4")
plt.legend(loc="lower right")
plt.grid(True, which="both")
plt.axis([-100, 100, 0, 100])
plt.show()

###########################################

#Loss functions

# MAE loss function
def mae_loss(y_pred, y_true):
    abs_error = np.abs(y_pred - y_true)
    sum_abs_error = np.sum(abs_error)
    loss = sum_abs_error / y_true.size
    return loss

# MSE loss function
def mse_loss(y_pred, y_true):
    squared_error = (y_pred - y_true) ** 2
    sum_squared_error = np.sum(squared_error)
    loss = sum_squared_error / y_true.size
    return loss

# Huber loss function
def huber_loss(y_pred, y, delta=1.0):
    huber_mse = 0.5*(y-y_pred)**2
    huber_mae = delta * (np.abs(y - y_pred) - 0.5 * delta)
    return np.where(np.abs(y - y_pred) <= delta, huber_mse, huber_mae)


