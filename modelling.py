from pandasgui import show
from sklearn.linear_model import SGDRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import scale
from tabular_data import clean_tabular_data
from tabular_data import load_airbnb
import matplotlib.pyplot as plt

def simple_model():
    #df = ()
    airbnb = load_airbnb()
    x = airbnb[0]
    y = airbnb[1]
    show(x)
    show(y)
    x = scale(x)
    y = scale(y)

    #create train and test samples made of 15% of the data each
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=.15)

    sgdr = SGDRegressor(alpha=0.0001, epsilon=0.01, eta0=0.1,penalty='elasticnet')

    #train data and check the model accuracy score.
    sgdr.fit(xtrain, ytrain)
    score = sgdr.score(xtrain, ytrain)
    print("R-squared:", score)


    ypred = sgdr.predict(xtest)

    mse = mean_squared_error(ytest, ypred)
    print("MSE: ", mse)
    print("RMSE: ", mse**(1/2.0))

    # visualize the original and predicted data in a plot.
    x_ax = range(len(ytest))
    plt.plot(x_ax, ytest, label="original")
    plt.plot(x_ax, ypred, label="predicted")
    plt.title("AirBnB test and predicted data")
    plt.xlabel('X-axis')
    plt.ylabel('Price per night')
    plt.legend(loc='best',fancybox=True, shadow=True)
    plt.grid(True)
    plt.show() 

if __name__ == "__main__":
    simple_model()
