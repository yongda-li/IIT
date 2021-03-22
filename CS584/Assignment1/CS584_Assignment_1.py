from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
import numpy as np

def load_dataset():
    '''
    ** Do not modify this function. **
    Load diabetes dataset. We only use one feature and 60 instances.
    '''

    X, y = load_diabetes(return_X_y=True)
    return X[:60, 2], y[:60]

def train_test_split(X, y):
    '''
    Randomly split data into train and test set.
    Two thirds of the raw data will be the train set and one third of the raw data will be the test set.
    '''

    ######################################################################################
    # Full Mark: 10                                                                      #
    # TODO:                                                                              #
    # 1. shuffle the indices of data first.                                              #
    # (Hint: use numpy.arange and numpy.random.shuffle)                                  #
    # 2. select two thirds of the data as train set, the rest of data as test set.       #
    ######################################################################################
    index = np.arange(60)
    np.random.shuffle(index)
    X_train = X[index[:40]]
    X_test = X[index[40:]]
    y_train = y[index[:40]]
    y_test = y[index[40:]]


    #########################################################################
    #                       END OF YOUR CODE                                #
    #########################################################################
    
    return X_train, X_test, y_train, y_test

def plot_data(X, y):
    '''
    Draw scatter plot using raw data.
    '''

    #########################################################################
    # Full Mark: 10                                                         #
    # TODO:                                                                 #
    # 1. make a scatter plot of the raw data                                #
    # 2. set title for the plot                                             #
    # 3. set label for X,y axis                                             #
    # e.g.,                                                                 #
    #https://matplotlib.org/3.2.0/api/_as_gen/matplotlib.pyplot.scatter.html#
    #########################################################################
    fig = plt.figure()
    plt.scatter(X, y)
    plt.title('Raw Data')
    plt.xlabel('X')
    plt.ylabel('y')


    #########################################################################
    #                       END OF YOUR CODE                                #
    #########################################################################

    # return the plt object
    return plt

def cost_function(weights, X, y):
    '''
    Define the cost function.
    '''

    #########################################################################
    # Full Mark: 25                                                         #
    # TODO:                                                                 #
    # Implement the Mean Square Error function:                             #
    # https://en.wikipedia.org/wiki/Mean_squared_error#Mean                 #
    #                                                                       #
    # (Hint: Use numpy functions)                                           #
    #########################################################################
    num_instance = X.shape[0]
    t = X.dot(weights)
    cost = np.sum(np.square(y - t)) / num_instance

    
    #########################################################################
    #                       END OF YOUR CODE                                #
    #########################################################################

    # return cost
    return cost

def gradient_descent(weights, X, y):
    '''
    Update weights using gradient descent algorithm.
    '''

    # define your learning_rate and epoch
    lr = 0.1
    epoch = 20000
    
    # define cost
    cost_list = []
    
    num_instance = X.shape[0]
    # for loop
    for i in range(epoch):
        #########################################################################
        # Full Mark: 25                                                         #
        # TODO:                                                                 #
        # 1. update weights with learning rate lr                               #
        # 2. append the updated cost to cost list                               #
        # (Hint: Use numpy functions)                                           #
        #########################################################################
        cost = cost_function(weights,X,y)
        cost_list.append(cost)
        dw = (2 * np.sum((y - X.dot(weights)) * (-X[:,1]))) / num_instance
        db = (-2 * np.sum((y - X.dot(weights)))) / num_instance
        weights[0] += -lr * db
        weights[1] += -lr * dw


        #########################################################################
        #                       END OF YOUR CODE                                #
        ######################################################################### 
    # return updated weights and cost list
    return weights, cost_list

def plot_iteration(cost, epoch=20000):
    '''
    Plot the cost for each iteration.
    '''

    #########################################################################
    # Full Mark: 10                                                         #
    # TODO:                                                                 #
    # 1. plot the cost for each iteration                                   #
    # 2. set title and labels for the plot                                  #
    # (Hint: Use plt.plot function to plot and range(n))                    #
    #########################################################################
    x = np.arange(epoch)
    plt.plot(x,cost)
    plt.title('Gradient Descent')
    plt.xlabel('epoch')
    plt.ylabel('cost')
    

    
    #########################################################################
    #                       END OF YOUR CODE                                #
    #########################################################################
    
    # show plot
    plt.show()

def plot_final(weights, X, y):
    '''
    Draw the simple linear regression model.
    '''

    # draw the raw data first
    model_plot = plot_data(X, y)

    #########################################################################
    # Full Mark: 10                                                         #
    # TODO:                                                                 #
    # 1. create a series of x coordinates in proper range.                  #
    # (Hint: use numpy.arange)                                              #
    # 2. calculate y coordinates:                                           #
    #                         y = w * X + b                                 #
    # 3. plot the curve and set title                                       #
    #########################################################################
    index = np.arange(X.shape[0])
    y = X * weights[1] + weights[0]
    plt.plot(X[index], y, color='red')
    plt.title('Regression line')
    plt.xlabel('x')
    plt.ylabel('y')

    #########################################################################
    #                       END OF YOUR CODE                                #
    #########################################################################

    # show plot
    model_plot.show()

def print_test_error(weights, X, y_true):
    '''
    Use optimized weights to predict y, and print test error.
    '''

    #########################################################################
    # Full Mark: 10                                                         #
    # TODO:                                                                 #
    # 1. predict the target value y of X:                                   #
    #                            y = w * X + b                              #
    # 2. calculate the Mean Square Error using true y and predicted y       #
    #########################################################################
    y = X * weights[1] + weights[0]
    num_test = X.shape[0]
    error = np.sum(np.square(y_true - y)) / num_test


    #########################################################################
    #                       END OF YOUR CODE                                #
    #########################################################################

    # print test error
    print("Test error: %.4f" % error)
    return error

def main():
    '''
    ** Do not modify this function. **
    '''

    # Plot raw data points
    X, y = load_dataset()
    plot = plot_data(X, y)
    plot.show()

    # Split train and test set
    X = np.c_[np.ones(X.size), X]
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # initialize weight
    weights = np.ones(X_train.shape[1])

    # calculate training cost
    init_cost = cost_function(weights, X_train, y_train)
    print("Initial cost: %.4f" % init_cost)

    # gradient descent to find the optimal fit
    weights, cost_list = gradient_descent(weights, X_train, y_train)

    # draw the cost change for iterations
    plot_iteration(cost_list)

    # draw the final linear model
    # it is shown as a red line, you can change the color anyway
    plot_final(weights, X_train[:, 1], y_train)

    # Print test error
    print_test_error(weights, X_test[:, 1], y_test)

if __name__ == '__main__':
    main()