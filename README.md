# Data_Science_AirBnB

The aim of this project is to build a framework for the evaluation of various machine learning models which can then be applied to different datasets. It involves an initial step of exploratory data analysis on the first dataset - AirBnB data, which is cleaned and seperated into text, tabular and image data. Regression and classification models are then trained, tuned and evaluated to discover the best performing model which are saved as a .joblib file. Finally a configurable neural network was created.

## Data Preparation
- The tabular dataset was read in from a csv file and functions were defined to remove/edit the dataframe based on the column in its parameters.
    - The tabular data had rows removed if certain columns were NaN or some NaN values were set to a specific int.
    - The text data within the dataset had its several string descriptions combined into 1 string in its specific column.
    - These functions accumulated in a load_airbnb() function which outputs the features and labels based on the input parameters.
    ```python
    def load_airbnb(label: str = "Price_Night", str_cols: list = ["ID", "Category", "Title", "Description", "Amenities", "Location", "url"]):
    ```
    - Here the label parameter isolates the specified column, the str_cols parameter columns are removed from the dataframe which leaves the rest as features.
- The image folder and files were downloaded and resized to the same height and width based on the smallest images height, whilst maintaining the aspect ratio. 
- Once the data is cleaned it is split into 3 subsets: train, validation and test.
```python
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3)
xvalidation, xtest, yvalidation, ytest = train_test_split(xtest, ytest, test_size=0.5)
```
-   The train set(70%) is used for model optimisation
-   the validation set(15%) is used to make decisions about the model(e.g. which is the best model)
-   The test set(15%) is used to show how the model will perform on unseen data

## Regression 
- Regression models were chosen, trained on the same dataset and evaluated with gridsearchcv with custom hyperparameters to determine the optimal model. 
### Evaluation
- Loss functions provide a measurement on the models performance.
- The root mean squared error was the loss function chosen to determine the best model. More can be found about it here: https://www.brainstobytes.com/mean-absolute-error-vs-root-mean-square-error/
- Several other metrics such as Rsquared and mean absolute error can also be tracked. (https://www.mygreatlearning.com/blog/r-square/#:~:text=R-square%20is%20a%20goodness-of-fit%20measure%20for%20linear%20regression,variable%20on%20a%20convenient%200%20%E2%80%93%20100%25%20scale.)

## Models
### Stochastic Gradient Descent
- Usually a Gradient Descent optimisation technique (seen in image below) would be very computationally expensive with a large dataset, but Stochastic Gradient Descent selects a random data sample to calculate the derivitives instead of the whole dataset.
- ![image](https://user-images.githubusercontent.com/108297203/200419171-2dd31e1a-1b87-44fa-a1e2-b716df9cff64.png)
- By selecting a random point in the data to then travel down the slope to find the minimum for the label.
- The learning rate is important as if it is too small it will take to long, but if its too large it may miss the minimum and settle for a false minimum.

### Gradient Boosting
- This ML model is sequential - it works to improve the previous models.
- It works by building trees with the features to predict the label.
![image](https://user-images.githubusercontent.com/108297203/200449229-b4b34bd2-5b2a-4f56-be7e-d0908572257a.png)
    -  Initially it takes the average of the predicted label and calculates Pseudo Residual by: (Observed label - Predicted Label)
    -  These Pseudo Residual are mapped and averaged to the leafs in the first tree where its feature values follow along.
    -  Pseudo Residual are then added to the training label
    -  To prevent overfitting/high variance a learning rate is used to scale the trees contribution (e.g. learning rate = 0.1)
        - This limits the prediction after making one tree, but several more are made and added to the calculation.
        - Multiple small steps equates to a better prediction with lower Variance
    - Trees are added based on the errors from previous trees, i.e. the second tree will have Pseudo Residual based on the first tree scaled by learning rate.
    - This continues until more trees do not significantly reduce the size of residuals or it reaches the maximum trees specified
###
### Regularization
- This parameter is present in SGDRegressor and Logstic Regression.
- It reduces the overfitting/generalsation error (difference between training and validation sets) by discouraging a learning a more complex/flexible model.
- More can be found at: 
    - https://www.pluralsight.com/guides/linear-lasso-ridge-regression-scikit-learn
    - https://towardsdatascience.com/regularization-in-machine-learning-76441ddcf99a


## Neural Networks
- These are computing systems which are comprised of many layers of interconnected units. 
![image](https://user-images.githubusercontent.com/108297203/200390046-30515704-46c2-41e2-a751-d84341f99ae1.png)
- Neural networks can have many layers, the increase in layers allows the network to perfrom more complex calculations.
- Each neuron in the network 
- The
- To discover the best configuration for the neural network a list of possible combinations in generate_nn_configs(). 
```python
config_dict = {'optimiser': ['torch.optim.SGD'],
                'learning_rate': [0.01, 0.001, 0.0001, 0.00001],
                'hidden_layer_width': [16, 32],
                'depth': [1, 2],
                'loss_func': ['mse_loss']}
    grid = list(ParameterGrid(config_dict))
    configs = []
    for params in grid:
        configs.append(params)
```
- These were then iterated through, each one made a ordered dict to pass into the neural network class, which was then trained and its metrics were found.
```python 
configs_list = generate_nn_configs()
metrics_list = []
trained_model_list = []
rmse_metric = []
hyperparameters = []
for config in configs_list:
        hyperparameters.append(config)

        #Parse the details from dict key
        optimiser_name = config['optimiser']
        learning_rate = config['learning_rate']
        loss_func = config['loss_func']
        hidden_layer = config['hidden_layer_width']
        linear_depth = config['depth']

        #create ordered dict
        config_dict = OrderedDict()
        # first layer input is 11 to match the number of features
        config_dict['input'] = nn.Linear(11, hidden_layer)
        for idx in range(linear_depth):
            rel_idx = f'relu{idx}'
            config_dict[rel_idx] = nn.ReLU()
            idx += 1
            od_idx = f'layer{idx}'
            config_dict[od_idx] = nn.Linear(hidden_layer, hidden_layer)
            idx +=1
        config_dict[f'layer{linear_depth}'] = nn.Linear(hidden_layer, 10)
        linear_depth +=1 
        config_dict[f'relu{linear_depth}'] = nn.ReLU()
        config_dict['output'] = nn.Linear(10, 1)
        #neurons for class last layer == to input classes
        #softmax for ouptput
        model = LinearRegression(config_dict)
        
        try:
            trained_model, metrics = train(model, train_loader, val_loader, test_loader, 200, optimiser_name, learning_rate, loss_func)
            trained_model_list.append(trained_model)
            metrics_list.append(metrics)
            rmse_metric.append(metrics['RMSE_loss_test'])
        except:
            print('Error in neural network.')
```
The training of each configuration of neural networks can be observed here:
![image](https://user-images.githubusercontent.com/108297203/203389157-246f61a5-f7b3-4921-bf0b-d50ea070905f.png)

![image](https://user-images.githubusercontent.com/108297203/203389050-c9d51ece-bef5-4afa-a84f-7b887024732c.png)
- These images show the drastic difference in loss between the diverse configs. The bottom image is a close up view of the best performing as opposed to the two runs which had a terrible performance in comparison.
- The R-squared and rmse was also calculated for all 3 train, validation and test sets. The figures for train and validation were compared to observe and decrease any overfitting: {"RMSE_loss_train": 0.8288193345069885, "RMSE_loss_val": 0.8430336117744446, "RMSE_loss_test": 0.6353606581687927, "R_squared_train": 0.2855689986481355, "R_squared_val": 0.26073228757584077, "R_squared_test": 0.3644241055577916, "training_duration": 53.118608713150024, "inference_latency": 0.14666998505592346}

- The metrics show that the train and validation sets RMSE and R-squared values are very similar which indicates the model has not been overfit.
- The test sets root mean squared error was used determine the optimal configuration. This can be seen as the orange line in the second graph above.
- In this run the best hyperparameters found were: {"depth": 2, "hidden_layer_width": 16, "learning_rate": 0.001, "loss_func": "mse_loss", "optimiser": "torch.optim.SGD"}
- A representation of this network would appear as:
![image](https://user-images.githubusercontent.com/108297203/203393130-ae8414e7-488d-4730-bb5c-4c6926dcb6ae.png)

- a





