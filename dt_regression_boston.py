import numpy as np
import data_service
import plotting_service
from sklearn.tree import DecisionTreeRegressor


def fit_predict_score(max_depth, X_train, X_test, Y_train, Y_test):

    dt_regressor = DecisionTreeRegressor(max_depth=max_depth)
    dt_regressor_model = dt_regressor.fit(X_train, Y_train)
    test_predictions = dt_regressor_model.predict(X_test)

    print("Predictions: ")
    print(test_predictions)

    print("Actuals: ")
    print(Y_test)

    print("Mean Absolute Error: ")
    # mse = np.sqrt(np.mean(np.square(test_predictions - Y_test)))
    mae = np.mean(np.abs(test_predictions - Y_test))
    print(mae)

    print("RMSE: ")
    rmse = np.sqrt(np.mean(np.square(test_predictions - Y_test)))
    print(rmse)

    return mae


np.set_printoptions(suppress=True, precision=2)

X_train, X_test, Y_train, Y_test = \
    data_service.load_and_split_data(scale_data=True, transform_data=False, test_size=0.2, random_slice=None,
                                     random_seed=None, dataset='boston')

mean_absolute_errors = []
max_depths = range(1, 10)

for max_depth in max_depths:

    #do it 5 times for the depth
    mean_abs_errors_for_depth = []
    for train_run in range(5):
        X_train, X_test, Y_train, Y_test = \
            data_service.load_and_split_data(scale_data=True, transform_data=False, test_size=0.2, random_slice=None,
                                             random_seed=None, dataset='boston')
        mean_abs_error = fit_predict_score(max_depth, X_train, X_test, Y_train, Y_test)
        mean_abs_errors_for_depth.append(mean_abs_error)
    mean_absolute_errors.append(np.mean(mean_abs_errors_for_depth))

plotting_service.plot_simple_line_chart(max_depths, mean_absolute_errors, "max_depth", "Mean Abs Error", "Regress DTree Mean Abs Error for max_depth")



