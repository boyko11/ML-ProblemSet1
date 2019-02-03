import numpy as np
from sklearn.linear_model import LinearRegression
import plotting_service

X = np.asarray([1, 3, 6, 10, 11, 13]).reshape(6,1)
y = np.asarray([1, 0, 5, 2, 1, 4])
linreg_model = LinearRegression().fit(X, y)
print(linreg_model.coef_)
print(linreg_model.intercept_)

plotting_service.plot_scatter_plot_and_best_fit_line(X, y, 'data_x', 'data_y', 'Lin Reg', linreg_model.coef_, linreg_model.intercept_)

