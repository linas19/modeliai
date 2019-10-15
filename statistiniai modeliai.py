

model_linear_regression = LinearRegression()
result_linear_regression = model_linear_regression.fit(X_train, y_train)

model = LogisticRegression()
result = model.fit(X_train, y_train)
