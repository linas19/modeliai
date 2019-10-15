#Zemiau pateiktas linijines regresijos modelis
model_linear_regression = LinearRegression()
result_linear_regression = model_linear_regression.fit(X_train, y_train)

#Zemiau pateiktas logistines regresijos modelis
model_logistic_regression = LogisticRegression()
result_logistic_regression = model_logistic_regression.fit(X_train, y_train)

#Zemiau pateiktas sprendimu medzio modelis
model_classification_tree = tree.DecisionTreeClassifier(max_depth=6)
result_classification_tree = model_classification_tree.fit(X_train, y_train)

