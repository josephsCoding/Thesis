# Hyperparameter tuning using GridSearchCV

dt = tree.DecisionTreeClassifier()
num_trials = 1
nested_scores = []
start = timeit.default_timer()

param = {'max_depth': range(3, 10), 'criterion': ['gini', 'entropy'],
        'min_samples_split': np.linspace(0.1, 1, 10), 'min_samples_leaf': np.linspace(0.01,0.5,5) }

for i in range(0,num_trials):

    cv_inner = KFold(n_splits=10, shuffle=True, random_state=i)
    cv_outer = KFold(n_splits =10, shuffle=True, random_state=i)
    
    # Inner Loop
    clf = GridSearchCV(estimator = dt, param_grid=param, cv=cv_inner, scoring='accuracy')
    
    # Outer Loop
    scores = cross_val_score(clf, X=x_train, y=y_train, cv=cv_outer)
    nested_score = scores.mean()
    nested_score_std = scores.std()
    
    nested_scores.append(nested_score)
    
stop = timeit.default_timer()
print('Time: ', stop - start) 
print nested_scores  # This score is a unbiased estimate of the performance of the model
print nested_score_std  # This should be as low as possible 

# If you are happy with the unbiased performance, retrain the model using all of the training data on the best parameters. 
# clf will already be refit with the best parameters 
clf.fit(x_train, y_train)
print "Training Set Accuracy: " + str(clf.score(x_train, y_train))
prediction = clf.predict(x_test)
print "Test Set Accuracy: " + str(accuracy_score(y_test, prediction, normalize=True))

# If you are happy with the test and training accuracies, then you are finished. The test accuracy should not be significantly different
# from the training and nested accuracies. 
