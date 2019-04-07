from sklearn import tree
from sklearn import svm
from sklearn.linear_model import LogisticRegression

X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,Y)

prediction = clf.predict(X)
prediction = clf.score(X,Y)
print(clf.predict([[190, 70, 43]]))
print(prediction)

clf =  svm.SVC(gamma = 'scale')
clf.fit(X,Y)
prediction = clf.predict(X)
prediction = clf.score(X,Y)
print(clf.predict([[190, 70, 43]]))
print(prediction)

clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial')
clf.fit(X,Y)
prediction = clf.predict(X)
prediction = clf.score(X,Y)
print(clf.predict([[190, 70, 43]]))
print(prediction)
LogisticRegression
