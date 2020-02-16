from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
x = pd.DataFrame(iris.data,columns=[iris.feature_names])
y = iris.target

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

sv = SVC()
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3)
model = sv.fit(x,y)

# import pickle
# pickle.dump(model,open("model.pkl","wb"))



