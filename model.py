from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
x = pd.DataFrame(iris.data,columns=["s1","s2","p1","p2"])
y = iris.target

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

sv = SVC()
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3)
model22 = sv.fit(x,y)

# import pickle
# pickle.dump(model,open("model.pkl","wb"))
# print(x_test.head())
# data22 = pd.DataFrame({
# 	"s1" : [5.2,5.1],
#     "s2" : [3.2,3.1],
#     "p1" :[1.5,1.4],
#     "p2" :[0.4,0.3]
	
# 	})

# output = model22.predict(data22)
# print(str(output))
# print(x_test.info())
