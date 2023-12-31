import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


from azureml.core import Workspace, Dataset
subscription_id = 'cbd01ab5-5792-49a4-a518-b094e967e9a6'
resource_group = 'aditya.kuchulakanti-rg'
workspace_name = 'ml-ops-workspace'
workspace = Workspace(subscription_id, resource_group, workspace_name)
dataset = Dataset.get_by_name(workspace, name='Iris-data')
df = dataset.to_pandas_dataframe()

features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
taget = 'Species'

X_train, X_test, y_train, y_test = train_test_split(df[features],df[taget], test_size=0.2 , shuffle=True)

#Step:1 initilise the model class
clf = DecisionTreeClassifier(criterion = "entropy")

#Step:2 train the model on training set
clf.fit(X_train,y_train)

#step:3 evaluate the data on testing set 
y_pred = clf.predict(X_test)

print(f"Accuracy of the model is {accuracy_score(y_test,y_pred)*100}")



