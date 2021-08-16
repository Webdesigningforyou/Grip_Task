from sklearn import linear_model,metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Load the csv file
data=pd.read_csv("student_data.csv")

print(data.columns)
print(data.head(10))


print(data.info())

print(data.describe())
plt.scatter(data.Hours,data.Scores,color="blue", marker="o")
plt.xlabel('Hours')
plt.ylabel("Scores")

plt.show()
#define train variables for the linear_model
x_train=np.array(data["Hours"]).reshape((-1,1))
y_train=np.array(data["Scores"])

reg=linear_model.LinearRegression()
reg.fit(x_train,y_train)

#plot the training and predicted data
plt.scatter(x_train,y_train,color="blue", marker="o") #training data
y_predict=reg.predict(x_train)
plt.plot(x_train,y_predict,color="black") #predicted values
plt.xlabel('Hours')
plt.ylabel("Scores")

plt.show()


# output for the 9.25 hours/day
hours=9.25
predicted_score=reg.predict([[hours]])


print("Score={0:.2f} when hours are {1}".format(predicted_score[0],hours))


#Model Evaluation metrics
mean_squ_error=mean_squared_error(y_train,y_predict)
mean_abs_error=mean_absolute_error(y_train,y_predict)
print("Mean Squared Error is {:.2f}".format(mean_squ_error))
print("Mean Absolute Error is {:.2f}".format(mean_abs_error))
print("END")
