import pandas
from sklearn import linear_model
import matplotlib.pyplot as plt



df = pandas.read_csv("cars.csv")

X = df[['Weight', 'Volume']]
y = df['CO2']

regr = linear_model.LinearRegression()
regr.fit(X, y)


#predict the CO2 emission of a car where the weight is 2300g, and the volume is 1300ccm:
predictedCO2 = regr.predict([[2300, 1300]])

print(predictedCO2)




threedee = plt.figure().add_subplot(projection='3d')

threedee.scatter(  df['Weight'], df['Volume'], df['CO2'])
threedee.set_xlabel('Weight')
threedee.set_ylabel('Volume')
threedee.set_zlabel('CO2')
plt.show()