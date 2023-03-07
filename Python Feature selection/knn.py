import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

ds = pd.read_csv('insurance.csv')

label_encoder = LabelEncoder()
ds["sex"] = label_encoder.fit_transform(ds["sex"]) 
ds["smoker"] = label_encoder.fit_transform(ds["smoker"])
ds["region"] = label_encoder.fit_transform(ds["region"])


cols=np.array(["age","sex","bmi","children","smoker","region","charges"])
y = ds["charges"].values

def findScores(index,scores,features):
  x = ds[cols[index]].values

  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)

  scaler = StandardScaler()
  x_train = scaler.fit_transform(x_train)
  x_test = scaler.transform(x_test)

  y_pred = KNeighborsRegressor().fit(x_train,y_train).predict(x_test)
  
  r2 = r2_score(y_test,y_pred)
  mae = mean_absolute_error(y_test, y_pred) 
  mse = mean_squared_error(y_test,y_pred,squared = False)

  print(cols[index])

  if(scores[0]<r2):
    scores[0] = r2
    features[0] = cols[index]
  
  if(scores[1]>mae):
    scores[1] = mae
    features[1] = cols[index]

  if(scores[2]>mse):
    scores[2] = mse
    features[2] = cols[index]

  print(f"r2 score: {r2}\nmae score: {mae}\nmse score: {mse}\n")
  
  return (scores,features)


scores = [-100000,100000,100000]
features = [""]*3
for i in range(0,6):
  (scores,features) = findScores([i],scores,features)
  for j in range(i+1,6):
    (scores,features) = findScores([i,j],scores,features)
    for k in range(j+1,6):
      (scores,features) = findScores([i,j,k],scores,features)
      for a in range(k+1,6):
        (scores,features) = findScores([i,j,k,a],scores,features)
        for b in range(a+1,6):
          (scores,features) = findScores([i,j,k,a,b],scores,features)
          for c in range(b+1,6):
            (scores,features) = findScores([i,j,k,a,b,c],scores,features)


print(f"\n{features[0]}\nmax r2 score: {scores[0]}\n\n{features[1]}\nmin mae score: {scores[1]}\n\n{features[2]}\nmin mse score: {scores[2]}\n")