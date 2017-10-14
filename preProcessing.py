import pandas as pd
from pandas import DataFrame as df
import numpy as np
import numbers
import sys

def getDatasetPath(x):
   if x=="ds1":
      return "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
   elif x=="ds2":
      return "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
   elif x=="ds3":
      return "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
   else:
      return x
# url= "adult.csv"
def preProcess(inputPath,outputPath):
   rawData = pd.read_csv(inputPath, header=None)
   numOfAttributes= rawData.shape[1]

   #check if any instance has missing data
   for column in rawData:
      for index,row in rawData.iterrows():

         if rawData.loc[index][column]=="?" :
            rawData.drop(rawData.index[index], inplace = True)

   print("rawData.shape: ", rawData.shape)
   processedDf = pd.DataFrame()


   for column in rawData.iloc[:,:-1]:
      #standardize the data
      if isinstance(rawData.iloc[0,column], numbers.Number) :
         rawRow=rawData.iloc[:,column]
         rawRow=(rawRow-rawRow.mean())/rawRow.std()
         # rawData.iloc[:,column]=rawRow
         processedDf=pd.concat([processedDf, rawRow], axis=1)
         pass
      #convert the categorical data into numeric ones
      else:
         processedDf=pd.concat([processedDf, pd.get_dummies(rawData[column])], axis=1)
   classdf=df(rawData.iloc[:,-1])
   classdf.columns=['class']
   processedDf=pd.concat([processedDf, classdf], axis=1)
   print("dataFrame columns:", processedDf.columns)

   print("dummy shape: ", processedDf.shape[1])
   processedDf.to_csv(outputPath)


   #split data
   # return split(processedDf, outSize, percent)

   # print(trainData[1])
inputPath=getDatasetPath(sys.argv[1])
preProcess(inputPath, sys.argv[2])
