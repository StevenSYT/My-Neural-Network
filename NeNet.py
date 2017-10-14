import numpy as np
import pandas as pd
import random
import preProcessing as pp
import matplotlib.pyplot as plt
import sys

def getDatasetUrl(x):
   return {
        'ds1': "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data",
        'ds2': "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
        'ds3': "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
   }[x]

def sigmoid(x):
    return (1+np.exp(-1*x))**(-1)
class NeNet:
    def __init__(self,dimList, rate):
        self.weightList = []
        self.layers = []
        self.biases = []
        self.rate = rate
        for val in range(len(dimList)-1):
            randome = lambda x: random.randrange(-100,100)/200
            randomes = lambda x: list(map(lambda x: randome(x), range(x)))
            rand = randomes(dimList[val+1]*dimList[val])
            rand2 = randomes(dimList[val+1])

            newWeight = np.array(rand).reshape(dimList[val+1], dimList[val])
            newBias = np.array(rand2).reshape(dimList[val+1],1)

            self.weightList.append(newWeight)
            self.biases.append(newBias)

    def backward(self,t,o):
        ones = np.array(list(map(lambda x: 1, range(t.shape[1])))).reshape(t.shape[1],1)
        tnorm = (1/t.shape[1])
        accumulator = -(t-o)
        newWeights = []
        newBiases = []
        self.reverseLists()
        for (weight, bias, layer) in zip(self.weightList, self.biases, self.layers):
            naccumulator = np.transpose(np.dot(np.transpose(accumulator),weight)) * (1-layer) * layer
            weight = self.update(weight, tnorm*np.dot(accumulator,np.transpose(layer)))
            newWeights.append(weight)
            bias = self.update(bias, tnorm*np.dot(accumulator,ones))
            newBiases.append(bias)
            accumulator = naccumulator
        self.weightList = newWeights
        self.biases = newBiases
        self.layers = []
        self.reverseLists()

    def reverseLists(self):
        self.weightList.reverse()
        self.biases.reverse()
        self.layers.reverse()

    def update(self, weight, grad):
        return weight - self.rate*grad

    def forward(self,inpData):
        ones = np.array(list(map(lambda x: 1, range(inpData.shape[1])))).reshape(inpData.shape[1],1)
        nLayer = inpData
        newLayers = []
        for (weight, bias) in zip(self.weightList, self.biases):
            layer = nLayer
            newLayers.append(layer)
            nLayer = sigmoid(np.dot(weight,layer) + np.dot(bias,np.transpose(ones)))
        self.layers = newLayers
        return nLayer

    def train(self, t ,inpData):
        o=self.forward(inpData)
        self.backward(t,o)
        return o
    def accuracy(self, outPut, target):
      maxInOutput=np.argmax(outPut, axis=0)
      maxInTarget=np.argmax(target,axis=0)
      count=0
      for i in range(len(maxInOutput)):
         if maxInOutput[i] == maxInTarget[i]:
            count+=1
      return count/len(maxInOutput)
   #  def printNN(self):
   #    for layer in self.



url=getDatasetUrl(sys.argv[1])
rawData = pd.read_csv(url, header=None)
percentage=float(sys.argv[2])/100
maxIteration=int(sys.argv[3])
print("maxIteration is ", maxIteration)
layers=list(map(lambda layer: int(layer), sys.argv[5:]))
print("laysers are ", layers)
x_train, x_test, y_train, y_test=pp.preProcess(rawData, percentage)
dimList=[]
dimList=np.append(dimList, x_train.shape[0])
dimList=np.append(dimList, layers)
dimList=np.append(dimList, y_train.shape[0])
dimList=dimList.astype(int)
neuralN = NeNet(dimList, 0.5)
accuArray=[]
accuTemp=0
j=0
print("Start training ...")
while j<maxIteration and accuTemp<0.99:
   trainOutPut=neuralN.train(y_train, x_train)
   j+=1
   accuTemp=neuralN.accuracy(trainOutPut, y_train)
   accuArray=np.append(accuArray,accuTemp)
   print("Training process: "+str(j)+"/"+str(maxIteration), end='\r')
testOutPut=neuralN.forward(x_test)
print("training accuracy is ", neuralN.accuracy(trainOutPut, y_train))
print("testing accuracy is ", neuralN.accuracy(testOutPut, y_test))
plt.plot(np.arange(j),accuArray)
plt.show()
# print("outPut is ",outPut)
# print("outPut Size is", outPut.shape)

# print("accuracy is: ", accuracy)
