import numpy as np
import random
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
