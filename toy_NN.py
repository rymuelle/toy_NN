import numpy as np

nCategories=4
inDim=5
datasets=np.random.rand(nCategories,inDim)-.5
print
ys = [i for i in range(nCategories)]

nNodes1=7
nNodes2=nCategories
class toy_NN:
    def __init__(self,inDim,nNodes1,nNodes2,learningRate):
        self.w1=np.random.rand(inDim,nNodes1)-.5
        self.b1=np.random.rand(nNodes1)-.5
        
        self.w2=np.random.rand(nNodes1,nNodes2)-.5
        self.b2=np.random.rand(nNodes2)-.5
        self.learningRate = learningRate

    def softMax(self,output):
        exponent = np.exp(output)
        return exponent/np.sum(exponent,0)

    def softMaxDer(self, values, y):
        exponent = np.exp(values)
        derivative = -(exponent[y]+exponent)/np.sum(np.power(exponent,2))
        derivative[y]=(exponent[y]+(np.sum(exponent))-exponent[y])/np.sum(np.power(exponent,2))
        return derivative

    def loss(self,prob,y):
        return np.log(prob[y])*-1, -1/prob[y]

    def forwardPass(self,dataset):
        self.firstLayer = datasets.dot(self.w1)
        self.firstLayerBias = self.firstLayer+self.b1
        self.firstLayerReLU = np.maximum(self.firstLayerBias,0)
        self.secondLayer = self.firstLayerReLU.dot(self.w2)
        self.secondLayerBias = self.secondLayer+self.b2
        softMaxOutput = self.softMax(self.secondLayerBias)
        return softMaxOutput

    def train(self, datasets, y):
        self.forwardPass(datasets)
        #for count, dataset in enumerate(datasets):
        #    softMaxOutput=self.forwardPass(dataset)
        #    loss,der=self.loss(softMaxOutput,y[count])
        #    softMaxDer=self.softMaxDer(self.secondLayerBias, y[count])
        #    dl=der*softMaxDer
        #    b2Slope.append(dl)
        #    print self.firstLayerReLU.T.dot(b2Slope)
        #    w2Slope.append(np.asarray([x*dl for x in self.firstLayerReLU]))
        #    firstLayerBiasSlope = dl*self.w2
        #    #print firstLayerBiasSlope, dl

rand_net = toy_NN(inDim,nNodes1,nNodes2,1e-4)

rand_net.train(datasets,ys)