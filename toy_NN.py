import numpy as np
import math

nCategories=4
inDim=5
datasets=np.random.rand(nCategories,inDim)
i=np.linspace(0,3,4,dtype='int')
y1 =np.zeros((4,4))
y1[i,i] = 1

nNodes1=7
nNodes2=nCategories
class toy_NN:
    def __init__(self,inDim,nNodes1,nNodes2):
        self.w1=np.random.rand(inDim,nNodes1)-.5
        self.b1=np.random.rand(nNodes1)-.5
        self.w2=np.random.rand(nNodes1,nNodes2)-.5
        self.b2=np.random.rand(nNodes2)-.5

    def softMax(self,output):
        exponent = np.exp(output)
        return exponent/np.sum(exponent,1,keepdims=True)

    def softMaxDer(self, values, y):
        exponent = np.exp(values)
        correctExp=np.sum(np.multiply(exponent,y),1,keepdims=True)
        invertedY=(1-y)
        offDiagonal = invertedY*correctExp
        onDiagonal=np.sum(invertedY*exponent,1,keepdims=True)*y
        sumMatrix=offDiagonal+onDiagonal
        summedExponent=np.sum(exponent,1,keepdims=True)
        derivative = (exponent+sumMatrix)/np.power(summedExponent,2)*(-invertedY+y)
        return derivative

    def loss(self,prob,y):
        correctProb = np.sum(np.multiply(prob,y),1,keepdims=True)
        return -np.log(correctProb)

    def lossDerivative(self,prob,y):
        return -np.power(np.sum(np.multiply(prob,y),1,keepdims=True),-1)

    def forwardPass(self,dataset):
        self.firstLayer = datasets.dot(self.w1)
        self.firstLayerBias = self.firstLayer+self.b1
        self.firstLayerReLU = np.maximum(self.firstLayerBias,0)
        self.meanFirstLayer =0# np.mean(self.firstLayerReLU)
        self.sigmaFirstLayer =1# np.std(self.firstLayerReLU)
        self.normFirstLayer = (self.firstLayerReLU-self.meanFirstLayer)/self.sigmaFirstLayer
        self.secondLayer = self.normFirstLayer.dot(self.w2)
        self.secondLayerBias = self.secondLayer+self.b2
        self.meanSecondLayer =0# np.mean(self.secondLayerBias)
        self.sigmaSecondLayer =1# np.std(self.secondLayerBias)
        self.normSecondLayer = (self.secondLayerBias-self.meanSecondLayer)/self.sigmaSecondLayer
        #print "sigma\n",self.normSecondLayer
        softMaxOutput = self.softMax(self.normSecondLayer)
        return softMaxOutput

    def train(self, datasets, y,nCategories,learningRate,reg):
        softMaxOutput = self.forwardPass(datasets)
        self.softMaxOutput2 = softMaxOutput
        loss=self.loss(softMaxOutput,y)
        lossDerivative=self.lossDerivative(softMaxOutput,y)
        softMaxDer=-1*self.softMaxDer(self.secondLayerBias,y)/nCategories#*lossDerivative/nCategories
        softMaxDer=-y+softMaxOutput
        print softMaxDer
        softMaxDer=softMaxDer/self.sigmaSecondLayer
        self.softMaxDer2=softMaxDer
        #print "sigmaSecondLayer\n",self.sigmaSecondLayer
        #print "softMaxOutput\n",softMaxOutput
        #print "softMaxDer\n", softMaxDer
        #print "self.w2\n", self.w2
        #print "bias\n",self.secondLayerBias
        b2Grad=np.sum(softMaxDer,0)
        w2Grad=np.dot(self.firstLayerReLU.T,softMaxDer)
        normGrad = softMaxDer/self.sigmaFirstLayer
        firstLayerReLUGrad=np.dot(normGrad,self.w2.T)
        downStreamReLUGrad=firstLayerReLUGrad*np.where(self.firstLayerReLU>0,1,0)
        b1Grad=np.sum(downStreamReLUGrad,0)
        w1Grad=np.dot(datasets.T,downStreamReLUGrad)
        self.w1+=-1*learningRate*w1Grad #- reg*self.w1
        self.w2+=-1*learningRate*w2Grad #- reg*self.w2
        self.b1+=-1*learningRate*b1Grad #- reg*self.b1
        self.b2+=-1*learningRate*b2Grad #- reg*self.b2

        reg_loss = 0.5*reg*(np.sum(self.w1)+np.sum(self.b1)+np.sum(self.w2)+np.sum(self.b2))
        return np.mean(loss)+reg_loss

lastloss=1000
rand_net = toy_NN(inDim,nNodes1,nNodes2)
for i in range(10000):
    reg = 1e-40 # regularization strength
    loss=rand_net.train(datasets,y1,nCategories,1e-3,reg)
    if math.isnan(loss): break
    if loss > lastloss+.1: 
        print "sigmaSecondLayer\n",rand_net.sigmaSecondLayer
        print "softMaxDer\n",rand_net.softMaxDer2
        print "softMaxOutput2\n",rand_net.softMaxOutput2
        print "softMaxOutput2\n",np.sum(rand_net.softMaxOutput2,1)
        print "self.normSecondLayer\n", rand_net.normSecondLayer
        print "self.w2\n", rand_net.w2
        print "bias\n",rand_net.secondLayerBias
        break
    lastloss=loss
    if i%20==0: print i,loss
    if i%200==0:
        print "sigmaSecondLayer\n",rand_net.sigmaSecondLayer
        print "softMaxDer\n",rand_net.softMaxDer2
        print "softMaxOutput2\n",rand_net.softMaxOutput2
        print "softMaxOutput2\n",np.sum(rand_net.softMaxOutput2,1)
        print "self.normSecondLayer\n", rand_net.normSecondLayer
        print "self.w2\n", rand_net.w2
        print "bias\n",rand_net.secondLayerBias
print rand_net.forwardPass(datasets)