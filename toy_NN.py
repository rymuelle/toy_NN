import numpy as np
import math

#create a fake dataset to train on
nCategories=4
inDim=5
dataset=np.random.rand(nCategories,inDim)
i=np.linspace(0,3,4,dtype='int')
y1=np.zeros((4,4))
y1[i,i]=1

#define some hyper paramters, nHiddenLayer defines the hidden layer, outDim defines the output layer
#outDim is set to equal the number of inputs as we have the same nuber of examples as classifications
nHiddenLayer=7
outDim=nCategories

class toy_NN:
    def __init__(self,inDim,nHiddenLayer,outDim,batchNorm=False):
        #Initilize the weights. Such a shallow network doesn't require Xavier initialzation.
        self.w1=np.random.rand(inDim,nHiddenLayer)-.5
        self.b1=np.random.rand(nHiddenLayer)-.5
        self.w2=np.random.rand(nHiddenLayer,outDim)-.5
        self.b2=np.random.rand(outDim)-.5
        self.batchNorm=batchNorm
        #save initial values 
        self.w1Init = self.w1
        self.b1Init = self.b1
        self.w2Init = self.w2
        self.b2Init = self.b2

    def reset(self):
        self.w1=self.w1Init
        self.b1=self.b1Init
        self.w2=self.w2Init
        self.b2=self.b2Init

    def softMax(self,output):
        exponent=np.exp(output)
        return exponent/np.sum(exponent,1,keepdims=True)

    def softMaxDer(self,output,y):
        softMaxValue=self.softMax(output)
        softMaxMult=y-softMaxValue
        derivative=softMaxMult*softMaxValue
        return derivative

    def loss(self,prob,y):
        #here we use the L=-ln(pi) cross entropy loss, rather than the information theory L=-pi*ln(pi)
        correctProb=np.sum(np.multiply(prob,y),1,keepdims=True)
        return -np.log(correctProb)

    def lossDerivative(self,prob,y):
        return -np.power(np.sum(np.multiply(prob,y),1,keepdims=True),-1)

    def forwardPassProp(self,dataset):
        #forward prop is defined as a function so we can evaluate outside of training
        #hidden layer: wx+b->ReLU
        self.firstLayer=dataset.dot(self.w1)
        self.firstLayerBias=self.firstLayer+self.b1
        self.firstLayerReLU=np.maximum(self.firstLayerBias,0)
        #batch norm: I wanted to introduce batch normalization for fun, but our batch is so small it is a bit unstable
        self.meanFirstLayer=np.mean(self.firstLayerReLU) if self.batchNorm else 0
        self.sigmaFirstLayer=np.std(self.firstLayerReLU) if self.batchNorm else 1
        self.normFirstLayer=(self.firstLayerReLU-self.meanFirstLayer)/self.sigmaFirstLayer
        #output wx+b
        self.secondLayer=self.normFirstLayer.dot(self.w2)
        self.secondLayerBias=self.secondLayer+self.b2
        #batch norm
        self.meanSecondLayer=np.mean(self.secondLayerReLU) if self.batchNorm else 0
        self.sigmaSecondLayer=np.std(self.secondLayerReLU) if self.batchNorm else 1
        self.normSecondLayer=(self.secondLayerBias-self.meanSecondLayer)/self.sigmaSecondLayer
        #softmax
        softMaxOutput=self.softMax(self.normSecondLayer)
        return softMaxOutput

    def train(self,dataset,y,nCategories,learningRate,reg,useFullDerivatives=False):
        softMaxOutput=self.forwardPassProp(dataset)
        loss=self.loss(softMaxOutput,y)
        #I implemented the full backwards prop from the loss, but the simpler "L1" difference works fine here
        if useFullDerivatives:
            lossDerivative=self.lossDerivative(softMaxOutput,y)
            softMaxDer=self.softMaxDer(self.secondLayerBias,y)*lossDerivative/nCategories
        else:
            softMaxDer=-y+softMaxOutput
        softMaxDer=softMaxDer/self.sigmaSecondLayer
        b2Grad=np.sum(softMaxDer,0)
        w2Grad=np.dot(self.firstLayerReLU.T,softMaxDer)
        normGrad=softMaxDer/self.sigmaFirstLayer
        firstLayerReLUGrad=np.dot(normGrad,self.w2.T)
        downStreamReLUGrad=firstLayerReLUGrad*np.where(self.firstLayerReLU>0,1,0)
        b1Grad=np.sum(downStreamReLUGrad,0)
        w1Grad=np.dot(dataset.T,downStreamReLUGrad)
        #update! simple grad descent, consider implementing something like Adam or other regularization schemes
        self.w1+=-1*learningRate*w1Grad - reg*self.w1
        self.w2+=-1*learningRate*w2Grad - reg*self.w2
        self.b1+=-1*learningRate*b1Grad - reg*self.b1
        self.b2+=-1*learningRate*b2Grad - reg*self.b2
        #return loss
        reg_loss=0.5*reg*(np.sum(self.w1)+np.sum(self.b1)+np.sum(self.w2)+np.sum(self.b2))
        return np.mean(loss)+reg_loss

rand_net=toy_NN(inDim,nHiddenLayer,outDim)
reg=1e-4 # regularization strength
lRate=1e-1 #learing rate
for i in range(10000):
    loss=rand_net.train(dataset,y1,nCategories,lRate,reg)
    if math.isnan(loss): break
    if i%20==0: print i,loss

print rand_net.forwardPassProp(dataset)