from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet,UnsupervisedDataSet
from pybrain.structure import LinearLayer, SigmoidLayer, SoftmaxLayer, TanhLayer
from pprint import pprint

from CorpusReader import corpusReader

[numInputNodes, datasets, hashCodeDict]=corpusReader() 
#get no of words from corpus and store in numInputNodes and numOutputNodes
numHiddenNodes = 5
numOutputNodes = numInputNodes

ds = datasets
print datasets

#create a network: Inner laye : Linear, Hiddenlayer : Sigmoid, Outputlayer: Softmax, Type = recurrent
net = buildNetwork(numInputNodes, numHiddenNodes, numOutputNodes, hiddenclass=SigmoidLayer, outclass=SoftmaxLayer,bias=True, recurrent=True)

#train the network
#learningRate = 0.1 ~ according to paper
#lrDecay = rate of change of learningRate in between iterations of the same epoch
#batchlearning = if set to true the parameters are changed only after completion of entire epoch
trainer = BackpropTrainer(net, ds,  learningrate=0.1, lrdecay=1,batchlearning=True )
for i in range(20):
	trainer.trainEpochs( 1 )

#test it on unsupervised dataset validation
#use the network to predict next word for the validation dataset and calculate the log likelyhood

# Instead of using the above method of manually checking till convergence we can also use the below. 
#trainer.trainUntilConvergence(validationProportion=0.25, verbose = True);

inpt=[0]*numInputNodes
inpt[1]=1

ans = net.activate(inpt)
print "###################################"
print ans
import numpy as np
#print numpy.argmax(ans)
topSixSuggestions= np.argsort(ans)[::-1][:16]
print hashCodeDict.keys()
print "###################################"
print "Top 16 suggestions:"
for i in range(16):
	key =  hashCodeDict.keys()[topSixSuggestions[i]]
	print key
