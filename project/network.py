from pybrain.structure import RecurrentNetwork
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.structure import FullConnection

#get no of words from corpus and store in numInputNodes and numOutputNodes
numInputNodes = 10
numHiddenNodes = 5
numOutputNodes = 10

#Creating a recurrent network with 1 input node, 1 output node and 10 hidden nodes
network = RecurrentNetwork()
network.addInputModule(LinearLayer(numInputNodes, name='in'))
network.addModule(SigmoidLayer(numHiddenNodes, name='hidden'))
network.addOutputModule(LinearLayer(numOutputNodes, name='out'))
in_to_hidden = FullConnection(network['in'], network['hidden'], name='connection1')
hidden_to_out = FullConnection(network['hidden'], network['out'], name='connection2')
hidden_to_hidden = FullConnection(network['hidden'], network['hidden'], name='connection3')
network.addConnection(in_to_hidden)
network.addConnection(hidden_to_out)
network.addRecurrentConnection(hidden_to_hidden)
network.sortModules()
print network.activate([1,0,0,0,0,0,0,0,0,0])
'''
print in_to_hidden.params
print "\n\n"
print hidden_to_out.params
print "\n\n"
print hidden_to_hidden.params
print "\n\n"
'''

#Input value initially: 1 of n coding of word + previous state s(t-1)
