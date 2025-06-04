import numpy
import neuralnetwork
import matplotlib.pyplot
import neuralnetwork

# number of input, hidden and output nodes
input_nodes = 784
hidden_nodes = 200
output_nodes = 10

# learning rate
learning_rate = 0.01

# create instance of neural network
n = neuralnetwork.NeuralNetwork(input_nodes,hidden_nodes,output_nodes, learning_rate)

# run the network backwards, given a label, see what image it produces

# label to test
label = 0
# create the output signals for this label
targets = numpy.zeros(output_nodes) + 0.01
# all_values[0] is the target label for this record
targets[label] = 0.99
print(targets)

# get image data
image_data = n.backquery(targets)

# plot image data
matplotlib.pyplot.imshow(image_data.reshape(28, 28), cmap='Greys', interpolation='None')
matplotlib.pyplot.show()