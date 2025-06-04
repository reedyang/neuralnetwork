import glob
import numpy
import neuralnetwork
import datetime
from PIL import Image

# number of input, hidden and output nodes
input_nodes = 784
hidden_nodes = 200
output_nodes = 10

# learning rate
learning_rate = 0.01

# create instance of neural network
n = neuralnetwork.NeuralNetwork(input_nodes,hidden_nodes,output_nodes, learning_rate)


# test the neural network
# scorecard for how well the network performs, initially empty
scorecard = []

# load the png image data as test data set
for image_file_name in glob.glob('my_own_images/2828_my_own*_?.png'):
    print("loading ...", image_file_name)
    # using the filename to set the correct label
    correct_label = int(image_file_name[-5:-4])

    image = Image.open(image_file_name).convert('L')
    width, height = image.size
    if width != 28 or height != 28:
        image = image.resize((28,28))
        pass
    
    # load image data from png files into an array
    img_array = numpy.asarray(image, dtype=float)

    # reshape from 28*28 to list of 784 values, invert values
    img_data = 255.0 - img_array.reshape(784)
    # then scale data to range from 0.01 to 1.0
    inputs = (img_data / 255.0 * 0.99) + 0.01
    
    # query the network
    outputs = n.query(inputs)
    # the index of the highest value corresponds to the label
    label = numpy.argmax(outputs)
    # append correct or incorrect to list
    if label == correct_label:
        # network's answer matches correct answer, add 1 to scorecard
        print("Correct label: ", label)
        scorecard.append(1)
    else:
        # network's answer doesn't match correct answer, add 0 to scorecard
        print("Incorrect label: ", label)
        scorecard.append(0)
        pass

    pass

# calculate the performance score, the fraction of correct answers
scorecard_array = numpy.asarray(scorecard)
performance = scorecard_array.sum() / scorecard_array.size
print(str(datetime.datetime.now().time()) + " performance = ", performance)
