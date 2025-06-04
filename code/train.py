import numpy
import scipy.special
import scipy.ndimage
import neuralnetwork
import datetime

# number of input, hidden and output nodes
input_nodes = 784
hidden_nodes = 200
output_nodes = 10

# learning rate
learning_rate = 0.01

# create instance of neural network
n = neuralnetwork.NeuralNetwork(input_nodes,hidden_nodes,output_nodes, learning_rate)

performance = 0.0

def test():
    # load the mnist test data CSV file into a list
    test_data_file = open("data/mnist_test.csv", 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    # test the neural network

    # scorecard for how well the network performs, initially empty
    scorecard = []

    # go through all the records in the test data set
    for record in test_data_list:
        # split the record by the ',' commas
        all_values = record.split(',')
        # correct answer is first value
        correct_label = int(all_values[0])
        # scale and shift the inputs
        inputs = (numpy.asarray(all_values[1:], dtype=numpy.float32) / 255.0 * 0.99) + 0.01
        # query the network
        outputs = n.query(inputs)
        # the index of the highest value corresponds to the label
        label = numpy.argmax(outputs)
        # append correct or incorrect to list
        if label == correct_label:
            # network's answer matches correct answer, add 1 to scorecard
            scorecard.append(1)
        else:
            # network's answer doesn't match correct answer, add 0 to scorecard
            scorecard.append(0)
            pass
        pass


    # calculate the performance score, the fraction of correct answers
    scorecard_array = numpy.asarray(scorecard)
    new_performance = scorecard_array.sum() / scorecard_array.size
    print(str(datetime.datetime.now().time()) + " performance = ", new_performance)
    
    global performance
    
    if performance >= new_performance:
        print(str(datetime.datetime.now().time()) + " Performance reduced")
        return False
    else:
        n.save()

    performance = new_performance

    return True

# do initial test to set last performance
test()

# load the mnist training data CSV file into a list
training_data_file = open("data/mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# train the neural network

# epochs is the number of times the training data set is used for training
epoch = 0
continue_train = True
while continue_train:
    epoch += 1
    print(str(datetime.datetime.now().time()) + " Start training. epoch = ", epoch)
    # go through all records in the training data set
    for record in training_data_list:
        # split the record by the ',' commas
        all_values = record.split(',')
        # scale and shift the inputs
        inputs = (numpy.asarray(all_values[1:], dtype=numpy.float32) / 255.0 * 0.99) + 0.01
        # create the target output values (all 0.01, except the desired label which is 0.99)
        targets = numpy.zeros(output_nodes) + 0.01
        # all_values[0] is the target label for this record
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)

        # create rotated variations
        # rotated anticlockwise by x degrees
        inputs_plusx_img = scipy.ndimage.rotate(inputs.reshape(28, 28), 10, cval=0.01, order=1, reshape=False)
        n.train(inputs_plusx_img.reshape(784), targets)
        # rotated clockwise by x degrees
        inputs_minusx_img = scipy.ndimage.rotate(inputs.reshape(28, 28), -10, cval=0.01, order=1, reshape=False)
        n.train(inputs_minusx_img.reshape(784), targets)
        pass

    continue_train = test()
    if not continue_train:
        print(str(datetime.datetime.now().time()) + " Stop training")
        pass
    
    pass
