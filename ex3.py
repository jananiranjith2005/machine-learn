import streamlit as st
import numpy as np

class NeuralNetwork():

    def __init__(self):
        # seeding for random number generation
        np.random.seed(1)

        #converting weights to a 3 by 1 matrix with values from -1 to 1 and mean of 0
        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1

    def sigmoid(self, x):
        #applying the sigmoid function
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        #computing derivative to the Sigmoid function
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, training_iterations):

        #training the model to make accurate predictions while adjusting weights continually
        for iteration in range(training_iterations):
            #siphon the training data via  the neuron
            output = self.think(training_inputs)

            #computing error rate for back-propagation
            error = training_outputs - output

            #performing weight adjustments
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))

            self.synaptic_weights += adjustments

    def think(self, inputs):
        #passing the inputs via the neuron to get output
        #converting values to floats

        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        return output

def main():
    st.title("Neural Network Streamlit App")

    st.write("This app demonstrates a simple neural network trained on the XOR dataset.")

    # Initializing the neural network
    neural_network = NeuralNetwork()

    # Training data consisting of 4 examples--3 input values and 1 output
    training_inputs = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_outputs = np.array([[0, 1, 1, 0]]).T

    # Training the neural network
    neural_network.train(training_inputs, training_outputs, 15000)

    # Allowing user input for new predictions
    user_input_one = st.text_input("User Input One", "0")
    user_input_two = st.text_input("User Input Two", "0")
    user_input_three = st.text_input("User Input Three", "0")

    if st.button("Predict"):
        output = neural_network.think(np.array([user_input_one, user_input_two, user_input_three]))
        st.write("Output:", output)

if __name__ == "__main__":
    main()
