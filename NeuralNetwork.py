import numpy as np
from activation import relu, reluDerivative, sigmoid, sigmoidDerivative

class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        """Initialize the neural network with random weights and biases."""

        self.W1 = 2 * np.random.randn(hidden_size, input_size) - 1      # Weights for hidden layer
        self.b1 = np.zeros((hidden_size, 1))                            # Biases for hidden layer
        self.W2 = 2 * np.random.randn(output_size, hidden_size) - 1     # Weights for output layer
        self.b2 = np.zeros((output_size, 1))                            # Biases for output layer

        self.y_pred = None                                              # predicted results of y_test

    def feed_forward(self, X):
        """Propagate the input data forward through the network."""

        Z1 = np.dot(self.W1, X) + self.b1            # Weighted sum with biases (hidden layer)
        A1 = relu(Z1)                                # Apply ReLU activation (hidden layer)
        Z2 = np.dot(self.W2, A1) + self.b2           # Weighted sum with biases (output layer)
        A2 = sigmoid(Z2)                             # Apply sigmoid activation (output layer)

        return Z1, Z2, A1, A2
    
    def categorical_crossentropy(self, A2, Y):
        """CrossEntropy loss function"""
        y_true = np.array(Y)
        y_pred = np.array(A2)

        # Clip y_pred to avoid log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Compute cross-entropy loss
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

        return loss
        
        
    def back_propogation(self, Z1, A1, A2, X, Y, learning_rate):
        delta2 =  (A2 - Y) * sigmoidDerivative(A2)
        dW2 = delta2.dot(A1.T)
        db2 = np.mean(delta2, axis=1, keepdims=True)

        delta1 = self.W2.T.dot(delta2) * reluDerivative(Z1)
        dW1 = delta1.dot(X.T)
        db1 = np.mean(delta1, axis=1, keepdims=True)

        # update parameters
        self.W1 = self.W1 - learning_rate * dW1
        self.b1 = self.b1 - learning_rate * db1    
        self.W2 = self.W2 - learning_rate * dW2  
        self.b2 = self.b2 - learning_rate * db2  
        

    def train_epoch(self, X_train, y_train, learning_rate, batch_size):
        """Train the network for one epoch."""
        permutation = np.random.permutation(X_train.shape[1])
        X_train_shuffled = X_train[:, permutation]
        y_train_shuffled = y_train[permutation]
        
        for i in range(0, X_train.shape[1], batch_size):
            X_batch = X_train_shuffled[:, i:i+batch_size]
            y_batch = y_train_shuffled[i:i+batch_size]
            
            Z1, Z2, A1, A2 = self.feed_forward(X_batch)
            self.back_propogation(Z1, A1, A2, X_batch, y_batch, learning_rate)


    def train(self, X_train, y_train, learning_rate, epochs, batch_size):
        """Train the neural network with gradient descent."""
        prev_loss = float('inf')
        avg_loss = 0
        
        for epoch in range(epochs):
            self.train_epoch(X_train, y_train, learning_rate, batch_size)

            # Print loss after each epoch
            loss = self.categorical_crossentropy(self.feed_forward(X_train)[3], y_train)
            avg_loss += loss
            
            if (epoch + 1) % 10 == 0:
                avg_loss /= 10  # Calculate average loss over the last 10 epochs
                print(f"\nEpoch: {epoch+1}\n  Loss: {loss}")

                if avg_loss > prev_loss:
                    print(f"\nStopping training as loss increased on average of 10 consecutive epochs.")
                    break
                
                prev_loss = avg_loss
                avg_loss = 0  # Reset average loss for the next 10 epochs



    def roundResults(self, y_pred):
        """Round float type values of prediction"""
        if y_pred >= 0.5:
            return 1
        
        return 0
    
    def predict(self, X_test):
        """Evaluate the neural network on test data."""

        # Forward propagation
        Z1, Z2, A1, A2 = self.feed_forward(X_test)

        # Convert predicted probabilities to class labels
        self.y_pred = np.vectorize(self.roundResults)(A2)[0]

        return self.y_pred
