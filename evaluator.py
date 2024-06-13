import numpy as np
import matplotlib.pyplot as plt

class ConfusionMatrix:
    def __init__(self):
        # Initialize counts for true positives, true negatives, false positives, and false negatives
        self.TruePositive = 0
        self.TrueNegative = 0
        self.FalsePositive = 0
        self.FalseNegative = 0

    def fillMatrix(self, y_test, y_pred):
        # Fill the confusion matrix based on the actual and predicted values
        for pred, test in zip(y_pred, y_test):
            if pred == 1:
                if pred == test:
                    self.TruePositive += 1
                else:
                    self.FalsePositive += 1
            else:
                if pred == test:
                    self.TrueNegative += 1
                else:
                    self.FalseNegative += 1

    def precision(self):
        # Calculate precision (TP / (TP + FP))
        precision_value = self.TruePositive / (self.TruePositive + self.FalsePositive)
        return round(precision_value, 3)
        
    def accuracy(self):
        # Calculate accuracy ((TP + TN) / (TP + TN + FP + FN))
        total = self.TruePositive + self.TrueNegative + self.FalsePositive + self.FalseNegative
        accuracy_value = (self.TruePositive + self.TrueNegative) / total
        return round(accuracy_value, 3)
    
    def sensivity(self):
        # Calculate sensitivity (recall) (TP / (TP + FN))
        sensivity_value = self.TruePositive / (self.TruePositive + self.FalseNegative)
        return round(sensivity_value, 3)
    
    def specifity(self):
        # Calculate specificity (TN / (TN + FP))
        specifity_value = self.TrueNegative / (self.TrueNegative + self.FalsePositive)
        return round(specifity_value)

    def getEvaluation(self, y_test, y_pred):
        # Fill the confusion matrix and calculate evaluation metrics
        self.fillMatrix(y_test, y_pred)
        precision_value = self.precision()
        accuracy_value = self.accuracy()
        sensivity_value = self.sensivity()
        specifity_value = self.specifity()
        return precision_value, accuracy_value, sensivity_value, specifity_value
    
    def getGraph(self):
        # Plot the confusion matrix as a heatmap
        cm = np.array([[self.TruePositive, self.FalsePositive],
                       [self.FalseNegative, self.TrueNegative]])
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks([0, 1], ['Positive', 'Negative'])
        plt.yticks([0, 1], ['Positive', 'Negative'])
        plt.tight_layout()

        # Add text annotations for each cell in the matrix
        width, height = cm.shape
        for x in range(width):
            for y in range(height):
                plt.annotate(str(cm[x][y]), xy=(y, x),
                             horizontalalignment='center',
                             verticalalignment='center')
        
        plt.show()