# Bounnoy Phanthavong (ID: 973081923)
# Homework 5
#
# This is a machine learning program that uses a linear support
# vector machine to classify spam.
#
# This program was built in Python 3.

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_curve
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':

    # Load data.
    fName = "spambase/spambase.data"
    fileTrain = Path(fName)

    if not fileTrain.exists():
        sys.exit(fName + " not found")

    trainData = np.genfromtxt(fName, delimiter=",")

    # Split input data into X and labels into Y.
    X = trainData[:,0:-1]
    Y = trainData[:,-1]

    # Split X and Y by 50% for training and testing.
    # Keep proportion of labels same across both sets.
    XTrain, XTest, YTrain, YTest = train_test_split(X, Y, test_size = 0.5, stratify = Y)

    # Scale data using standardization.
    sc = StandardScaler()
    XTrain = sc.fit_transform(XTrain)
    XTest = sc.transform(XTest)

    # Run SVM learner on training data and then test learned model on test data.
    classify = SVC(kernel = "linear")
    classify.fit(XTrain, YTrain)
    YPredict = classify.predict(XTest)

    # Report accuracy, precision, and recall.
    report = classification_report(YTest, YPredict, output_dict = True)

    print("Accuracy:", report['accuracy'])
    print("Spam (precision):", report['1.0']['precision'])
    print("Spam (recall):", report['1.0']['recall'])
    print("Non-spam (precision):", report['0.0']['precision'])
    print("Non-spam (recall):", report['0.0']['recall'])

    # Create ROC curve for the SVM on test data.
    YScore = classify.decision_function(XTest)
    fpr, tpr, thresholds = roc_curve(YTest, YScore)

    plt.figure()
    plt.plot(fpr, tpr, label = "ROC Curve")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for linear SVM')
    plt.legend(loc = "lower right")
    plt.savefig("roc_curve")
