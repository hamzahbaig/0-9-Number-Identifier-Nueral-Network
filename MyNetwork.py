import sys
import numpy as np
import time

finalAns = [] # this is used to write in file plot.txt I will fill this list 
# in this pattern [timeTaken, learningRate, accuracy]

def readLabels(labelsFile):
    labels = open(labelsFile, "r")
    data = []
    for entry in labels:
        data.append(int(entry))
    return data


def readMatrix(matrixFile,length):
    file = open(matrixFile,"r")
    data = []
    for _ in range(length):
        matrix = ""
        for _ in range(44):
            lines = file.readline()
            lines = lines.strip("[")

            lines = lines.replace("]", "")
            matrix = matrix + lines
        
        matrix = matrix.split()
        matrix = map(int,matrix)

        data.append(matrix)
    return data

def normalise(matrix):
    mean = np.mean(matrix)
    standardDev = np.std(matrix)

    matrix = (matrix - mean) / standardDev
    return matrix

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def train(matrixFile,labelsFile,learningRate):
    global finalAns
    startTime = time.time()

    print("\n\t----- STARTING TRAINING -----")

    inputMatrix = readMatrix(matrixFile,60000) # reads the training data file and returns a list of list of pixels

    trainLabels = readLabels(labelsFile) #read all the labels corresponding to the pixel

    hiddenWeight = 2* np.random.random((784,30)) -1 #generating 784*30 matrix of weights between input and hiddenlayer
    outputWeight = 2* np.random.random((30,10)) -1 #generating 30*10 matrix of weights betwwen hiddenLayer and output layer

    for _ in range(2):
        for count in range(len(inputMatrix)):

            inputLayer = np.matrix(inputMatrix[count])
            inputLayer = normalise(inputLayer)

            hiddenLayer = np.dot(inputLayer,hiddenWeight)
            hiddenLayer = sigmoid(hiddenLayer)

            outputLayer = np.dot(hiddenLayer,outputWeight)
            outputLayer = sigmoid(outputLayer)

            # --- Error Calculation ---
            targetMatrix = np.zeros((1,10))
            targetMatrix[0,trainLabels[count]] = 1

            deltaK = outputLayer - targetMatrix
            deltak = np.transpose(deltaK)


            # --- Weight Update ---
            deltaHMultiplier1 = np.multiply(hiddenLayer,1-hiddenLayer)
            deltaHMultiplier2 = np.dot(outputWeight,deltak)
            deltaHMultiplier2 = np.transpose(deltaHMultiplier2)
            deltaH = np.multiply(deltaHMultiplier1,deltaHMultiplier2)

            hiddenWeight -= learningRate * np.dot(inputLayer.T, deltaH)
            outputWeight -= learningRate * np.dot(hiddenLayer.T, deltak.T)

    endTime = time.time()

    print("\n\t--- TRAINING ENDED ---")
    print("\n\t\tTime Taken: " + str(endTime-startTime))
    print("\n\t--- WRITING FILE ---")

    print("\n\t\tWriting calculated weights on the file..")

    file = open("netWeights.txt","w")
    np.savetxt(file,hiddenWeight)
    np.savetxt(file,outputWeight)
    file.close()

    print("\n\t\tWriting Done")
    finalAns.append(endTime-startTime)
    finalAns.append(learningRate)

    test("test.txt","test-labels.txt","netWeights.txt")

   
    

def setWeights(string,row,col,count):
    randomWeight = []
    for _ in range(col):
        temp = []
        for _ in range(row):
            temp.append(string[count])
            count +=1
        
        randomWeight.append(temp)
    return np.matrix(randomWeight).reshape(row,col),count
    
def test(matrixFile,labelsFile,weights):
    global finalAns	
    print("\n\t----- STARTING TESTING -----")
    
    testInputs = readMatrix(matrixFile,10000) #read testing matrix
    testLables = readLabels(labelsFile) #read labels
    accuracy = 0

    file = open(weights, "r")

    string = ""
    for line in file:
        line = line.replace("/n", " ")
        string = string + line
    
    string = string.split()
    string = map(float,string)

    count = 0
    hiddenWeight,count = setWeights(string,784,30,count)
    outputWeight,count = setWeights(string,30,10,count)

    for count in range(len(testInputs)):
        inputLayer = np.matrix(testInputs[count])
        inputLayer = normalise(inputLayer)

        hiddenLayer = np.dot(inputLayer,hiddenWeight)
        hiddenLayer = sigmoid(hiddenLayer)

        outputLayer = np.dot(hiddenLayer,outputWeight)
        outputLayer = sigmoid(outputLayer)

        target = np.argmax(outputLayer)

        if(target == testLables[count]):
            accuracy += 1

    accuracy = float(accuracy)
    print("\n\tAfter Epoch number 2: " + str(accuracy) + "/10,000 images correctly classified")
    print("\n\tAccuracy: " + str(accuracy/100) + "%")
    print("\n\tError: " +str((10000-accuracy)/100) + "%")

    finalAns.append(accuracy/100)

    # Saving finalAns (time,learningRate,Accuracy) in plot.txt for graph plotting
    file = open("plot.txt","a")
    file.write(str(finalAns) + "\n")
    file.close()



def main():

    condition = sys.argv[1]
    matrixFile = sys.argv[2]
    labelsFile = sys.argv[3]
    rateOrWeights = sys.argv[4]


    if (condition == "train"):
        learningRate = float(rateOrWeights) # learning rate
        train(matrixFile,labelsFile,learningRate)
    elif (condition == "test"):
        test(matrixFile,labelsFile,rateOrWeights)

main()