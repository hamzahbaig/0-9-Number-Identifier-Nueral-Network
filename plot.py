import numpy as np
from matplotlib import pyplot as plt

learningRate = []
accuracy = []
timeTaken = []

def createPlot(x,y,title,xLabel,ylabel):

	plt.plot(x,y, color = 'g')
	plt.title(title)
	plt.xlabel(xLabel)
	plt.ylabel(ylabel)
	plt.show()

def main():

	
	file = open("plot.txt","r")
	for _ in range(5):
		data = file.readline()
		data = data.strip("[")
		data = data.replace("]","")
		for _ in range(3):
			data = data.replace(",","")
		data = data.split()
		a = float(data[0])
		timeTaken.append(a)
		learningRate.append(data[1])
		accuracy.append(data[2])

	createPlot(learningRate,accuracy,"LearningRate VS Accuracy","Learning Rate","Accuracy")
	createPlot(timeTaken,accuracy,"timeTaken VS Accuracy","timeTaken","Accuracyc")
main()