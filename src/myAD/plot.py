import helper
from mydetector import mad, Models
import matplotlib.pyplot as plt
import os

PATH = "./Plots"


def plotTrace(data,trace = 0):
    datasets = {}
    #Select trace from data:
    trace = data[trace]
    for i in range(len(trace)):   #For each moment in trace
        for j in range(len(trace[i])):    #For each feature of specific moment
            if "f"+str(j) not in datasets:
                #Create as many datasets as the features number
                #We are planning to do forecast on each feature
                datasets["f"+str(j)] = {"x":[],"y":[]}
            datasets["f"+str(j)]["x"].append(i)
            datasets["f"+str(j)]["y"].append(trace[i][j])
    for j in range(len(trace[i])):
        plot(datasets["f"+str(j)]["x"],datasets["f"+str(j)]["y"],f"Feature {str(j)}")

def plot(x,y,title):
    fig, ax = plt.subplots()
    ax.scatter(x, y,  label=title)
    ax.set_xlabel('Moment')
    ax.set_ylabel('Feature Value')
    ax.set_title(f'{title} value changes in time')
    fig.savefig(f'{PATH}//{title}.png')

if __name__ == '__main__':
    data=helper.get_test_data()
    datatrain=helper.get_train_data()
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    plotTrace(datatrain["train"],0)