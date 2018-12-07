import random
import sys
import datetime
from nist_classifier import *

#check arguments
nbArgs = len(sys.argv)
if nbArgs < 2:
    print("Please define drift type")
    exit()
drift_type = sys.argv[1]

percent = 0.3 # 30%
totalLayers = 12
layers = list(range(totalLayers))

beginTime = datetime.datetime.now()

topList = []
numberLayersEngaged = 0
while(len(layers) > 0):
    random.shuffle(layers)
    index = int(percent * len(layers) + 0.5)
    if index == 0:
        break
    engageLayers = layers[:index] # 30% of random layers
    layers = layers[index:]
    accuracyList = []
    for layer in engageLayers:
        # calculate the final accuracy of this engagement
        acc = Run_one_engagement(drift_type, layer+1)
        numberLayersEngaged += 1
        accuracyList.append((layer, acc))
    accuracyList.sort(key=lambda tup: tup[1], reverse=True)
    print("accuracyList: ", accuracyList)
    if len(topList) == 0: # first time
        topList = accuracyList
        index = int(len(topList) * percent + 0.5)
        index = max(1, index) # at least one
        topList = topList[:index] # the best 30% in accuracy 
    else:
        if max(accuracyList, key=lambda tup: tup[1])[1] <= min(topList, key=lambda tup: tup[1])[1]: # all accuracy is smaller --> stop
            break
        for acc in accuracyList:
            for i in range(len(topList)-1, -1, -1): # reverse the topList --> lower accuracy is replaced first
                if acc[1] > topList[i][1]:
                    topList[i] = acc # replace
                    break
            topList.sort(key=lambda tup: tup[1], reverse=True)

    print("engageLayers: ", engageLayers)
    print("topList: ", topList)
print("number of layers engaged: ", numberLayersEngaged)

endTime = datetime.datetime.now()
print(endTime - beginTime)
    