import random
import sys

#check arguments
nbArgs = len(sys.argv)
if nbArgs < 2:
    print("Please define drift type")
    exit()
drift_type = sys.argv[1]

percent = 0.4
totalLayers = 12
layers = list(range(totalLayers))

while(len(layers) > 1):
    # engage 40% of all layers randomly
    random.shuffle(layers)
    index = int(percent * len(layers) + 0.5)
    engageLayers = layers[:index]
    print(engageLayers) # 0 based !
    if len(engageLayers) == 1:
        break

    layers = layers[index:]