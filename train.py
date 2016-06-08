import data
import numpy as np
import externalMemoryNetwork


filename="../data/pdtb-for-naacl2016/dev-levelone.txt"
vocabSize=300
totalIts=20
MBSize=20
vocabSize,vocab,trainingSetWordFormat,trainingSetIndicesFormat=data.createTrainingSetAndBuildVocab(filename,vocabSize)


wordEmbeddings=np.identity(vocabSize)
emn=externalMemoryNetwork.ExternalMemoryNetwork(MBSize, vocabSize, 30, 4, .09)

for iter in range(totalIts):
    totalCost = 0.
    correct = 0
    totalEgs=0
    for eg in trainingSetIndicesFormat:
        arg1=eg[0]
        target=eg[2]
        predicted, cost = emn.train(wordEmbeddings[arg1],target)
        if int(predicted) == int(target):
            correct+=1
        totalEgs+=1
        totalCost+=cost
    print "Iter:%d\tCost:%f\tAccuracy:%f "%(iter, cost, correct*1.0/totalEgs)