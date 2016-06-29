import data
import numpy as np
import externalMemoryNetwork
import theano
import pickle
from collections import Counter

filename="../data/correct_data/trn-levelone.txt.10K"
test_filename="../data/correct_data/tst-levelone.txt.10K"


# filename="../data/pdtb-for-naacl2016/dev-levelone.txt"
# test_filename="../data/pdtb-for-naacl2016/tst-levelone.txt"
# vocabSize=1000#300
vocabSize=300
totalIts=40
MBSlots=20
# hiddenDim=30#6
hiddenDim=6
MBSize=(MBSlots,hiddenDim)
numLabels=4
learningRate=0.02
# learningRate=0.05#85% accuracy
floatX=theano.config.floatX


vocabSize,vocab,trainingSetWordFormat,trainingSetIndicesFormat=data.createTrainingSetAndBuildVocab(filename,vocabSize)
testSetWordFormat,testSetIndicesFormat=data.createTestSet(test_filename,vocab)
with open('testTrain-'+str(vocabSize), 'w') as f:  # Python 3: open(..., 'wb')
    pickle.dump([vocabSize,vocab,trainingSetWordFormat,trainingSetIndicesFormat,testSetWordFormat,testSetIndicesFormat], f)
# with open('testTrain-'+str(vocabSize+3)) as f:  # Python 3: open(..., 'rb')
#     vocabSize,vocab,trainingSetWordFormat,trainingSetIndicesFormat,testSetWordFormat,testSetIndicesFormat = pickle.load(f)



wordEmbeddings=np.identity(vocabSize)
random_seed = 42
rng = np.random.RandomState(random_seed)
initMB = np.asarray(rng.normal(loc=0.0, scale=0.1, size=MBSize), dtype=floatX)
emn=externalMemoryNetwork.ExternalMemoryNetwork(MBSize, vocabSize, hiddenDim, numLabels, learningRate)


def trainOrTest(dataSetIndicesFormat,updateWeights = True ,type = "Train"):
    mcc=Counter()
    totalEgs=0
    totalCost=0
    correct_before_update=0
    for (arg1,arg2,target) in dataSetIndicesFormat:
        oneHotLabel=np.zeros((numLabels))
        oneHotLabel[target]=1
        if updateWeights==True:
            predicted_before_update, cost = emn.train(wordEmbeddings[arg1],wordEmbeddings[arg2],arg2,oneHotLabel,initMB)
        else:
            predicted_before_update,cost = emn.train_no_update(wordEmbeddings[arg1],wordEmbeddings[arg2],arg2,oneHotLabel,initMB)
        mcc.update({target:1})
        if int(predicted_before_update) == int(target):
            correct_before_update+=1

        totalEgs+=1
        totalCost+=cost

    if not updateWeights:
        if type=="Train":
            print "%s Iter:%d \tCost:%f\tAccuracy:%f\tMCC: %f"\
          %(type, iter, totalCost, correct_before_update*1.0/totalEgs, mcc.most_common(1)[0][1]*1.0/totalEgs),
        else:
            print "Test   Cost:%f\tAccuracy:%f\tMCC: %f"%(totalCost,correct_before_update*1.0/totalEgs,mcc.most_common(1)[0][1]*1.0/totalEgs)


    return totalCost

prevCost=999999
iter=0

while True:

    firstCost=trainOrTest(trainingSetIndicesFormat,updateWeights=True,type="Train")
    totalCost=trainOrTest(trainingSetIndicesFormat,updateWeights=False,type="Train")
    useless=trainOrTest(testSetIndicesFormat,updateWeights=False,type="Test")
    if totalCost>prevCost:
        break
    if prevCost<totalCost:
        if iter>20:
            break
        print "Costs: Prev: %f\tNew: %f"%(prevCost,totalCost)
    iter+=1
    # for (arg1,arg2,target) in trainingSetIndicesFormat:
    #     oneHotLabel=np.zeros((numLabels))
    #     # arg1=eg[0]
    #     # arg2=eg[1]
    #     # target=eg[2]
    #     oneHotLabel[target]=1
    #     predicted_before_update, cost = emn.train(wordEmbeddings[arg1],wordEmbeddings[arg2],arg2,oneHotLabel,initMB)
    #     predicted_after_update,newcost = emn.train_no_update(wordEmbeddings[arg1],wordEmbeddings[arg2],arg2,oneHotLabel,initMB)
    #     if int(predicted_before_update) == int(target):
    #         correct_before_update+=1
    #     if int(predicted_after_update) == int(target):
    #         correct_after_update+=1
    #
    #     totalEgs+=1
    #     totalCost+=cost
    #     totalNewCost+=newcost
    #
    #
    # print "Iter:%d Train\tCostBefore:%f\tCostAfter%f\tAccuracyBefore:%f\tAccuracyAfter:%f"\
    #       %(iter, totalCost,totalNewCost, correct_before_update*1.0/totalEgs, correct_after_update*1.0/totalEgs)



