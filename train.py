from __future__ import print_function
import data
import numpy as np
# import externalMemoryNetwork
import externalMemoryNetworkLSTM
import theano
import pickle
from collections import Counter
from gensim.models import word2vec
import math

def trainOrTest(dataSetIndicesFormat,dataSetEmbeddingsFormat,updateWeights = True ,printType = "Train", learningRate=None):
    mcc=Counter()
    totalEgs=len(dataSetIndicesFormat)
    totalCost=0
    correct_before_update=0
    for index,(arg1,arg2,target) in enumerate(dataSetIndicesFormat):
        if printType=="Train":
            print('Processed: ',index,' of ', totalEgs, end='\r')

        arg1Embeddings,arg2Embeddings,_ = dataSetEmbeddingsFormat[index] #todo change to access index if necessary
         # = dataSetEmbeddingsFormat[index][1]
        oneHotLabel=np.zeros((numLabels))
        oneHotLabel[target]=1

        if updateWeights==True:
            _,_ = emn.train(arg1Embeddings,arg2Embeddings,arg2,oneHotLabel,initMB, learningRate)
        else:
            predicted,cost = emn.train_no_update(np.array(arg1Embeddings),np.array(arg2Embeddings),arg2,oneHotLabel,initMB)
            mcc.update({target:1})
            if int(predicted) == int(target):
                correct_before_update+=1
            totalCost+=cost


    if not updateWeights:
        if printType=="Train":
            print('                                                                                        ', end='\r')
            print("%s Iter:%d \tCost:%f\tAccuracy:%f\tMCC: %f"\
          %(printType, iter, totalCost, correct_before_update*1.0/totalEgs, mcc.most_common(1)[0][1]*1.0/totalEgs),end='\t')
        else:
            print("Test   Cost:%f\tAccuracy:%f\tMCC: %f"%(totalCost,correct_before_update*1.0/totalEgs,mcc.most_common(1)[0][1]*1.0/totalEgs))


    return totalCost

if __name__ == '__main__':

    vocabSize=15000
    MBSlots=1
    hiddenDim=12#50#6,12
    # embeddingSize = 300
    embeddingSize = 50

    totalIts=40
    MBSize=(MBSlots,hiddenDim)
    numLabels=4
    floatX=theano.config.floatX
    random_seed = 42
    rng = np.random.RandomState(random_seed)
    prevCost=999999
    iter=0

    modelType = 'word2vecn'
    filename="../data/pdtb-for-naacl2016/trn-levelone.txt"
    # filename="../data/correct_data/trn-levelone.txt.10K"
    test_filename="../data/correct_data/tst-levelone.txt.10K"
    gloveFile = './glove.6B/glove.6B.'+str(embeddingSize)+'d.txt'

    #load the necessary model
    print("Preparing data...")
    if modelType == 'word2vec':
        print("Loading Word2Vec Embeddings...")
        model = word2vec.Word2Vec.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
        print("Done")
    else:
        model = data.loadGloveModel(gloveFile)

    #create training and test set
    vocab = data.buildVocab(filename,vocabSize)
    trainingSetWordFormat,trainingSetIndicesFormat,trainingSetEmbeddingsFormat=data.buildDataSet(filename,vocab,model) #todo check if its working
    testSetWordFormat,testSetIndicesFormat,testSetEmbeddingsFormat=data.buildDataSet(test_filename,vocab,model)
    with open('testTrain-'+str(vocabSize), 'w') as f:  # Python 3: open(..., 'wb')
        pickle.dump([vocabSize, trainingSetWordFormat,trainingSetIndicesFormat,testSetWordFormat,testSetIndicesFormat, trainingSetEmbeddingsFormat, testSetEmbeddingsFormat], f)

    # # load directly
    # with open('testTrain-'+str(vocabSize)) as f:  # Python 3: open(..., 'rb')
    #     vocabSize, trainingSetWordFormat,trainingSetIndicesFormat,testSetWordFormat,testSetIndicesFormat, trainingSetEmbeddingsFormat, testSetEmbeddingsFormat = pickle.load(f)

    print("Done")

    # initMB = np.asarray(rng.normal(loc=0.0, scale=0.1, size=MBSize), dtype=floatX)
    initMB = np.asarray(np.random.uniform(low = - math.sqrt(6./sum(MBSize)), high=math.sqrt(6./sum(MBSize)), size=MBSize), dtype=floatX)
    emn=externalMemoryNetworkLSTM.ExternalMemoryNetwork(vocabSize, embeddingSize, hiddenDim, numLabels)

    #todo remove change back
    # point = int(0.5 * len(trainingSetIndicesFormat))
    # trainingSetIndicesFormat = trainingSetIndicesFormat[0:point]
    # trainingSetEmbeddingsFormat = trainingSetEmbeddingsFormat[0:point]

    point = int(0.8 * len(trainingSetIndicesFormat))
    testSetIndicesFormat = trainingSetIndicesFormat[point:]
    testSetEmbeddingsFormat = trainingSetEmbeddingsFormat[point:]
    trainingSetIndicesFormat = trainingSetIndicesFormat[0:point]
    trainingSetEmbeddingsFormat = trainingSetEmbeddingsFormat[0:point]
    print("Start Training")
    learningRate = .05 /.9

    totalCost=trainOrTest(trainingSetIndicesFormat,trainingSetEmbeddingsFormat,updateWeights=False, printType="Train")
    useless=trainOrTest(testSetIndicesFormat,testSetEmbeddingsFormat,updateWeights=False, printType="Test")

    while True:
        iter+=1
        learningRate = learningRate*.9
        firstCost=trainOrTest(trainingSetIndicesFormat,trainingSetEmbeddingsFormat,updateWeights=True, printType="Train",learningRate=learningRate)
        totalCost=trainOrTest(trainingSetIndicesFormat,trainingSetEmbeddingsFormat,updateWeights=False, printType="Train")
        useless=trainOrTest(testSetIndicesFormat,testSetEmbeddingsFormat,updateWeights=False, printType="Test")
        if prevCost<totalCost:
            if iter>20:
                break
            print("Costs: Prev: %f\tNew: %f"%(prevCost,totalCost))

