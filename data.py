from collections import Counter
import string
from random import shuffle
import numpy as np

def createTrainingSetAndBuildVocab(filename,vacabSize):
    vocab = buildVocab(filename,vacabSize)
    trainingSetWordFormat,trainingSetIndicesFormat = readData(filename,vocab)
    return vocab,trainingSetWordFormat,trainingSetIndicesFormat

def buildVocab(filename,maxWords):
    wordCounter=Counter()
    counter=0
    
    dataFile=open(filename,'r')
    for line in dataFile:
        #ignore new document lines
        if '===' in line:
            continue
        
        [arg,label]=line.rstrip().split('\t')
        #ignore label -1 args for vocab
        if label==-1:
            continue
        
        #update wordCounter with words of this arg
        wordCounter.update(arg.translate(None, string.punctuation).split())
        
        #simple break out of loop
        counter+=1
        if counter==3:
            break
    
    dataFile.close()
    
    topWords = wordCounter.most_common(maxWords)
    topWords = [word for (word,count) in topWords]
    shuffle(topWords)
    topWords = ['<s>','</s>','UNK'] + topWords
    wordDict=dict([(word,index) for index,word in enumerate(topWords)])
    return wordDict
    
    
def getIndices(arg1,wordDict):    
    indices=[wordDict['<s>']]
    for word in arg1.split():
        if word in wordDict:
            indices.append(wordDict[word])
        else:
            indices.append(wordDict['UNK'])
    indices.append(wordDict['</s>'])
    return indices

def readData(filename,wordDict):
    trainingSetWordFormat=[]
    trainingSetIndicesFormat=[]
    arg1=None

    dataFile=open(filename,'r')
    for line in dataFile:
        #starting a new document. Set arg1 to None
        if '===' in line:
            arg1=None
            continue
            
        #set current line to arg2 and create new training data point    
        [arg2,label]=line.rstrip().split('\t')
        if arg1!=None:# and label!="-1":
            arg1Indices=getIndices(arg1,wordDict)
            arg2Indices=getIndices(arg2,wordDict)
            trainingSetWordFormat.append((arg1,arg2,label))
            trainingSetIndicesFormat.append((arg1Indices,arg2Indices,label))
        
        #prepare for next iteration, current arg1=arg2. Future: numbers just get their own token, also can use root word 
        arg1=arg2

    dataFile.close()
        
    return trainingSetWordFormat,trainingSetIndicesFormat

