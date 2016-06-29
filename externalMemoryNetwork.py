import theano
import numpy as np
import collections
import math

floatX=theano.config.floatX
class ExternalMemoryNetwork:
    def __init__(self, N, vocabSize,hidden_dim, numLabels, learningRate):

        self.params = collections.OrderedDict()
        self.h = hidden_dim
        self.wordEmbeddingSize = vocabSize
        self.learningRate = learningRate
        l2_regularisation = 0.0001
        random_seed=42
        # random number generator
        self.rng = np.random.RandomState(random_seed)

        arg1Embeddings = theano.tensor.imatrix('arg1Embeddings')
        erase = theano.shared(np.ones((1,self.h)), name='erase')
        targetClass = theano.tensor.iscalar('targetClass')
        arg2Embeddings = theano.tensor.imatrix('arg2Embeddings')
        arg2Indices = theano.tensor.ivector('arg2Indices')
        # MB = theano.tensor.matrix('MB', dtype=floatX)
        initMB = theano.tensor.matrix('initMB', dtype=floatX)
        oneHotLabel = theano.tensor.ivector('oneHotLabel')

        #init RNN1 shared vars
        Wh1 = self.createParameterMatrix('Wh1', (self.h, self.h), "Uniform")
        Wx1 = self.createParameterMatrix('Wx1', (self.wordEmbeddingSize, self.h),"Uniform")
        b1 = self.createParameterMatrix('b1', (self.h))

        # Wh1 = self.createParameterMatrix('Wh1', (self.h, self.h))
        # Wx1 = self.createParameterMatrix('Wx1', (self.wordEmbeddingSize, self.h))
        Wh_l = self.createParameterMatrix('Wh_l', (self.h, numLabels))
        bL = self.createParameterMatrix('b_l', (numLabels))
        # MB = self.createParameterMatrix('MB', (N, self.h))

        #init RNN2 shared vars
        Wh2 = self.createParameterMatrix('Wh2', (numLabels, self.h, self.h),"Uniform")
        Wx2 = self.createParameterMatrix('Wx2', (numLabels, self.wordEmbeddingSize, self.h),"Uniform")
        b2 = self.createParameterMatrix('b2', (numLabels, self.h),"Uniform")

        WV2 = self.createParameterMatrix('WV2', (numLabels, self.h, vocabSize))
        bV = self.createParameterMatrix('bV', (numLabels, vocabSize))
        #
        def RNN1(x, prevH, prevMB,Wh, Wx,b1, erase):
            #calculate next hidden state
            h = theano.tensor.nnet.sigmoid(theano.tensor.dot( prevH,Wh) +b1+ theano.tensor.dot(x, Wx))

            #find attention over memory based on hidden state
            weightDistribution = theano.tensor.dot(prevMB, theano.tensor.transpose(h))
            normalizedWeightDistribution = weightDistribution/weightDistribution.sum()
            normalizedWeightDistribution = normalizedWeightDistribution.reshape((1,normalizedWeightDistribution.shape[0]))

            #erase memory block
            curMB = prevMB - theano.tensor.dot(theano.tensor.transpose(normalizedWeightDistribution), erase)
            # MB = MB - theano.tensor.dot(theano.tensor.transpose(normalizedWeightDistribution), erase)


            #write to memory block {normalizedWeightDistribution:(1,20),
            curMB = curMB + theano.tensor.dot(theano.tensor.transpose(normalizedWeightDistribution), h.reshape((1,h.shape[0])))

            return [h,curMB]

        def RNN2P(curWord, nextWordIndex, prevH, probPrevWord, Wh, Wx, WV,b2,bV, pooledMemoryUnit):
            #calculate next hidden state

            v1=theano.tensor.batched_dot(prevH,Wh) #(l*h) * (l*h*h) -> l*h
            v2=theano.tensor.dot(curWord, Wx)#(x) * (l*x*h)   -> l*h
            h = theano.tensor.nnet.sigmoid(v1 + v2 +b2) #l*h
            MBFrac=0.1
            hForPredict = (1-MBFrac) * h + MBFrac * pooledMemoryUnit

            probOverWords = theano.tensor.nnet.softmax(theano.tensor.batched_dot(hForPredict,WV)+bV) #softmax{(l*h) * (l*h*V) -> (l*V)}
            probNextWord =  theano.tensor.log(probOverWords[:,nextWordIndex]) # array([l])

            return [h, probNextWord]

        def calcHArg1():
            initialHiddenVector = theano.tensor.alloc(np.array(0, dtype=floatX), self.h)
            (hiddenDims1,MBStates),_ = theano.scan(RNN1,sequences = arg1Embeddings, outputs_info = [initialHiddenVector,initMB] ,non_sequences=[Wh1, Wx1,b1, erase])
            finalHiddenDim1 = hiddenDims1[-1]
            finalMBState = MBStates[-1]
            finalLabelDist1 = theano.tensor.nnet.softmax(theano.tensor.dot(finalHiddenDim1, Wh_l)+bL)[0]
            return (finalLabelDist1,finalMBState)

        def calcProbArg2AllLabels(MBState):
            pooledMemoryUnit = MBState[theano.tensor.argmax(theano.tensor.square(MBState).sum(axis=1))]
            #todo remove target class as initial scalar
            initialHiddenVector = theano.tensor.alloc(np.array(0, dtype=floatX), numLabels,self.h) #todo define centrally

            initHiddenProbability = theano.shared(np.zeros(numLabels,dtype=floatX), name="initHiddenProbability")
            # hiddenDimsArg2Probs,_ = theano.scan(RNN2P,sequences = [arg2Embeddings[0:-1], arg2Indices[1:]], outputs_info = [initialHiddenVector, initHiddenProbability] ,non_sequences=[Wh2[targetClass], Wx2[targetClass], WV2[targetClass],pooledMemoryUnit])
            hiddenDimsArg2Probs,_ = theano.scan(RNN2P,sequences = [arg2Embeddings[0:-1], arg2Indices[1:]], outputs_info = [initialHiddenVector, initHiddenProbability] ,non_sequences=[Wh2, Wx2, WV2,b2,bV,pooledMemoryUnit])
            arg2Probs=hiddenDimsArg2Probs[1]
            # probArg2Total=arg2Probs.sum()
            # probArg2Total=theano.printing.Print("Prob each label for arg2")(arg2Probs.sum(axis=0))
            probArg2Total=theano.tensor.nnet.softmax(arg2Probs.sum(axis=0))[0]
            return probArg2Total



        def predictExample():
            # finalLabelDist1 = theano.printing.Print('t1')(-1 * theano.tensor.log(calcHArg1()))
            # probArg2Total = theano.printing.Print('t2')(calcProbArg2AllLabels())
            labelDist,MBState=calcHArg1()
            finalLabelDist1 = -1 * theano.tensor.log(labelDist)
            probArg2Total = calcProbArg2AllLabels(MBState)
            labels = -1* (probArg2Total +finalLabelDist1)
            cost = -1 * labels[targetClass]

            return theano.tensor.argmax(labels),cost


        def trainExampleCE():

            finalLabelDist1, MBState = calcHArg1()
            #todo add so that memory is pooled based on which label is correct
            probArg2Total = calcProbArg2AllLabels(MBState)

            logProbabilityOfTargetLabel=    theano.tensor.nnet.softmax(finalLabelDist1 + probArg2Total)[0]
            predictedClass = theano.tensor.argmax(logProbabilityOfTargetLabel)
            cost =theano.tensor.nnet.categorical_crossentropy(logProbabilityOfTargetLabel, oneHotLabel)
            # cost = theano.printing.Print("before")(theano.tensor.nnet.categorical_crossentropy(logProbabilityOfTargetLabel, oneHotLabel))


            #todo bring back the reg
            for param in self.params.values():
                cost += l2_regularisation * theano.tensor.sqr(param).sum()
                # cost += theano.printing.Print("after")(l2_regularisation * theano.tensor.sqr(param).sum())



            return cost, predictedClass

        def getUpdates(cost):
            gradients = theano.tensor.grad(cost,self.params.values())
            updates = [(w, w - self.learningRate*g) for w,g in zip(self.params.values(),gradients)]
            return updates

        cost,predictedClass = trainExampleCE()
        updates=getUpdates(cost)
        newClass,newCost=predictExample()

        #calculate the update to weights based on the cost function
        self.train = theano.function([arg1Embeddings, arg2Embeddings, arg2Indices,oneHotLabel, initMB], [predictedClass,cost], updates = updates, allow_input_downcast=True)

        #calculate the cost at the iteration but dont make any updates to the weights
        self.train_no_update = theano.function([arg1Embeddings, arg2Embeddings, arg2Indices,oneHotLabel, initMB], [predictedClass,cost], allow_input_downcast=True)

        #only predict the class of the pair
        self.predict = theano.function([arg1Embeddings, arg2Embeddings, arg2Indices, targetClass, initMB], [newClass,newCost], allow_input_downcast=True)


    def createParameterMatrix(self,name,size,type="Normal"):
        if type == "Uniform":
            vals = np.asarray(np.random.uniform(low = - math.sqrt(6./sum(size)), high=math.sqrt(6./sum(size)), size=size), dtype=floatX)
        else:
            vals = np.asarray(self.rng.normal(loc=0.0, scale=0.1, size=size), dtype=floatX)
        self.params[name] = theano.shared(vals, name)
        return self.params[name]