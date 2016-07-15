import theano
import numpy as np
import collections
import math

floatX=theano.config.floatX
class ExternalMemoryNetwork:
    def __init__(self, vocabSize,embeddingSize, hidden_dim, numLabels):

        self.params = collections.OrderedDict()
        self.h = hidden_dim
        self.wordEmbeddingSize = embeddingSize

        l2_regularisation = 0.0005 #todo check why this is giving nans for higher vals
        random_seed=42
        # random number generator
        self.rng = np.random.RandomState(random_seed)

        learningRate = theano.tensor.fscalar('learningRate')
        arg1Embeddings = theano.tensor.fmatrix('arg1Embeddings')
        erase = theano.shared(np.ones((1,self.h)), name='erase')
        targetClass = theano.tensor.iscalar('targetClass')
        arg2Embeddings = theano.tensor.fmatrix('arg2Embeddings')
        arg2Indices = theano.tensor.ivector('arg2Indices')
        initMB = theano.tensor.matrix('initMB', dtype=floatX)
        oneHotLabel = theano.tensor.ivector('oneHotLabel')

        def getLSTMParameters(LSTMNumber):
            #init LSTM1 shared vars
            if True:
            # if LSTMNumber == 1:
                hhTuple = (self.h, self.h)
                xhTuple = (self.wordEmbeddingSize, self.h)
                hTuple = (self.h)
            else:
                hhTuple = (numLabels, self.h, self.h)
                xhTuple = (numLabels, self.wordEmbeddingSize, self.h)
                hTuple = (numLabels, self.h)
            LSTMNumber=str(LSTMNumber)

            #input gate
            Whi = self.createParameterMatrix('Whi_'+LSTMNumber, hhTuple, "Ortho")
            Wxi= self.createParameterMatrix('Wxi_'+LSTMNumber, xhTuple,"Ortho")
            bi= self.createParameterMatrix('bi_'+LSTMNumber, hTuple)

            #forget gate
            Whf = self.createParameterMatrix('Whf_'+LSTMNumber, hhTuple, "Ortho")
            Wxf = self.createParameterMatrix('Wxf_'+LSTMNumber, xhTuple,"Ortho")
            bf = self.createParameterMatrix('bf_'+LSTMNumber, hTuple)

            #output gate
            Who = self.createParameterMatrix('Who_'+LSTMNumber, hhTuple, "Ortho")
            Wxo = self.createParameterMatrix('Wxo_'+LSTMNumber, xhTuple,"Ortho")
            bo = self.createParameterMatrix('bo_'+LSTMNumber, hTuple)

            #input gate gate
            Whg = self.createParameterMatrix('Whg_'+LSTMNumber, hhTuple, "Ortho")
            Wxg = self.createParameterMatrix('Wxg_'+LSTMNumber, xhTuple,"Ortho")
            bg = self.createParameterMatrix('bg_'+LSTMNumber, hTuple)


            return Whi, Wxi, bi, Whf, Wxf, bf, Who, Wxo, bo, Whg, Wxg, bg



        #init LSTM_arg1 shared vars
        Whi_1, Wxi_1, bi_1, Whf_1, Wxf_1, bf_1, Who_1, Wxo_1, bo_1, Whg_1, Wxg_1, bg_1 = getLSTMParameters(1)
        Wh_l = self.createParameterMatrix('Wh_l', (self.h, numLabels))
        bL = self.createParameterMatrix('b_l', (numLabels))

        #init LSTM_arg2 shared vars
        Whi_2, Wxi_2, bi_2, Whf_2, Wxf_2, bf_2, Who_2, Wxo_2, bo_2, Whg_2, Wxg_2, bg_2 = getLSTMParameters(2)
        Wh_lV= self.createParameterMatrix('Wh_lV', (numLabels, self.h, vocabSize))
        Wc_lV= self.createParameterMatrix('Wc_lV', (numLabels, self.h, vocabSize))
        bV = self.createParameterMatrix('bV', (numLabels, vocabSize))

        # Wh_lh = self.createParameterMatrix('Whl', (numLabels, self.h))


        def LSTM_arg1(x, prevH, prevCt, prevMB, Whi_1, Wxi_1, bi_1, Whf_1, Wxf_1, bf_1, Who_1, Wxo_1, bo_1, Whg_1, Wxg_1, bg_1, erase):
            #calculate next hidden state
            i = theano.tensor.nnet.hard_sigmoid(theano.tensor.dot( prevH,Whi_1) +bi_1+ theano.tensor.dot(x, Wxi_1))
            f = theano.tensor.nnet.hard_sigmoid(theano.tensor.dot( prevH,Whf_1) +bf_1+ theano.tensor.dot(x, Wxf_1))
            o = theano.tensor.nnet.hard_sigmoid(theano.tensor.dot( prevH,Who_1) +bo_1+ theano.tensor.dot(x, Wxo_1))

            g = theano.tensor.tanh( theano.tensor.dot(prevH, Whg_1) + bg_1 + theano.tensor.dot(x, Wxg_1) )
            ct = (f * prevCt) + (g * i)
            h = theano.tensor.tanh( ct ) * o

            #find attention over memory based on hidden state
            weightDistribution = theano.tensor.dot(prevMB, theano.tensor.transpose(h))
            #todo bring back
            # normalizedWeightDistribution = weightDistribution/weightDistribution.sum()
            normalizedWeightDistribution = theano.tensor.nnet.softmax(weightDistribution)[0]
            normalizedWeightDistribution = normalizedWeightDistribution.reshape((1,normalizedWeightDistribution.shape[0]))

            #erase memory block
            # curMB = prevMB - theano.tensor.dot(theano.tensor.transpose(normalizedWeightDistribution), erase)
            curMB = prevMB - theano.tensor.dot(theano.tensor.transpose(normalizedWeightDistribution), prevMB)#todo try
            # curMB = prevMB - theano.tensor.dot(theano.tensor.transpose(normalizedWeightDistribution), prevMB)#todo try

            #write to memory block {normalizedWeightDistribution:(1,20),
            curMB = curMB + theano.tensor.dot(theano.tensor.transpose(normalizedWeightDistribution), h.reshape((1,h.shape[0])))

            return [h, ct, curMB]

        def LSTM_arg2(curWord, nextWordIndex, prevH, prevCt, probPrevWord, Whi_2, Wxi_2, bi_2, Whf_2, Wxf_2, bf_2, Who_2, Wxo_2, bo_2, Whg_2, Wxg_2, bg_2,Wh_lV,Wc_lV,bV, pooledMemoryUnit):
            #calculate next hidden state
            i = theano.tensor.nnet.hard_sigmoid(theano.tensor.dot( prevH,Whi_2) +bi_2+ theano.tensor.dot(curWord, Wxi_2))
            f = theano.tensor.nnet.hard_sigmoid(theano.tensor.dot( prevH,Whf_2) +bf_2+ theano.tensor.dot(curWord, Wxf_2))
            o = theano.tensor.nnet.hard_sigmoid(theano.tensor.dot( prevH,Who_2) +bo_2+ theano.tensor.dot(curWord, Wxo_2))

            g = theano.tensor.tanh( theano.tensor.dot(prevH, Whg_2) + bg_2 + theano.tensor.dot(curWord, Wxg_2) )
            ct = (f * prevCt) + (g * i)
            h = theano.tensor.tanh( ct ) * o

            MBFrac = 0.5
            hForPredict = theano.tensor.nnet.sigmoid((1-MBFrac)*theano.tensor.dot(h,Wh_lV) +
                                                     MBFrac * theano.tensor.dot(pooledMemoryUnit,Wc_lV) + bV) # h -> l*V
            probOverWords = theano.tensor.nnet.softmax(hForPredict) #softmax{(l*h) * (l*h*V) -> (l*V)}
            probNextWord =  theano.tensor.log(probOverWords[:,nextWordIndex]) # array([l])


            return [h, ct, probNextWord]

        def LSTM_arg22(curWord, nextWordIndex, prevH, prevCt, probPrevWord, Whi_2, Wxi_2, bi_2, Whf_2, Wxf_2, bf_2, Who_2, Wxo_2, bo_2, Whg_2, Wxg_2, bg_2,WV,bV, pooledMemoryUnit):
            '''Dimension information about tensors
            v1=theano.tensor.batched_dot(prevH,Whi_1) #(l*h) * (l*h*h) -> l*h
            v2=theano.tensor.dot(curWord, Wxi_1)#(x) * (l*x*h)   -> l*h
            h = theano.tensor.nnet.sigmoid(theano.tensor.batched_dot(prevH,Whi_2) + theano.tensor.dot(curWord, Wxi_2) + bi_2) #l*h'''

            #calculate next hidden state
            i = theano.tensor.nnet.sigmoid(theano.tensor.batched_dot(prevH,Whi_2) + theano.tensor.dot(curWord, Wxi_2) + bi_2) #l*h
            f = theano.tensor.nnet.sigmoid(theano.tensor.batched_dot(prevH,Whf_2) + theano.tensor.dot(curWord, Wxf_2) + bf_2)
            o = theano.tensor.nnet.sigmoid(theano.tensor.batched_dot(prevH,Who_2) + theano.tensor.dot(curWord, Wxo_2) + bo_2)

            g = theano.tensor.tanh( theano.tensor.batched_dot(prevH,Whg_2) + theano.tensor.dot(curWord, Wxg_2) + bg_2)
            ct = (f * prevCt) + (g * i)
            h = theano.tensor.tanh( ct ) * o #l*h

            MBFrac=0.5 #todo change back
            # MBFrac=0.5 #todo change back
            hForPredict = theano.tensor.nnet.sigmoid((1-MBFrac) * h + MBFrac * pooledMemoryUnit) # l*h + h -> l*h

            probOverWords = theano.tensor.nnet.softmax(theano.tensor.batched_dot(hForPredict,WV)+bV) #softmax{(l*h) * (l*h*V) -> (l*V)}
            probNextWord =  theano.tensor.log(probOverWords[:,nextWordIndex]) # array([l])

            return [h, prevCt, probNextWord]

        def calcHArg1():
            initialHiddenVector = theano.tensor.alloc(np.array(0, dtype=floatX), self.h)
            (hiddenDims1,_,MBStates),_ = theano.scan(LSTM_arg1,sequences = arg1Embeddings, outputs_info = [initialHiddenVector,initialHiddenVector,initMB] ,
                                                     non_sequences=[Whi_1, Wxi_1, bi_1, Whf_1, Wxf_1, bf_1, Who_1, Wxo_1, bo_1, Whg_1, Wxg_1, bg_1, erase])
            finalHiddenDim1 = hiddenDims1[-1]
            finalMBState = MBStates[-1]
            finalLabelDist1 = theano.tensor.nnet.softmax(theano.tensor.dot(finalHiddenDim1, Wh_l)+bL)[0]
            return (finalLabelDist1,finalMBState)

        def calcProbArg2AllLabels(MBState):
            pooledMemoryUnit = MBState[theano.tensor.argmax(theano.tensor.square(MBState).sum(axis=1))]

            # pooledMemoryUnit = theano.tensor.nnet.sigmoid(Wh_lh * pooledMemoryUnit)#todo bring back
            #todo remove target class as initial scalar
            initialHiddenVector = theano.tensor.alloc(np.array(0, dtype=floatX),self.h) #todo define centrally

            initHiddenProbability = theano.shared(np.zeros(numLabels,dtype=floatX), name="initHiddenProbability")
            # (_,_,arg2Probs),_ = theano.scan(LSTM_arg2,sequences = [arg2Embeddings[0:-1], arg2Indices[0:-1]],
            (_,_,arg2Probs),_ = theano.scan(LSTM_arg2,sequences = [arg2Embeddings[0:-1], arg2Indices[1:]],#todo change back to get the word order
                                            outputs_info = [initialHiddenVector, initialHiddenVector, initHiddenProbability],
                                            non_sequences=[Whi_2, Wxi_2, bi_2, Whf_2, Wxf_2, bf_2, Who_2, Wxo_2, bo_2, Whg_2, Wxg_2, bg_2, Wh_lV, Wc_lV, bV, pooledMemoryUnit])#todo remove 2 from pooled unit

            probArg2Total=theano.tensor.nnet.softmax(arg2Probs.sum(axis=0))[0]
            return probArg2Total

        def trainExampleCE():

            finalLabelDist1, MBState = calcHArg1()
            #todo add so that memory is pooled based on which label is correct
            probArg2Total = calcProbArg2AllLabels(MBState)
            frac1 = 0.85
            #todo bring back
            # logProbabilityOfTargetLabel=    theano.tensor.nnet.softmax(finalLabelDist1 + probArg2Total)[0]
            # logProbabilityOfTargetLabel =  theano.tensor.nnet.softmax(theano.tensor.log(finalLabelDist1) + theano.tensor.log( probArg2Total))[0]
            # logProbabilityOfTargetLabel =  theano.tensor.nnet.softmax(theano.printing.Print('p1')((1-frac1)*finalLabelDist1) * theano.printing.Print('p2')(frac1*probArg2Total))[0]#todo bring back
            logProbabilityOfTargetLabel =  theano.tensor.nnet.softmax((1-frac1)* theano.tensor.log(finalLabelDist1) + frac1*theano.tensor.log(probArg2Total))[0]#todo bring back
            # logProbabilityOfTargetLabel=    theano.tensor.nnet.softmax(theano.tensor.log(finalLabelDist1) + theano.tensor.log(probArg2Total))[0]


            predictedClass = theano.tensor.argmax(logProbabilityOfTargetLabel)

            #todo maybe try a different loss
            # cost = -1 * theano.tensor.log(logProbabilityOfTargetLabel[targetClass])
            # cost = theano.printing.Print('Cost - Main')(theano.tensor.nnet.categorical_crossentropy(logProbabilityOfTargetLabel, oneHotLabel))
            cost = theano.tensor.nnet.categorical_crossentropy(logProbabilityOfTargetLabel, oneHotLabel)


            #todo check reg values if theyre comparable to above loss
            for param in self.params.values():
                cost += l2_regularisation * theano.tensor.sqr(param).sum()
                # cost += theano.printing.Print("Cost - Extra")(l2_regularisation * theano.tensor.sqr(param).sum())

            return cost, predictedClass



        def getUpdates(cost):
            gradients = theano.tensor.grad(cost,self.params.values())
            updates = [(w, w - learningRate*g) for w,g in zip(self.params.values(),gradients)]
            return updates

        cost,predictedClass = trainExampleCE()
        updates=getUpdates(cost)

        #calculate the update to weights based on the cost function
        self.train = theano.function([arg1Embeddings, arg2Embeddings, arg2Indices,oneHotLabel, initMB, learningRate], [predictedClass,cost], updates = updates, allow_input_downcast=True)

        #calculate the cost at the iteration but dont make any updates to the weights
        self.train_no_update = theano.function([arg1Embeddings, arg2Embeddings, arg2Indices,oneHotLabel, initMB], [predictedClass,cost], allow_input_downcast=True)

    def createParameterMatrix(self,name,size,type="Normal"):
        if type == "Ortho":
            randomMatrix = np.random.uniform(low = - math.sqrt(6./sum(size)), high=math.sqrt(6./sum(size)), size=size)
            # randomMatrix = self.rng.normal(loc=0.0, scale=0.1, size=size)
            u, s, vt = np.linalg.svd(randomMatrix, full_matrices = False)
            q = u if u.shape == size else vt
            vals = q.reshape(size)
        elif type == "Uniform":
            vals = np.asarray(np.random.uniform(low = - math.sqrt(6./sum(size)), high=math.sqrt(6./sum(size)), size=size), dtype=floatX)
        else:
            vals = np.asarray(self.rng.normal(loc=0.0, scale=0.1, size=size), dtype=floatX)# try scale = 1 by root m
        self.params[name] = theano.shared(vals, name)
        return self.params[name]

#diff
'''
Ideas:
1. change to dclm model
2. the way next word is predcted. Not just that word, but anything like that word
1. switching to ortho inits. Not yet done though
2. remove bias of model
3. changed to MSE
4. Should we try.. predict the cosine of the next word for each label?
 '''