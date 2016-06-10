import theano
import numpy as np
import collections

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
        initHiddenProbability = theano.shared(0.0, name="initHiddenProbability")

        #init RNN1 shared vars
        Wh1 = self.createParameterMatrix('Wh1', (self.h, self.h))
        Wx1 = self.createParameterMatrix('Wx1', (self.wordEmbeddingSize, self.h))
        Wop = self.createParameterMatrix('Wop', (self.h, numLabels))
        MB = self.createParameterMatrix('MB', (N, self.h))

        #init RNN2 shared vars
        Wh2 = self.createParameterMatrix('Wh2', (numLabels, self.h, self.h))
        Wx2 = self.createParameterMatrix('Wx2', (numLabels, self.wordEmbeddingSize, self.h))
        WV2 = self.createParameterMatrix('WV2', (numLabels, self.h, vocabSize))
        #
        def RNN1(x, prevH, Wh, Wx, MB, erase):
            #calculate next hidden state
            h = theano.tensor.nnet.sigmoid(theano.tensor.dot( prevH,Wh) + theano.tensor.dot(x, Wx))

            #find attention over memory based on hidden state
            weightDistribution = theano.tensor.dot(MB, theano.tensor.transpose(h))
            normalizedWeightDistribution = weightDistribution/weightDistribution.sum()
            normalizedWeightDistribution = normalizedWeightDistribution.reshape((1,normalizedWeightDistribution.shape[0]))

            #erase memory block
            MB = MB - theano.tensor.dot(theano.tensor.transpose(normalizedWeightDistribution), erase)
            # MB = MB - theano.tensor.dot(theano.tensor.transpose(normalizedWeightDistribution), erase)


            # # #write to memory block
            # MB = MB + theano.tensor.dot(theano.tensor.transpose(normalizedWeightDistribution), h)

            h=h + 0.0000001*MB[0]
            return h

        def RNN2(curWord, nextWordIndex, prevH, probPrevWord, Wh, Wx, WV, pooledMemoryUnit):
            #calculate next hidden state
            h = theano.tensor.nnet.sigmoid(theano.tensor.dot(prevH,Wh) + theano.tensor.dot(curWord, Wx))

            hForPredict = 0.5 * h + 0.5 * pooledMemoryUnit
            probOverWords = theano.tensor.nnet.softmax(theano.tensor.dot(hForPredict,WV))[0]
            probNextWord = -1 * theano.tensor.log(probOverWords[nextWordIndex])

            return [h, probNextWord]

        # def RNN2Predict(curWord, nextWordIndex, prevH, probNextWord, Wh, Wx, WV, pooledMemoryUnit):
        #     h = theano.tensor.nnet.sigmoid(theano.tensor.tensordot(prevH,Wh,axes=[[2,1],[1,0]]) + theano.tensor.tensordot(curWord, Wx, axes=[1,1]))
        #
        #     hForPredict = 0.5 * h + 0.5 * pooledMemoryUnit
        #     probOverWords = theano.tensor.nnet.softmax(theano.tensor.tensordot(hForPredict,WV))
        #     probNextWord = -1 * theano.tensor.log(probOverWords[:,nextWordIndex])
        #
        #     return [h, probNextWord]
        #     pass



        def calcHArg1():
            initialHiddenVector = theano.tensor.alloc(np.array(0, dtype=floatX), self.h)
            hiddenDims1,_ = theano.scan(RNN1,sequences = arg1Embeddings, outputs_info = initialHiddenVector ,non_sequences=[Wh1, Wx1, MB, erase])
            finalHiddenDim1 = hiddenDims1[-1]
            finalLabelDist1 = theano.tensor.nnet.softmax(theano.tensor.dot(finalHiddenDim1, Wop))[0]
            return finalLabelDist1

        def calcProbArg2ForTarget():
            pooledMemoryUnit = MB[theano.tensor.argmax(theano.tensor.square(MB).sum(axis=1))]
            #todo remove target class as initial scalar
            initialHiddenVector = theano.tensor.alloc(np.array(0, dtype=floatX), self.h) #todo define centrally
            hiddenDimsArg2Probs,_ = theano.scan(RNN2,sequences = [arg2Embeddings[0:-1], arg2Indices[1:]], outputs_info = [initialHiddenVector, initHiddenProbability] ,non_sequences=[Wh2[targetClass], Wx2[targetClass], WV2[targetClass],pooledMemoryUnit])
            arg2Probs=hiddenDimsArg2Probs[1]
            probArg2Total=arg2Probs.sum()
            return probArg2Total

        # def calcProbArg2ForAllLabels():
        #     pooledMemoryUnit = MB[theano.tensor.argmax(theano.tensor.square(MB).sum(axis=1))]
        #     #todo remove target class as initial scalar
        #     initialHiddenVector = theano.tensor.alloc(np.array(0, dtype=floatX), (1, numLabels, self.h)) #todo define centrally
        #     hiddenDimsArg2Probs,_ = theano.scan(RNN2Predict,sequences = [arg2Embeddings[0:-1], arg2Indices[1:]], outputs_info = [initialHiddenVector, initHiddenProbability] ,non_sequences=[Wh2, Wx2, WV2,pooledMemoryUnit])
        #     arg2Probs=hiddenDimsArg2Probs[1]
        #     probArg2Total=arg2Probs.sum()
        #     return probArg2Total

        def predictExample():
            finalLabelDist1 = calcHArg1()
            #todo add so that memory is pooled based on which label is correct
            probArg2Total = calcProbArg2ForTarget(targetClass)
            predictedClass = theano.tensor.argmax(finalLabelDist1)

        def trainExample():

            finalLabelDist1 = calcHArg1()
            #todo add so that memory is pooled based on which label is correct
            probArg2Total = calcProbArg2ForTarget()
            predictedClass = theano.tensor.argmax(finalLabelDist1)

            cost= -1.0 * theano.tensor.log(finalLabelDist1[targetClass]) + probArg2Total
            for param in self.params.values():
                cost += l2_regularisation * theano.tensor.sqr(param).sum()

            gradients = theano.tensor.grad(cost,self.params.values())
            updates = [(w, w - self.learningRate*g) for w,g in zip(self.params.values(),gradients)]

            return updates,cost, predictedClass

        updates,cost,predictedClass = trainExample()

        self.train = theano.function([arg1Embeddings, arg2Embeddings, arg2Indices, targetClass], [predictedClass,cost], updates = updates, allow_input_downcast=True)
        # self.predict = theano.function([arg1Embeddings, arg2Embeddings, arg2Indices], [predictedClass,cost], allow_input_downcast=True)
        # self.train = theano.function([arg1Embeddings, targetClass], [predictedClass,cost], updates = updates, allow_input_downcast=True)
        # self.train = theano.function([arg1Embeddings, arg2, targetClass], [finalHiddenDim1], updates = updates)

    def createParameterMatrix(self,name,size):
        vals = np.asarray(self.rng.normal(loc=0.0, scale=0.1, size=size), dtype=floatX)
        self.params[name] = theano.shared(vals, name)
        return self.params[name]