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
        targetClass = theano.tensor.iscalar('targetClass')

        Wh = self.createParameterMatrix('Wh', (self.h, self.h))
        Wx = self.createParameterMatrix('Wx', (self.wordEmbeddingSize, self.h))
        Wop = self.createParameterMatrix('Wop', (self.h, numLabels))
        MB = self.createParameterMatrix('MB', (N, self.h))
        #
        def RNN1(x, prev_h, Wh, Wx):
            h = theano.tensor.nnet.sigmoid(theano.tensor.dot( Wh,prev_h) + theano.tensor.dot(x, Wx))
            return h



        initialHiddenVector = theano.tensor.alloc(np.array(0, dtype=floatX), self.h)
        hiddenDims,_ = theano.scan(RNN1,sequences = arg1Embeddings, outputs_info = initialHiddenVector ,non_sequences=[Wh, Wx])
        finalHiddenDim = hiddenDims[-1]
        finalLabelDist = theano.tensor.nnet.softmax(theano.tensor.dot(finalHiddenDim, Wop))[0]
        predictedClass = theano.tensor.argmax(finalLabelDist)
        cost= -1.0 * theano.tensor.log(finalLabelDist[targetClass])

        for param in self.params.values():
            cost += l2_regularisation * theano.tensor.sqr(param).sum()

        gradients = theano.tensor.grad(cost,self.params.values())
        updates = [(w, w - self.learningRate*g) for w,g in zip(self.params.values(),gradients)]

        self.train = theano.function([arg1Embeddings, targetClass], [predictedClass,cost], updates = updates, allow_input_downcast=True)
        # self.train = theano.function([arg1Embeddings, targetClass], [predictedClass,cost], updates = updates, allow_input_downcast=True)
        # self.train = theano.function([arg1Embeddings, arg2, targetClass], [finalHiddenDim], updates = updates)

    def createParameterMatrix(self,name,size):
        vals = np.asarray(self.rng.normal(loc=0.0, scale=0.1, size=size), dtype=floatX)
        self.params[name] = theano.shared(vals, name)
        return self.params[name]