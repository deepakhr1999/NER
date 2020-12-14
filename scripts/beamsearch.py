import torch
from models.utils import getSignal

class BeamSearcher:
    def __init__(self, beamSize, model):
        self.beamSize = beamSize
        self.model = model
        self.decoderUnits = model.sequenceLabeller.decoderUnits
        self.outputUnits = model.numTags
        
    def rootInitialization(self, batch):
        """Batch specific initialization for beam search.
            This method is called everytime a batch is to be processed
            The model passed will be called to encode the sequence
        """
        words, chars, charMask, targets = batch
        self.batchSize  = words.batch_sizes[0].item()
        self.batchArray = words.batch_sizes
        self.unsortedIndices = words.unsorted_indices

        # alive and dead ids
        self.live = list(range(self.batchSize))
        self.dead = []
        
        # find the length of each sequence
        self.lengths = torch.zeros(self.batchSize, dtype=torch.int)
        for x in words.batch_sizes:
            self.lengths[:x] += 1
            
        # values[i, j] contains the heuristic beam of the ith example. j in range(beamSize)
        # we maintain a queue like tensor, each example has a queue of size beamSize
        self.values = torch.zeros(self.batchSize, self.beamSize, device=self.model.device)
        #paths [i, j] contains the corresponding paths
        # beamSize no. of alive paths for each example
        self.paths  = [ [list() for _ in range(self.beamSize)] for _ in range(self.batchSize) ]
        
        """
            Pass the examples through the encoder
        """
        with torch.no_grad():
            encoded, initHiddenState, initPrevTarget = self.model.encode(words, chars, charMask)        
        self.hiddenState = initHiddenState
        self.prevTarget  = initPrevTarget
        
        # convert encoded into repeated pages
        start = 0
        self.encodedPages = []
        for pageLen in self.batchArray:
            if start == 0:
                page = encoded[start:start+pageLen] # first page is not repeated
            else:
                page = encoded[start:start+pageLen].repeat(self.beamSize, 1)
            self.encodedPages.append(page) # [e1, e2, e3, e1, e2, e3.. etc, repeated beamSize times]
            start += pageLen

    def pathSum(self, values, logProbs):
        """
            Adds the prev sum to current logProbs
            to get the effective logprob
        """
        # self.values is batch, beam
        values = torch.unsqueeze(values, -1) # batch, beam, 1
        values = values.repeat(1, 1, self.outputUnits)  # batch, beam, units
        values = values.permute(1,0,2)       # beam, batch, units
        values = torch.cat(list(values), -1) # batch, units * beam

        # logprobs is [batch, units*beam]
        ps = logProbs + values

        # ps is [batch, units*beam]
        return ps

    def updateHiddenState(self, depth, currBatchSize):
        """Model decodes once, updating the hiddenstate. Returns logProbs."""
        # add the timeSignal to the target embeddings
        self.prevTarget += getSignal(1, self.decoderUnits, depth, self.model.device)
        
        # update the hiddenState
        with torch.no_grad():
            self.hiddenState, logits = self.model.sequenceLabeller.decode_once(
                self.encodedPages[depth],
                self.prevTarget,
                self.hiddenState
            )
            
        # return logProbs from logits
        logProbs = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        # [b*beamSize, units] ie [l1, l2, l3, l1, l2, l3 ... numTag d vectors repeated]
        return logProbs
    
    def updateValues(self, depth, currBatchSize, logProbs):
        """
            Adds the logprobs to self.values, and returns indices
            resulting from the topk operation.
        """
        if depth == 0:
            ps = logProbs
        else:
            logProbs = logProbs.reshape((self.beamSize, currBatchSize, self.outputUnits))
            # now becomes [[l1, l2, l3], [l1, l2, l3], [l1, l2, l3]...]
            
            logProbs = torch.cat(list(logProbs), dim=-1)
            # [l1l1l1..., l2l2l2..., l3l3l3...]
            
            ps = self.pathSum(self.values[:currBatchSize], logProbs)

        # Filter the top beamSize no. of pathsums
        self.values[:currBatchSize], indices = ps.topk(dim=-1, k=self.beamSize) # values is [batch, beam]
        return indices
    
    def updatePaths(self, children, parents):
        """
            Each example has
                * beamSize candidate paths.
                * parent idx array
                * child idx array
            
            for each example
                for each i
                    the path at parent[i] has to be extended by child[i]
        """
        numFinished = 0
        for qidx, valBeam, childbeam, parentBeam in zip(self.live, self.values, children, parents):
            """
                Narrow our sight to each example:
                    At qidx, extend the path of parentBeam[i] with childBeam[i]
                    You will get beam no. of new paths.
                    This is your new path beam.
            """
            newQueue = []
            for v, c, p in zip(valBeam, childbeam, parentBeam):
                oldPath = self.paths[qidx][p]
                newPath = oldPath + [c.item()]
                newQueue.append(newPath)
            self.paths[qidx] = newQueue

            # Mark completed if the lenghts of the paths match the word count
            if len(newQueue[0]) == self.lengths[qidx]:
                numFinished += 1
                self.dead.append(qidx)
        return numFinished

    def updateForNextIteration(self, currBatchSize, numFinished, children, parents):
        # If an example has been done with, we remove it from the live array
        # Because of how a packed sequence works, we always remove from the end
        numAlive = currBatchSize-numFinished
        self.live = self.live[:numAlive]

        # idxs children and parents are [batchSize beamSize]
        # we remove the rows related to finished 
        children = children[:numAlive].T.reshape(-1)
        parents = parents[:numAlive].T.reshape(-1)

        # where each example is repeated beamSize times.
        # we take the transpose to follow the convention [l1 l2 l3 l1 l2 l3 .. etc]
        self.prevTarget = self.model.sequenceLabeller.targetEmbedding(children)

        runner = torch.arange(numAlive).repeat(self.beamSize)
        self.hiddenState = self.hiddenState.reshape((-1 , currBatchSize, self.decoderUnits))
        self.hiddenState = self.hiddenState[parents, runner]
        
    def expandOnce(self, depth, currBatchSize):
        """Get the previous target and make the forward pass"""
        logProbs = self.updateHiddenState(depth, currBatchSize)

        """Add the logProbs to the current paths to get newPathSums"""
        indices = self.updateValues(depth, currBatchSize, logProbs)

        """
            Indices represent max over arrays of size units * beam
            parent represents the idx [0, 1, .. beamSize] in the path array for each example.
            and the child represents the arg of the logit
        """
        parents, children = indices // self.outputUnits, indices % self.outputUnits

        """Extend paths using new values. No. of examples that were completely labelled is returned."""
        numFinished = self.updatePaths(children, parents)

        self.updateForNextIteration(currBatchSize, numFinished, children, parents)

    def collectPredictions(self):
        """
            Gathers the optimal queue from the deepest beam for each training example.
        """
        preds = [beam[0] for beam in self.paths]
        preds = [preds[idx] for idx in self.unsortedIndices]
        return preds

    def __call__(self, batch):
        self.rootInitialization(batch)
        for depth, currBatchSize in enumerate(self.batchArray):
            self.expandOnce(depth, currBatchSize)

        preds = self.collectPredictions()
        return preds

    def writePreds(self, output, tags, filename, referenceFile):
        """Writes the predictions of beam search into a file
            Args
            * outputs - list of lists of idx predicted as labels through beamsearch
            * tags - array to tag names, used to convert idx in output to readable tags
            * filename      - filename to write to
            * referenceFile - filename that has the input. Used for predicting 'O' for -DOCSTART-
        """
        with open(filename, 'w') as file, open(referenceFile, 'r') as refFile:
            idx = 0
            for line in refFile:
                if line.strip() == '-DOCSTART-':
                    file.write('O\n')
                else:
                    pred = [tags[tagIdx] for tagIdx in output[idx]]
                    line = " ".join(pred)
                    file.write(line+"\n")
                    idx += 1