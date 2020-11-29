import torch
from torch import nn
import torch.nn.functional as F
from models.utils import param, reverse_packed_sequence, Namespace
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence, pack_padded_sequence
import pytorch_lightning as pl

class CNNEmbedding(nn.Module):
    """Model that takes input of shape [n_words, n_chars] and returns char embedding of shape [n_words, embeddingDim]
       
        Parameters: 
            numChars(int): total chars(93)
            embeddingDim(int): use default 128
    """
    def __init__(self, numChars, embeddingDim):
        super().__init__()
        self.embedding = nn.Embedding(numChars, embeddingDim)
        
        # init zero index to a large negative value
        self.embedding.weight.data[0] = 0
        
        self.conv1d = nn.Conv1d(embeddingDim, embeddingDim, 3, 1, 1)
    
    def forward(self, sequences, mask=None): # packed with shape (pack_len w) or p w
        x = self.embedding(sequences.data) # p w u
        x = self.conv1d( x.permute(0, 2, 1) ).permute(0, 2, 1) # conv doesnt change shape; p w u
                
        # x is still p w u we have to take max across each word
        # big brain time for non zero maxing
        x, _ = torch.max(x + mask, 1) # max is across each word
        _, *args = sequences
        return PackedSequence(x, *args)
    


class TransitionGRU(nn.Module):
    def __init__(self, hidden_units):
        super().__init__()
        # save config
        self.hidden_units = hidden_units
        
        self.reset_gate     = param(hidden_units, hidden_units)
        self.update_gate    = param(hidden_units, hidden_units)
        self.candidate_gate = param(hidden_units, hidden_units)
        
    
    def forward(self, ht):
        """
        Special GRU that uses on hidden state as input
        Expected format is b u 
            r = sigma(Wr * h)
            z = sigma(Wz * h)
            n = tanh (r .* (Wn * h))

            y = (1 - z) * h + z * n
        """
        # reset gate xt -> (b u) (u u) -> (b u)
        r = torch.mm(ht, self.reset_gate)
        r = torch.sigmoid(r)
        
        # update gate
        z = torch.mm(ht, self.update_gate)
        z = torch.sigmoid(z)
        
        # candidate state
        n = r * torch.mm(ht, self.candidate_gate)
        n = torch.tanh(n)
        
        out = (1-z) * n  +  z * ht
        return out
    

class LinearEnchancedGRU(nn.Module):
    def __init__(self, input_units, hidden_units):
        super().__init__()
        # save config
        self.input_units = input_units
        self.hidden_units = hidden_units
        
        # weights for connecting input to a gate (3 gates)
        cat_units = input_units + hidden_units
        self.reset_gate  = param(cat_units, hidden_units)
        self.update_gate = param(cat_units, hidden_units)
        
        # extra params for linear enhancement
        self.linear_gate = param(cat_units, hidden_units)
        self.linear_transform = param(input_units, hidden_units)
        
        # weights to connect input to candidate activation
        self.Cx = param(input_units, hidden_units)
        self.Ch = param(hidden_units, hidden_units)
    
    def forward(self, x, hx=None):
        # expected format is b u
        if hx is None:
            hx = torch.zeros(x.size(0), self.hidden_units, dtype=x.dtype, device=x.device)
        
        hx = hx[:x.size(0)]
        concat_out = torch.cat([x, hx], dim=-1)
        
        # reset gate
        r = torch.mm(concat_out, self.reset_gate)
        r = torch.sigmoid(r)
        
        # update gate
        z = torch.mm(concat_out, self.update_gate)
        z = torch.sigmoid(z)
        
        # linear enhanced gate
        l = torch.mm(concat_out, self.linear_gate)
        l = torch.sigmoid(l)
        
        # candidate state
        n =  torch.mm(x, self.Cx) + r * torch.mm(hx, self.Ch)
        n = torch.tanh(n) + l * torch.mm(x, self.linear_transform)
        
        # linear combination
        ht = (1-z) * hx  +  z * n
        return ht

class DeepTransitionRNN(nn.Module):
    def __init__(self, inputUnits, outputUnits, transitionNumber):
        super().__init__()

        self.linearGRU = LinearEnchancedGRU(inputUnits, outputUnits)

        tgru = [TransitionGRU(outputUnits)  for _ in range(transitionNumber)]
        self.transitionGRU = nn.Sequential(*tgru)
    
    def cell_forward(self, xt, ht):
        ht = self.linearGRU(xt, ht)
        return self.transitionGRU(ht)

    def forward(self, sequence):
        inputSequence, batchSizes, sortedIndices, unsortedIndices = sequence
        start = 0
        ht = None
        outputs = []
        for batch in batchSizes:
            xt = inputSequence[start:start+batch]
            ht = self.cell_forward(xt, ht)
            outputs.append(ht)
            start = start + batch
        outputs = torch.cat(outputs)
        return PackedSequence(outputs, batchSizes, sortedIndices, unsortedIndices)


class SequenceLabelingEncoderDecoder(nn.Module):
    def __init__(self, inputUnits, encoderUnits, decoderUnits, transitionNumber):
        super().__init__()
        # save config
        self.inputUnits = inputUnits
        self.encoderUnits = encoderUnits
        self.decoderUnits = decoderUnits
        self.transitionNumber = transitionNumber
        
        # encoder is bidirectional, but decoder is unidirectional
        self.fowardEncoder   = DeepTransitionRNN(inputUnits, encoderUnits, transitionNumber)
        self.backwardEncoder = DeepTransitionRNN(inputUnits, encoderUnits, transitionNumber)
        self.decoder         = DeepTransitionRNN(2*encoderUnits, decoderUnits, transitionNumber)
        self.output          = nn.Linear(decoderUnits, decoderUnits)
        
    def enforced_logits(self, sequence, targets):
        data, batchSizes, sortedIndices, unsortedIndices = sequence
        reversedSequence = reverse_packed_sequence(sequence)
        forwardEncoded  = self.fowardEncoder(sequence)
        backwardEncoded = self.backwardEncoder(reversedSequence)
        reversedBackwardEncoded = reverse_packed_sequence(backwardEncoded)
        encoded = torch.cat([forwardEncoded.data, reversedBackwardEncoded.data], dim=-1)

        # encoded is a tensor here and not a packed sequence
        start = 0
        logits = []
        prevTarget = None
        for batch in batchSizes:
            xt = encoded[start:start+batch]
            ht = self.decoder.cell_forward(xt, prevTarget)
            logit = self.output(ht)
            logits.append(logit)
            
            # update prevTarget after processing current timestep
            prevTarget = targets.data[start:start+batch]
            start = start + batch
        logits = torch.cat(logits)
        return logits


"""
    Glove and char embeddings to global embeddings
"""
class GlobalContextualEncoder(nn.Module):
    def __init__(self, numChars, charEmbedding, numWords, wordEmbedding, outputUnits, transitionNumber):
        super().__init__()
        self.cnn = CNNEmbedding(numChars, charEmbedding)
        self.glove = nn.Embedding(numWords, wordEmbedding)
        self.glove.weight.requires_grad=False
        
        encoderInputUnits = wordEmbedding + charEmbedding
        self.forwardEncoder  = DeepTransitionRNN(encoderInputUnits, outputUnits, transitionNumber)
        self.backwardEncoder = DeepTransitionRNN(encoderInputUnits, outputUnits, transitionNumber)
        
    def forward(self, words, chars, charMask):
        _, *args = words
        
        w = self.glove(words.data)
        c = self.cnn(chars, charMask)
        
        # word and char concat, pass through encoder and we get directional global context
        wc = torch.cat([w, c.data], dim=-1)
        forwardInput  = PackedSequence( wc, *args )
        forwardG  = self.forwardEncoder(forwardInput)
        
        backwardInput = reverse_packed_sequence(forwardInput)      
        backwardG = self.backwardEncoder(backwardInput)
        backwardG = reverse_packed_sequence(backwardG)
        
        nonDirectionalG = torch.cat([forwardG.data, backwardG.data], dim=-1)
        
        # mean pooling is done by padding with zeros, taking timewise sum and dividing by lengths
        nonDirectionalG = PackedSequence(nonDirectionalG, *args)
        nonDirectionalG, lens = pad_packed_sequence(nonDirectionalG)

        # here we see what
        device = next(self.parameters()).device
        lens = torch.unsqueeze(torch.unsqueeze(lens, -1), 0).to(device)

        nonDirectionalG_sum = nonDirectionalG.sum(dim=0, keepdim=True)
        g = nonDirectionalG_sum / lens
        
        # need to broadcast g and concat with wc
        new_shape = [nonDirectionalG.data.shape[0] // g.shape[0]] + [-1] * (len(g.shape) - 1)
        g = pack_padded_sequence(g.expand(*new_shape), lens[0,:, 0], enforce_sorted=False)
        
        wcg = torch.cat([g.data, wc], dim=-1)
        wcg = PackedSequence(wcg, *args)
        return wcg

"""
    Model from the paper
"""
class GlobalContextualDeepTransition(pl.LightningModule):
    def __init__(self, numChars, charEmbedding, numWords,
                     wordEmbedding, contextOutputUnits, contextTransitionNumber,
                        encoderUnits, decoderUnits, transitionNumber):
        super().__init__()
        self.contextEncoder = GlobalContextualEncoder(numChars, charEmbedding, numWords,
                                                          wordEmbedding, contextOutputUnits, contextTransitionNumber)
        self.labellerInput = wordEmbedding + charEmbedding + 2 * contextOutputUnits # units in g
        self.sequenceLabeller = SequenceLabelingEncoderDecoder(self.labellerInput, encoderUnits, decoderUnits, transitionNumber)
    
    def time_series_cross_entropy(self, logits, targets):
        loss = -targets * F.log_softmax(logits, dim=1)
        return loss.sum(dim=1).mean()
    
    def training_step(self, batch, batch_idx):
        words, chars, charMask, targets = batch
        
        # compute the global representation and concat with word and char representations
        wcg = self.contextEncoder(words, chars, charMask)
        
        # encode concaatentated input and decode logits using sequence labeller
        logits = self.sequenceLabeller.enforced_logits(wcg, targets)
        
        # compute loss using averaging loss across different timesteps
        loss = self.time_series_cross_entropy(logits.data, targets.data)
        
        return loss