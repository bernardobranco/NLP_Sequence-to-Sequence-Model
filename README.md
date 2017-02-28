# SNLP-CW3
Sequence-to-sequence model using LSTMs to predict order of sentences in a story (5 sentences per story).

Brief explanation of model:

Fristly, the pipeline function tokenizes the sentences of the story, then the GloVe embeddings are used to find a vector representation for each word in the sentences.
For the submssion file PCA was run to reduce the 300 dimensional vector representations to 50 (value obtained through training).

Each word of each sentence is then fed into an LSTM (I implemented this by running an LSTM on all of the first sentences, then all of the second, and so forth, where each LSTM reuses the parameters of the LSTM before). 
By doing this I am able obtain a vector representation for each sentence which is based on the word embeddings of the respective sentence.

Afterwards, the vectors of each sentence are fed into a sequence-to-sequence model. The idea behind this is that the encoder LSTM is able to learn the general context, “state”, of each story. 
Then, the decoder has the role of interpreting that story state at each timestep and choose the sentence that most likely fits at that position in the sequence.

At training time the sentences are fed in the correct order. The goal with this is for the model to learn how correct orders of sentences looks like as they’re fed into the decoder. 
The scoring function consists of multiplying the hidden state at each timpstep with a hidden-by-hidden-size weight matrix and then adding a bias (a hidden size vector). 
Then, the vector representations for each sentence (of each story in the batch) is multiplied to this result. A vector of 5 entries is thus obtained at each timestep for each story in the batch. 
After running a softmax on these vectors I obtain the probabilities of each sentence of the story to occur at that timestep. 
Taking the argmax of this vector gives me the sentence that the model deems most likely to occur at that timestep. 
Running the softmax cross-entropy on these logits defines the loss of the model.

At test time the process is a bit different since I no longer have the correct order of the sentences. 
Consequently, I coded a loop function (given as an argument to the tied_rnn_seq2seq function) where at each timestep the scoring function described above is run.
From these results it chooses the sentence for each story which has the highest probability of occurring at that point in time. 
The respective vectors of these sentences (one per story in the batch) are then used as the input for the next timestep of the decoder LSTM (this corresponds to a dynamic ordering of the inputs as the LSTM is being looped).
In order for this to work an initial “GO” symbol is fed (which consists of a batch_size*hidden_size matrix) to the sequence-to-sequence model.
