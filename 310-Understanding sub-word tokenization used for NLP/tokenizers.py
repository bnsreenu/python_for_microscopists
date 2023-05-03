# https://youtu.be/dBk98xqEtoA
"""
This code demonstrates how to use the BERT tokenizer from the transformers library 
and SentencePieceBPETokenizer & ByteLevelBPETokenizer from the tokenizers library 
to tokenize a sentence.

!pip install transformers
!pip install tokenizers
"""



from transformers import BertTokenizer

bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
input_sentence = "Digitalsreeni is a great resource for python "

bert_tokens = bert_tokenizer.tokenize(input_sentence)
print(bert_tokens)

# ['digital', '##sr', '##een', '##i', 'is', 'a', 'great', 'resource', 'for', 'python']
#Digitalsreeni is not present in the tokenizer vocabulary, so it split it into
#"Digital" "sr" "een" and "i". All other words are its vocabulary.
# Note that "##" means the token should be attached to the previous one


#############################################################################

"""
Both ByteLevelBPETokenizer and SentencePieceBPETokenizer are tokenizers used 
for subword tokenization, but they use different algorithms to learn the 
vocabulary and perform tokenization.

ByteLevelBPETokenizer is a tokenizer from the Hugging Face tokenizers library 
that learns byte-level BPE (Byte Pair Encoding) subwords. It starts by splitting 
each input text into bytes, and then learns a vocabulary of byte-level subwords 
using the BPE algorithm. This tokenizer is particularly useful for languages 
with non-Latin scripts, where a character-level tokenizer may not work well.

On the other hand, SentencePieceBPETokenizer is a tokenizer from the SentencePiece 
library that learns subwords using a unigram language model. It first tokenizes 
the input text into sentences, and then trains a unigram language model on the 
resulting sentence corpus to learn a vocabulary of subwords. This tokenizer can 
handle a wide range of languages and text types, and can learn both character-level 
and word-level subwords.

In terms of usage, both tokenizers are initialized and trained in a similar way.


"""


from tokenizers import SentencePieceBPETokenizer
BPE_sentence_tokenizer = SentencePieceBPETokenizer()
#input_sentence = "Digitalsreeni is a great resource for python "
# Define the input sentence
input_sentence = "Digitalsreeni is a great resource for python and all my friends love Digitalsreeni videos on YouTube. As you know YouTube has many channels but I like Digitalsreeni channel the most"


#Train on vocab from a file
BPE_sentence_tokenizer.train('some_text.txt', vocab_size=1000)
BPE_sentence_tokens = BPE_sentence_tokenizer.encode(input_sentence)
print(BPE_sentence_tokens.tokens)
# To decode with SentencePiece all tokens can just be concatenated and "▁" is replaced by a space.




from tokenizers import ByteLevelBPETokenizer
byte_level_BPE_tokenizer = ByteLevelBPETokenizer()
input_sentence = "Digitalsreeni is a great resource for python and all my friends love Digitalsreeni videos on YouTube. As you know YouTube has many channels but I like Digitalsreeni channel the most"


#Train on vocab from a file
byte_level_BPE_tokenizer.train('some_text.txt', vocab_size=1000)
byte_level_BPE_tokens = byte_level_BPE_tokenizer.encode(input_sentence)
print(byte_level_BPE_tokens.tokens)
# The Ġ symbol is used to represent the beginning of a word, so when you see Ġis, 
# it means the word starts with "is"


    
"""
#The Ġ symbol is used to represent the beginning of a word, so when you see Ġi, 
# it means the word starts with the letter "i"


In the encoded sentence, "Digitalsreeni" and "channel" are encoded as single 
tokens because they appear frequently in the input sentence. When the Byte Pair 
Encoding (BPE) algorithm is applied to the input sentence, it iteratively merges 
the most frequent character sequences into new tokens until it reaches the desired 
vocabulary size.

In this case, the BPE algorithm has learned that "Digitalsreeni" and "channel" 
are frequent character sequences in the input sentence and has merged them into 
single tokens. The Ġ symbol is used to represent the beginning of each word, 
so when you see ĠDigitalsreeni, it means the word "Digitalsreeni" starts at that 
position.

Note that the BPE algorithm does not always merge frequent character sequences 
into single tokens. The decision to merge or not to merge depends on the frequency 
and context of the character sequences in the input data. In some cases, it may be 
more appropriate to split a word into multiple subword units rather than merging 
it into a single token.

"""

#############################################################################
"""
#Train versus train_from_iterator

The train method takes a list of strings as input, where each string represents 
a document or a sequence of tokens. It tokenizes each document and uses the 
resulting tokens to learn the vocabulary of the tokenizer. This method is 
suitable when the entire dataset can be loaded into memory at once.

The train_from_iterator method, on the other hand, takes an iterable as input, 
where each element of the iterable represents a document or a sequence of tokens. 
It iterates through the elements of the iterable and tokenizes each document, 
without loading the entire dataset into memory at once. This method is useful 
when the dataset is too large to fit into memory at once, or when the documents 
are generated on-the-fly and cannot be pre-loaded into a list.

"""

# Initialize a tokenizer object
BPE_sentence_tokenizer_1 = SentencePieceBPETokenizer()

# Train the tokenizer using the `train` method
documents = ["some_text.txt", "some_text2.txt", "some_text3.txt"]
BPE_sentence_tokenizer_1.train(documents)




BPE_sentence_tokenizer_2 = SentencePieceBPETokenizer()
# Train the tokenizer using the `train_from_iterator` method
def document_iterator():
    yield "some_text.txt"
    yield "some_text2.txt"
    yield "some_text3.txt"
BPE_sentence_tokenizer_2.train_from_iterator(document_iterator())




#In both cases, we get the same training but the second approach is better
#if all training data cannot be fit in memory at once. 

################################################################


    
