import re
import numpy as np
import random
from keras import backend
from keras.layers import Input
from keras.layers.core import Dense, Lambda, Reshape
from keras.layers.convolutional import Convolution1D
from keras.layers.merge import concatenate, dot
from keras.models import Model
from keras.preprocessing.text import Tokenizer
import tensorflow.contrib.learn as learn
VocabularyProcessor = learn.preprocessing.VocabularyProcessor
vp1 = VocabularyProcessor(10)
vp2 = VocabularyProcessor(200)

# %%

spc_sym = ['article','<d>','<p>','<s>','</d>','</p>','</s>','abstract','=']
qry_l = []
pos_doc_l = []
J = 4
re_splt = re.compile(' |=|-') # 因为keras的tokenizer把abc-def这种处理成一个单词了, 这里为了简化
with open('text_data','r') as f:
    all_l = f.readlines()
    for i,line in enumerate(all_l):
        if len(line) > 10:
            one_l = line.split('\t')
            qry_l.append(' '.join(filter(lambda x: x.lower() not in spc_sym , re_splt.split(one_l[1]) )))
            pos_doc_l.append(' '.join(filter(lambda x: x.lower() not in spc_sym , re_splt.split(one_l[0]) )))

print(len(qry_l), qry_l)

i = 20
N_qry = len(qry_l)
rnd_rng = list(range(0,i))+list(range(i+1,N_qry))
rnd_rng
random.choice(rnd_rng)
neg_docs_l = [[pos_doc_l[j] for j in random.sample(list(range(0,i))+list(range(i+1,N_qry)), J) ] for i in range(len(qry_l)) ]


# %%
vp2.fit(qry_l+pos_doc_l)
vp1.fit(qry_l+pos_doc_l)
list(vp2.fit_transform([pos_doc_l[20]]))
list(vp2.reverse([[1,2,200]]))
# %% shape corpus
sample_size = 10
l_Qs = []
pos_l_Ds = []
J = 4

LETTER_GRAM_SIZE = 3 # See section 3.2.
WINDOW_SIZE = 3 # See section 3.2.
TOTAL_LETTER_GRAMS = int(3 * 1e4) # Determined from data. See section 3.2.
WORD_DEPTH = WINDOW_SIZE * TOTAL_LETTER_GRAMS # See equation (1).
K = 300 # Dimensionality of the max-pooling layer. See section 3.4.
L = 128 # Dimensionality of latent semantic space. See section 3.5.
J = 4 # Number of random unclicked documents serving as negative examples for a query. See section 4.
FILTER_LENGTH = 1 # We only consider one time step for convolutions.

vocab_size = 2000
n_gram = 3

# just a bag-of-word simplify
# use simple n_gram to replace for letter-n-gram + word-n-gram
def convert2input(np_array, size):
    len_na = np_array.shape[0]
    out = np.zeros((np_array.shape[0]-2,size))
    # print(out.shape)
    for i in range(1, len_na-1):
        temp = np.zeros((size,))
        temp[np_array[i-1]] += 1
        temp[np_array[i]] += 1
        temp[np_array[i+1]] += 1
        # print(temp.shape)
        out[i-1, :] = temp
    return out
list_of_np = list(vp2.fit_transform([pos_doc_l[20]]))
out = convert2input(list_of_np[0], vocab_size)
doc1 = [convert2input(list(vp2.fit_transform([pos_doc_l[i]]))[0], vocab_size) for i in range(sample_size) ]
qry1 = [convert2input(list(vp1.fit_transform([qry_l[i]]))[0], vocab_size) for i in range(sample_size) ]
qry1[0].shape
# np.zeros((3,))
# asd1 = np.array([[1,2,3],[3,4,5]])
# asd1[1,:] = np.array([7,8,9])
# asd1
# np.array([[1,2,3]])[0,1]
# np.array([1,2,3]).shape
# asd = np.array([1,2,3])
# asd[1] = 5
# asd
# np.array([[1],[2],[3]])[0,0]

# %%
# Variable length input must be handled differently from padded input.
BATCH = True

(query_len, doc_len) = (qry1[0].shape[0], doc1[0].shape[0])#(10, 200)

WORD_DEPTH = vocab_size

for i in range(sample_size):

    if BATCH:
        # l_Q = np.random.rand(query_len, WORD_DEPTH)
        l_Qs.append(qry1[i])

        # l_D = np.random.rand(doc_len, WORD_DEPTH)
        pos_l_Ds.append(doc1[i])
    else:
        query_len = np.random.randint(1, 10)
        l_Q = np.random.rand(1, query_len, WORD_DEPTH)
        l_Qs.append(l_Q)

        doc_len = np.random.randint(50, 500)
        l_D = np.random.rand(1, doc_len, WORD_DEPTH)
        pos_l_Ds.append(l_D)

neg_l_Ds = [[] for j in range(J)]
for i in range(sample_size):
    possibilities = list(range(sample_size))
    possibilities.remove(i)
    negatives = np.random.choice(possibilities, J, replace = False)
    for j in range(J):
        negative = negatives[j]
        neg_l_Ds[j].append(pos_l_Ds[negative])

# %%




# %% check query doc shape
np.array([1,2,3]).shape[0]
l_Q.shape
l_Q
l_D
l_D.shape
