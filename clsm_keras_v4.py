import re
import numpy as np
import random
import matplotlib.pyplot as plt
from keras import backend
from keras.layers import Input
from keras.layers.core import Dense, Lambda, Reshape
from keras.layers.convolutional import Convolution1D
from keras.layers.merge import concatenate, dot
from keras.models import Model
from keras.preprocessing.text import Tokenizer
import tensorflow.contrib.learn as learn
VocabularyProcessor = learn.preprocessing.VocabularyProcessor
Q_len = 8
D_len = 150
vp1 = VocabularyProcessor(Q_len)
vp2 = VocabularyProcessor(D_len)
N_files = 21
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
[len(qry_l[i].split()) for i in range(N_files)]
[len(pos_doc_l[i].split()) for i in range(N_files)]

i = 20
N_qry = len(qry_l)
rnd_rng = list(range(0,i))+list(range(i+1,N_qry))
rnd_rng
random.choice(rnd_rng)
neg_docs_l = [[pos_doc_l[j] for j in random.sample(list(range(0,i))+list(range(i+1,N_qry)), J) ] for i in range(len(qry_l)) ]


# %%
vp1.fit(qry_l+pos_doc_l)
vp2.fit(qry_l+pos_doc_l)
list(vp2.fit_transform([pos_doc_l[20]]))
list(vp2.reverse([[1,2,200]]))
# %% shape corpus
sample_size = 21
l_Qs = []
pos_l_Ds = []
# J = 8

LETTER_GRAM_SIZE = 3 # See section 3.2.
WINDOW_SIZE = 3 # See section 3.2.
TOTAL_LETTER_GRAMS = int(3 * 1e4) # Determined from data. See section 3.2.
WORD_DEPTH = WINDOW_SIZE * TOTAL_LETTER_GRAMS # See equation (1).
K = 300 # Dimensionality of the max-pooling layer. See section 3.4.
L = 128 # Dimensionality of latent semantic space. See section 3.5.
# J = 4 # Number of random unclicked documents serving as negative examples for a query. See section 4.
FILTER_LENGTH = 1 # We only consider one time step for convolutions.

vocab_size = 1600
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

# %% setup model
query = Input(shape = (None, WORD_DEPTH))
pos_doc = Input(shape = (None, WORD_DEPTH))
neg_docs = [Input(shape = (None, WORD_DEPTH)) for j in range(J)]

query_conv = Convolution1D(K, FILTER_LENGTH, padding = "same", input_shape = (None, WORD_DEPTH), activation = "tanh")(query) # See equation (2).

query_max = Lambda(lambda x: backend.max(x, axis = 1), output_shape = (K, ))(query_conv) # See section 3.4.

query_sem = Dense(L, activation = "tanh", input_dim = K)(query_max) # See section 3.5.

doc_conv = Convolution1D(K, FILTER_LENGTH, padding = "same", input_shape = (None, WORD_DEPTH), activation = "tanh")
doc_max = Lambda(lambda x: backend.max(x, axis = 1), output_shape = (K, ))
doc_sem = Dense(L, activation = "tanh", input_dim = K)

pos_doc_conv = doc_conv(pos_doc)
neg_doc_convs = [doc_conv(neg_doc) for neg_doc in neg_docs]

pos_doc_max = doc_max(pos_doc_conv)
neg_doc_maxes = [doc_max(neg_doc_conv) for neg_doc_conv in neg_doc_convs]

pos_doc_sem = doc_sem(pos_doc_max)
neg_doc_sems = [doc_sem(neg_doc_max) for neg_doc_max in neg_doc_maxes]

R_Q_D_p = dot([query_sem, pos_doc_sem], axes = 1, normalize = True) # See equation (4).
R_Q_D_ns = [dot([query_sem, neg_doc_sem], axes = 1, normalize = True) for neg_doc_sem in neg_doc_sems] # See equation (4).

concat_Rs = concatenate([R_Q_D_p] + R_Q_D_ns)
concat_Rs = Reshape((J + 1, 1))(concat_Rs)

weight = np.array([1]).reshape(1, 1, 1)
with_gamma = Convolution1D(1, 1, padding = "same", input_shape = (J + 1, 1), activation = "linear", use_bias = False, weights = [weight])(concat_Rs) # See equation (5).
with_gamma = Reshape((J + 1, ))(with_gamma)

prob = Lambda(lambda x: backend.softmax(x), output_shape = (J + 1, ))(with_gamma) # See equation (5).

model = Model(inputs = [query, pos_doc] + neg_docs, outputs = prob)
model.compile(optimizer = "adadelta", loss = "categorical_crossentropy")

# %% train model
if BATCH:
    y = np.zeros((sample_size, J + 1))
    y[:, 0] = 1

    l_Qs = np.array(l_Qs)
    pos_l_Ds = np.array(pos_l_Ds)
    for j in range(J):
        neg_l_Ds[j] = np.array(neg_l_Ds[j])

    history = model.fit([l_Qs, pos_l_Ds] + [neg_l_Ds[j] for j in range(J)], y, epochs = 1, verbose = 0)
else:
    y = np.zeros(J + 1).reshape(1, J + 1)
    y[0, 0] = 1

    for i in range(sample_size):
        history = model.fit([l_Qs[i], pos_l_Ds[i]] + [neg_l_Ds[j][i] for j in range(J)], y, epochs = 1, verbose = 0)

# %% test model
get_R_Q_D_p = backend.function([query, pos_doc], [R_Q_D_p])
if BATCH:
    get_R_Q_D_p([l_Qs, pos_l_Ds])
else:
    get_R_Q_D_p([l_Qs[0], pos_l_Ds[0]])

# # A slightly more complex function. Notice that both neg_docs and the output are
# # lists.
# get_R_Q_D_ns = backend.function([query] + neg_docs, R_Q_D_ns)
# if BATCH:
#     get_R_Q_D_ns([l_Qs] + [neg_l_Ds[j] for j in range(J)])
# else:
#     get_R_Q_D_ns([l_Qs[0]] + neg_l_Ds[0])

# %% show test
# pos_l_Ds.shape
# l_Qs[i].shape
# l_Qs.shape
np_out = np.zeros((sample_size, J+1))
np_out[:, 0] = get_R_Q_D_p([l_Qs, pos_l_Ds])[0].T
for i in range(J):
    np_out[:, i+1] = get_R_Q_D_p([l_Qs, neg_l_Ds[i]])[0].T
print(np_out)
%matplotlib inline
plt.plot(np_out)

# len(neg_l_Ds)
# neg_l_Ds[:,0,:,:].shape
