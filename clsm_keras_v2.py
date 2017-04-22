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
vp = VocabularyProcessor(200)

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

tokenizer = Tokenizer(num_words=None, lower=True, split=' ')

tokenizer.fit_on_texts(qry_l+pos_doc_l)

# fit的时候一整句话扔进去就行, 但是计算的时候就需要把词全都分开
for i in range(N_qry):
    asd1 = tokenizer.texts_to_sequences(qry_l[i].strip('.| ').split())
    print(len(qry_l[i].strip('.| ').split()), len(asd1),asd1)
    print(len(qry_l[i].strip('.| ').split()), len(asd1),qry_l[i].strip('.| ').split())

np.random.choice([1,2,3,4,5],2)
random.sample([1,2,3,4,5],2)

np.random.randint(1,10)
np.random.rand(3,5)
np.random.rand(1,3,5)
# %%
qry_l
haha = ['and ukraine would-be determined by their respective constitutions']
vp.fit(qry_l+pos_doc_l)
list(vp.fit_transform(haha))
