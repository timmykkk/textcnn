import os
import pprint
from keras.models import Sequential
from keras import backend as K
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout, Input, concatenate
from keras.preprocessing.text import Tokenizer
from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.preprocessing import sequence
import numpy as np
import re


#去除html标签
def remove_tags(text):
    re_tag=re.compile(r'<[^>+>]')
    return re_tag.sub('',text)

def read_files(filetype):
    #filetype: train or test
    #标签1表示正面，0表示负面
    all_labels=[1]*12500+[0]*12500
    all_texts = []
    file_list= []
    path = r'./Imdb/'
    #读取所有正例文本名
    pos_path=path+filetype+'/pos/'
    for file in os.listdir(pos_path):
        file_list.append(pos_path+file)
    print("posdone")
    #读取所有负例文本名
    neg_path=path+filetype+'/neg/'
    for file in os.listdir(neg_path):
        file_list.append(neg_path+file)
    print("negdone")
    #将所有文本内容加到all_texts
    #print(len(file_list))
    #cnt=0
    for file_name in file_list:
        #cnt+=1
        #print(cnt)
        with open(file_name, encoding='utf-8') as f:
            all_texts.append(remove_tags(" ".join(f.readlines())))
    #print(all_texts)
    return all_texts, all_labels

def text_cnn(maxlen=150, max_features=2000, embed_size=300):
    #输入 25000*150
    comment_seq=Input(shape=[maxlen],name='x_seq')
    #print("comment_seq.shape")
    #print(K.int_shape(comment_seq))  25000*150
    #embedding层，embed_size为词向量的长度
    emb_comment=Embedding(max_features,embed_size)(comment_seq)
    #print("emb_comment.shape")
    #print(K.int_shape(emb_comment))  25000*150*32

    #是对每个句子进行卷积
    #卷积层
    convs=[]
    filter_sizes=[2,3,4,5]
    for fsz in filter_sizes:
        #100个卷积核，卷积核高度是2,3,4,5
        l_conv=Conv1D(filters=100, kernel_size=fsz,activation='relu')(emb_comment)
        l_pool=MaxPooling1D(maxlen-fsz+1)(l_conv) #25000*1*100
        l_pool=Flatten()(l_pool) #25000*100
        convs.append(l_pool)

    merge = concatenate(convs,axis=1) #横向拼接 #25000*400
    #print("merge:")
    #print(K.int_shape(merge))
    out=Dropout(0.2)(merge) #dropout #25000*400
    #print("out:")
    #print(K.int_shape(out))
    output=Dense(32,activation='relu')(out) #全连接, 降维成32
    print("output:")
    print(K.int_shape(output))
    output=Dense(units=1,activation='sigmoid')(output) #全连接，计算概率
    print("output:")
    print(output)
    model=Model([comment_seq],output)
    model.compile(loss="binary_crossentropy",optimizer="adam",metrics=['accuracy'])
    return model

if __name__=='__main__':
    #print(1)
    train_texts,train_labels=read_files("train");
    #print(2)
    test_texts, test_labels=read_files("test")
    #print(3)
    #将输入的文本中的每个词编号，编号是根据词频的，词频越大，编号越小
    tokenizer=Tokenizer(num_words=2000)
    tokenizer.fit_on_texts(train_texts)
    #pprint.pprint(tokenizer.word_index)

    #把每句影评转为数字列表
    x_train_seq=tokenizer.texts_to_sequences(train_texts)
    #print(len(x_train_seq))
    x_test_seq=tokenizer.texts_to_sequences(test_texts)

    #使数字列表定长,即每句话有定长的词，（不够就补0）
    x_train=sequence.pad_sequences(x_train_seq,maxlen=150)
    #print(len(x_train)) #25000*150
    x_test=sequence.pad_sequences(x_test_seq,maxlen=150)
    #print(len(x_test))  #25000*150
    y_train=np.array(train_labels)
    y_test=np.array(test_labels)


    model=text_cnn()
    batch_size=128
    epochs=3
    model.fit(x_train, y_train,
              validation_split=0.1,
              batch_size=batch_size,
              epochs=epochs,
              shuffle=True)

    scores = model.evaluate(x_test,y_test)
    print('test_loss:%f,accuracy:%f'%(scores[0],scores[1]))
