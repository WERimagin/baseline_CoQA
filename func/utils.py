import random
import numpy as np
import torch
from tqdm import tqdm
from torch.autograd import Variable
from nltk.tokenize import word_tokenize,sent_tokenize


#使わない
#文のリスト(入力と出力の二つ)を取って、id化した物を返す
#ここでは非numpy,かつサイズがバラバラ->make_vectorでバッチごとに　揃える
class Word2Id:
    def __init__(self,enc_sentences,dec_sentences):
        self.words=["<pad>"]#素性として使うwordのリスト
        self.word2id={}#word->idの変換辞書
        self.id2word={}#id->wordの変換辞書
        self.enc_sentences=enc_sentences
        self.dec_sentences=dec_sentences
        self.vocab_size=0

    def __call__(self):
        #wordsの作成
        print(self.dec_sentences[0])
        sentences=self.enc_sentences+self.dec_sentences
        for sentence in tqdm(sentences):
            for word in sentence:
                if word not in self.words:
                    self.words.append(word)
        self.vocab_size=len(self.words)
        #word2idの作成
        #id2wordの作成
        for i,word in enumerate(self.words):
            self.word2id[word]=i
            self.id2word[i]=word
        #sentence->ids
        enc_id_sentences=[]
        dec_id_sentences=[]
        for sentence in self.enc_sentences:
            sentence=[self.word2id[word] for word in sentence]
            enc_id_sentences.append(sentence)
        for sentence in self.dec_sentences:
            sentence=[self.word2id[word] for word in sentence]
            dec_id_sentences.append(sentence)
        return enc_id_sentences,dec_id_sentences

class DataLoader:#使うデータをまとめてシャッフル、batch単位にして返す
    def __init__(self,data_size,batch_size,shuffle=True):
        self.data_size=data_size
        self.batch_size=batch_size
        self.data=list(range(self.data_size))
        self.shuffle=shuffle
    def __call__(self):
        if self.shuffle:
            random.shuffle(self.data)
        batches=[]
        batch=[]
        for i in range(self.data_size):
            batch.append(self.data[i])
            if len(batch)==self.batch_size:
                batches.append(batch)
                batch=[]
        if len(batch)>0:
            batches.append(batch)
        return batches

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def make_vec(sentences):
    maxsize=max([len(sentence) for sentence in sentences])
    sentences_ex=[]
    for sentence in sentences:
        sentences_ex.append(sentence+[0]*(maxsize-len(sentence)))
    return to_var(torch.from_numpy(np.array(sentences_ex,dtype="long")))

def make_vec_c(sentences):
    sent_maxsize=max([len(sentence) for sentence in sentences])
    char_maxsize=max([len(word) for sentence in sentences for word in sentence])
    sentence_ex=np.zeros((len(sentences),sent_maxsize,char_maxsize),dtype="long")
    for i,sentence in enumerate(sentences):
        for j,word in enumerate(sentence):
            for k,char in enumerate(word):
                sentence_ex[i,j,k]=char
    return to_var(torch.from_numpy(sentence_ex))

#sentenceを受け取り、tokenizeして返す
def tokenize(sent):
    return [token.replace('``','"').replace("''",'"') for token in word_tokenize(sent)]

#context_textを文分割して、answer_start~answer_end(char単位)のスパンが含まれる文を返す
#やってることはc2iと多分同じアルゴリズム
def answer_find(context_text,answer_start,answer_end):
    if answer_start==-1: return ""
    context=sent_tokenize(context_text)
    sent_start_id=-1
    sent_end_id=-1
    start_id_list=[context_text.find(sent) for sent in context]
    end_id_list=[start_id_list[i+1] if i+1!=len(context) else len(context_text) for i,sent in enumerate(context)]
    for i,sent in enumerate(context):
        start_id=start_id_list[i]
        end_id=end_id_list[i]
        if start_id<=answer_start and answer_start<=end_id:
            sent_start_id=i
        if start_id<=answer_end and answer_end<=end_id:
            sent_end_id=i

    if sent_start_id==-1 or sent_end_id==-1:
        print("error")
        exit(-1)
    answer_sent=" ".join(context[sent_start_id:sent_end_id+1])

    return answer_sent
