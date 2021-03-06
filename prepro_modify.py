#SQuADのデータ処理
#必要条件:CoreNLP
#Tools/core...で
#java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000

import os
import sys
sys.path.append("../")
import json
import gzip
import pandas as pd
import numpy as np
from tqdm import tqdm
from nltk.tokenize import word_tokenize,sent_tokenize
import pickle
import collections

from func.tf_idf import tf_idf
from func.utils import tokenize,answer_find

def head_find(tgt):
    q_head=["what","how","who","when","which","where","why","whose","whom","is","are","was","were","do","did","does"]
    tgt_tokens=word_tokenize(tgt)
    true_head="<none>"
    for h in q_head:
        if h in tgt_tokens:
            true_head=h
            break
    return true_head

def modify(sentence,question_interro):
    #head=head_find(question)
    """
    if answer in sentence:
        sentence=sentence.replace(answer," ans_rep_tag ")
    """
    #sentence=" ".join([sentence,"ans_pos_tag",answer,"interro_tag",question_interro])
    sentence=" ".join([sentence,"interro_tag",question_interro])
    return sentence

def modify_history(history,now):
    #head=head_find(question)
    """
    if answer in sentence:
        sentence=sentence.replace(answer," ans_rep_tag ")
    """
    #sentence=" ".join([sentence,"ans_pos_tag",answer,"interro_tag",question_interro])
    sentence=" ".join([history,"history_append_tag",now])
    return sentence

def history_maker(neg_interro,question_interro):
    interro_list=["what","where","who","why","which","whom","how",""]
    while True:
        index=random.randrange(len(interro_list))
        if interro_list[index]!=question_interro.split()[0]:
            break
    question=interro_list[index]+" "+neg_interro
    return question


def c2wpointer(context_text,context,answer_start,answer_end):#answer_start,endをchara単位からword単位へ変換
    #nltk.tokenizeを使って分割
    #ダブルクオテーションがなぜか変化するので処理
    token_id={}
    cur_id=0
    for i,token in enumerate(context):
        start=context_text.find(token,cur_id)
        token_id[i]=(start,start+len(token))
        cur_id=start+len(token)
    for i in range(len(token_id)):
        if token_id[i][0]<=answer_start and answer_start<=token_id[i][1]:
            answer_start_w=i
            break
    for i in range(len(token_id)):
        if token_id[i][0]<=answer_end and answer_end<=token_id[i][1]:
            answer_end_w=i
            break
    return answer_start_w,answer_end_w

def data_process(input_path,dict_path,modify_path,train=False):
    with open(input_path,"r") as f:
        data=json.load(f)
    with open(dict_path,"r") as f:
        corenlp_data=json.load(f)

    modify_data=[]
    with open(modify_path,"r") as f:
        for line in f:
            modify_data.append(line.rstrip())

    contexts=[]
    questions=[]
    answer_starts=[]
    answer_ends=[]
    answer_texts=[]
    answers=[]
    sentences=[]
    ids=[]

    count=0
    modify_count=0
    modify=True

    new_data={"version":"1.0",
                "data":[]}

    for paragraph in tqdm(data["data"]):
        context_text=paragraph["story"].lower()
        question_history=[]
        new_paragraph={"source": paragraph["source"],
                        "id": paragraph["id"],
                        "filename":paragraph["filename"],
                        "story":paragraph["story"],
                        "questions":[],
                        "answers":[]}
        if train==False:
            new_paragraph["additional_answers"]=paragraph["additional_answers"]

        for i in range(len(paragraph["questions"])):
            question_dict=paragraph["questions"][i]
            answer_dict=paragraph["answers"][i]
            question_text=question_dict["input_text"].lower()
            answer_text=answer_dict["input_text"].lower()
            question_history.append(question_text)

            span_start=answer_dict["span_start"]
            span_end=answer_dict["span_end"]
            span_text=answer_dict["span_text"]
            turn_id=question_dict["turn_id"]

            interro=corenlp_data[count]["interro"]
            vb_check=corenlp_data[count]["vb_check"]
            count+=1

            question_text=" ".join(tokenize(question_text.lower()))

            if interro!="" and span_start!=-1:
                #元のままの質問文
                question_dict["modify_question"]=False
                if interro!="" and vb_check==False:
                    question_dict["interro_question"]=True
                else:
                    question_dict["interro_question"]=False
                question_dict["input_text"]=question_text
                new_paragraph["questions"].append(question_dict)
                new_paragraph["answers"].append(answer_dict)

                if modify or train==False:
                    #修正した質問文
                    new_question_dict=question_dict.copy()
                    new_question_dict["turn_id"]=turn_id+len(paragraph["questions"])
                    new_question_dict["modify_question"]=True
                    new_question_dict["input_text"]=modify_data[modify_count]
                    modify_count+=1
                    new_answer_dict=answer_dict.copy()
                    new_answer_dict["turn_id"]=turn_id+len(paragraph["questions"])
                    new_paragraph["questions"].append(new_question_dict)
                    new_paragraph["answers"].append(new_answer_dict)

            """
            if d["vb_check"]==False and d["interro"]!="" and span_start!=-1:
                question_dict["interro_question"]=True
                if interro_modify:
                    question_dict["input_text"]=modify_data[modify_count]
                modify_count+=1
            else:
                question_dict["interro_question"]=False
            new_paragraph["questions"].append(question_dict)
            new_paragraph["answers"].append(answer_dict)
            """
        new_data["data"].append(new_paragraph)

    print("count:{}".format(count))
    print("modify_count:{}".format(modify_count))

    if modify:
        type="modify-sentence"
        if train:
            with open("data/coqa-train-{}.json".format(type),"w")as f:
                json.dump(new_data,f,indent=4)
        else:
            with open("data/coqa-dev-{}.json".format(type),"w")as f:
                json.dump(new_data,f,indent=4)
    else:
        type="sentence"
        if train:
            with open("data/coqa-train-{}.json".format(type),"w")as f:
                json.dump(new_data,f,indent=4)
        else:
            with open("data/coqa-dev-{}.json".format(type),"w")as f:
                json.dump(new_data,f,indent=4)


data_process(input_path="data/coqa-dev-v1.0.json",
            dict_path="data/coqa-interro-dev.json",
            modify_path="data/coqa-pred-dev-sentence.txt",
            train=False
            )

data_process(input_path="data/coqa-train-v1.0.json",
            dict_path="data/coqa-interro-train.json",
            modify_path="data/coqa-pred-train-sentence.txt",
            train=True
            )
