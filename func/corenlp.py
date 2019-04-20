from pycorenlp import StanfordCoreNLP
from tqdm import tqdm

import matplotlib.pyplot as plt
import re



class CoreNLP():
    def __init__(self):
        self.nlp=StanfordCoreNLP('http://localhost:9000')
        self.interro_list=["WHNP","WHADVP","WRB","WHADJP","WDT","WP","WP$","WRB"]
        self.count=-1

    #動詞が含まれるかのチェック
    def verb_check(self,token_list):
        for token in token_list:
            pos=token["pos"]
            if "VB" in pos:
                return True
        return False

    #textの疑問詞句を抽出する
    #疑問詞句のリストと、それ以外の単語のリストを返す
    def forward(self,text):
        q=self.nlp.annotate(text, properties={'annotators': 'tokenize,ssplit,parse','outputFormat': 'json'})

        tokens=q["sentences"][0]["tokens"]              #文の中の単語
        deps=q["sentences"][0]["basicDependencies"]     #依存関係（使わない）
        parse_text=q["sentences"][0]["parse"]           #句構造のパース結果

        token_list=[{"index":token["index"],"text":token["originalText"],"pos":token["pos"]} for token in tokens]
        parse_text=parse_text.replace("(","( ").replace(")"," )").split()

        WP_list=[]      #疑問詞句に含まれる単語
        NotWP_list=[]   #疑問詞句に含まれない単語
        WP_flag=False   #疑問詞句を発見済みならTrue

        depth=0
        for i in range(len(parse_text)-1):
            #depthが0の場合かつ、（疑問詞の句構造に未突入、または、すでに疑問詞が見つかっている）
            if depth==0 and (parse_text[i] not in self.interro_list or WP_flag==True):
                if parse_text[i]!=")" and parse_text[i+1]==")":
                    NotWP_list.append(parse_text[i])
                continue
            #疑問詞の句構造の内部にいる時
            else:
                WP_flag=True
                depth=max(depth,1)
                if parse_text[i]=="(":
                    depth+=1
                elif parse_text[i]==")":
                    depth-=1
                if parse_text[i]!=")" and parse_text[i+1]==")":
                    WP_list.append(parse_text[i])

        return WP_list,NotWP_list

    #疑問詞句、疑問詞句以外、動詞の有無を返す
    def forward_verbcheck(self,text):#input:(batch,seq_len)
        q=self.nlp.annotate(text, properties={'annotators': 'tokenize,ssplit,parse','outputFormat': 'json'})

        tokens=q["sentences"][0]["tokens"]              #文の中の単語
        deps=q["sentences"][0]["basicDependencies"]     #依存関係（使わない）
        parse_text=q["sentences"][0]["parse"]           #句構造のパース結果

        token_list=[{"index":token["index"],"text":token["originalText"],"pos":token["pos"]} for token in tokens]
        parse_text=parse_text.replace("(","( ").replace(")"," )").split()

        WP_list=[]      #疑問詞句に含まれる単語
        NotWP_list=[]   #疑問詞句に含まれない単語
        WP_flag=False   #疑問詞句を発見済みならTrue

        #疑問詞句の探索
        depth=0
        for i in range(len(parse_text)-1):
            #depthが0の場合かつ、（疑問詞の句構造に未突入、または、すでに疑問詞が見つかっている）
            if depth==0 and (parse_text[i] not in self.interro_list or WP_flag==True):
                if parse_text[i]!=")" and parse_text[i+1]==")":
                    NotWP_list.append(parse_text[i])
                continue
            #疑問詞の句構造の内部にいる時
            else:
                WP_flag=True
                depth=max(depth,1)
                if parse_text[i]=="(":
                    depth+=1
                elif parse_text[i]==")":
                    depth-=1
                if parse_text[i]!=")" and parse_text[i+1]==")":
                    WP_list.append(parse_text[i])

        vb=self.verb_check(token_list)

        return WP_list,NotWP_list,vb
