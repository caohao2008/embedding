#encoding=UTF-8
import sys
import gensim
import numpy as np
from gensim.models import KeyedVectors

#file = 'tencent-ailab-embedding-zh-d100-v0.2.0-s.txt'
#file = 'tencent-ailab-embedding-en-d100-v0.1.0-s'
file = 'sim.txt'
#wv_from_text = KeyedVectors.load_word2vec_format(file, binary=False)
model = KeyedVectors.load_word2vec_format(file, binary=False)

# 定义计算两个文本相似度的函数
def text_similarity(text1, text2):
    # 把文本中的每个词转化为向量
    text1_vec = [model[word] for word in text1 if word in model.vocab]
    text2_vec = [model[word] for word in text2 if word in model.vocab]
   
    #print text1_vec
    #print text2_vec
    if(len(text1_vec)==0 or len(text2_vec)==0):
        return 0 
    # 对向量取平均值
    text1_vec_avg = sum(text1_vec) / len(text1_vec)
    text2_vec_avg = sum(text2_vec) / len(text2_vec)
    
    #print text1_vec_avg    
    #print text2_vec_avg    
    # 计算余弦相似度
    similarity = text1_vec_avg.dot(text2_vec_avg) / (np.linalg.norm(text1_vec_avg) * np.linalg.norm(text2_vec_avg))
    
    return similarity
    
# 示例：计算两个句子的相似度
text1 = 'I like apples'
#text2 = 'I love apples'
text2 = 'I hate potato'
while 1:
    try:
        text1 = input("please input first sentence : ")
        text2 = input("please input first sentence : ")
        similarity = text_similarity(text1.split(), text2.split())
        print('similarity', similarity)
    except KeyboardInterrupt:
        break
    except:
        print("error! please try again!")
