import pandas as pd
import numpy as np
import json
import re
from nltk.stem.porter import *
from nltk.stem.snowball import SnowballStemmer
import sqlite3

#cd e:/document/ids/data sets
#1
train = pd.read_csv("./ex1-1.csv")
train.columns

df = train.drop("Name",axis=1).drop("Ticket",axis=1)

df.columns

def func(data):
    if(type(data)==float): return data
    result = ""
    for string in data.split(" "):
        if(len(result)==0 or result[-1]!=string[0]):
            result += string[0]
    return result
df["Deck"] = df["Cabin"].map(func)
df["Deck"] = df["Deck"].astype('category').cat.codes
df = df.drop("Cabin",axis=1)

df["Sex"] = df["Sex"].astype('category').cat.codes
df["Embarked"] = df["Embarked"].astype('category').cat.codes

df["Survived"] = df["Survived"].fillna(df["Survived"].mode())
df["Pclass"] = df["Pclass"].fillna(df["Pclass"].mode())
df["Sex"] = df["Sex"].fillna(df["Sex"].mode())
df["Age"] = df["Age"].fillna(int(np.ceil(df["Age"].mean()))).map(lambda x:int(np.ceil(x)))
df["SibSp"] = df["SibSp"].fillna(df["SibSp"].mode())
df["Parch"] = df["Parch"].fillna(df["Parch"].mode())
df["Fare"] = df["Fare"].fillna(df["Fare"].mean())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode())
deckmdoe = int(df["Deck"].map(lambda x:x==-1 and np.nan or x).mode())
df["Deck"] = df["Deck"].map(lambda x:x==-1 and deckmdoe or x)

df.to_csv("./processed.csv",index=False)
df.to_json("./processed.json",orient="records",lines=True)


#2
#read file
#json_str = open('./Automotive_5.json', 'rb').readlines()
df = pd.read_json('./Automotive_5.json',lines=True)
stop = open("./stop-word-list.csv", 'rb').read()
#decode
#json_str = [obj.decode("utf-8") for obj in json_str]
stop = stop.decode("utf-8")
#formalize
#json_str = "["+",".join(json_str)+"]"
stop = np.array(stop.split(", "))
#df = pd.read_json(json_str)

df["reviewText"] = df["reviewText"].map(lambda x:x.lower()).map(lambda x:re.sub(r'[^\w\s]','',x))

df["reviewText"] = df["reviewText"].map(lambda x: [word.strip() for word in x.split(" ") if word not in stop and len(word.strip())>0])

#stemmer = PorterStemmer()
stemmer = SnowballStemmer("english")
df["reviewText"] = df["reviewText"].map(lambda x: [stemmer.stem(word) for word in x])

df[df["overall"]>=4].to_csv("pos.csv")
df[df["overall"]<=2].to_csv("neg.csv")

#3
database = "ex1-3.sqlite"
conn = sqlite3.connect(database)
query = "select * from player "+\
	"inner join "+\
	"(select distinct player_id from hall_of_fame where inducted='Y');"
df = pd.read_sql(query,conn)

print(rows)
conn.close()


'''
select a.player_id,name_first,name_last
	from player as a inner join 
	(select distinct player_id from hall_of_fame where inducted='Y') as b
	on a.player_id=b.player_id;
select college_id,count(a.player_id) from
	player a inner join
	(select distinct player_id from hall_of_fame where inducted='Y') b 
	on a.player_id=b.player_id inner join
	(select distinct player_id,college_id from player_college) c
	on b.player_id=c.player_id
	group by college_id;
'''


