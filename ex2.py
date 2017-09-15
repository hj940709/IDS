cd e:/document/ids/data sets

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D;
import seaborn as sns
import string
from scipy import stats
#1
processed = pd.read_csv("./processed.csv")
processed_noid = processed.drop(["PassengerId"],axis=1)
processed_noid.describe()
average_joe = np.array([int(processed_noid["Survived"].mode()),
                        int(processed_noid["Pclass"].mode()),
                        int(processed_noid["Sex"].mode()),
                        processed_noid["Age"].mean(),
                        int(processed_noid["SibSp"].mode()),
                        int(processed_noid["Parch"].mode()),
                        processed_noid["Fare"].mean(),
                        int(processed_noid["Embarked"].mode()),
                        int(processed_noid["Deck"].mode())])

survived = processed_noid[processed_noid["Survived"]==1]
survived_jane = np.array([1,int(survived["Pclass"].mode()),
                        int(survived["Sex"].mode()),
                        int(survived["Age"].mean()),
                        int(survived["SibSp"].mean()),
                        int(survived["Parch"].mean()),
                        survived["Fare"].mean(),
                        int(survived["Embarked"].mode()),
                        int(survived["Deck"].mode())])

dead = processed_noid[processed_noid["Survived"]==0]
dead_joe = np.array([0,int(dead["Pclass"].mode()),
                        int(dead["Sex"].mode()),
                        int(dead["Age"].mean()),
                        int(dead["SibSp"].mean()),
                        int(dead["Parch"].mean()),
                        dead["Fare"].mean(),
                        int(dead["Embarked"].mode()),
                        int(dead["Deck"].mode())]);

survived_similarity = np.square(np.array(survived)-survived_jane).sum(1)
dead_similarity = np.square(np.array(dead)-dead_joe).sum(1)
np.array(survived)[survived_similarity.argmin()]
survived_jane
np.array(dead)[dead_similarity.argmin()]
dead_joe

sns.countplot(survived["Sex"])
sns.countplot(dead["Sex"])

sns.distplot(survived["Age"],hist=False)
sns.distplot(dead["Age"],hist=False)

with plt.style.context("default"):
    plt.plot(np.array(survived["Age"]),np.array(survived["SibSp"]),"bo",alpha=0.3)
    plt.plot(np.array(dead["Age"]), np.array(dead["SibSp"]),"go", alpha=0.3)
    plt.show()

with plt.style.context("default"):
    ax = plt.figure().add_subplot(111, projection='3d')
    ax.scatter(xs =np.array(survived["Age"]),
                   ys=np.array(survived["SibSp"]),zs = np.array(survived["Fare"]),c="b")
    ax.scatter(xs=np.array(dead["Age"]),
               ys=np.array(dead["SibSp"]), zs=np.array(dead["Fare"]), c="g")
    ax.set_xlabel("Age")
    ax.set_ylabel("Sibsp")
    ax.set_zlabel("Fare")
    plt.show()


#2
raw_pos = open("./pos.txt", 'rb').read().decode("utf-8")
raw_neg = open("./neg.txt", 'rb').read().decode("utf-8")

pos_review = raw_pos.replace("[","").replace("'","")\
    .replace("]","").replace("\"","").replace("\r\n",",").split(",")
neg_review = raw_neg.replace("[","").replace("'","").replace("]","")\
    .replace("\"","").replace("\r\n",",").split(",")

pos = np.array([word.strip() for word in pos_review])
neg = np.array([word.strip() for word in neg_review])

def mode(array):
    (_, idx, counts) = np.unique(array, return_index=True, return_counts=True)
    index = idx[np.argmax(counts)]
    return array[index]

mode(pos)
mode(neg)

wordlist = np.unique(np.append(pos,neg))

getTfIdf = lambda x: np.array([(x==pos).sum()/len(pos),(x==neg).sum()/len(neg)])/(np.log2(3/(1+(x==pos).any()+(x==neg).any()))+1)

tfidf = np.fromiter(([getTfIdf(word) for word in wordlist]))

wordlist[tfidf[0].argmax()]
wordlist[tfidf[1].argmax()]



