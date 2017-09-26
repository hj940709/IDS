#cd ./data sets/HASY

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from tpot import TPOTClassifier
from scipy import misc
from matplotlib import pyplot as plt
#2
raw = pd.read_csv("./hasy-data-labels.csv")
data = np.array(raw)
data = data[np.where(np.logical_and(data[:,1]<=80,data[:,1]>=70))]

#data =  np.array([misc.imread(path,flatten=True,mode='L').flatten() for path in data[:,0]])

np.random.shuffle(data)
train = data[0:int(len(data)*0.8)]
test = data[int(len(data)*0.8):]

trainX = np.array([misc.imread(path,flatten=True,mode='L').flatten() for path in train[:,0]])
testX = np.array([misc.imread(path,flatten=True,mode='L').flatten() for path in test[:,0]])
trainY = train[:,2]
testY = test[:,2]

lr_clf = LogisticRegression(random_state=1,multi_class="ovr")
lr_clf.fit(trainX,trainY)

(y,counts)=np.unique(trainY,return_counts=True)
print("Guess Loss:", 1-(y[np.argmax(counts)]==testY).sum()/len(testY))
print("LR Loss:",1-(lr_clf.predict(testX)==testY).sum()/len(test))

with plt.style.context("default"):
    plt.imshow(testX[np.where((lr_clf.predict(testX)!=testY))][4].reshape((32,32)))
    plt.show()


#3
rf_clf = RandomForestClassifier(n_estimators=10,random_state=1)
rf_clf.fit(trainX,trainY)
print("LR Loss:",1-(rf_clf.predict(testX)==testY).sum()/len(test))

clf = [RandomForestClassifier(n_estimators=n,random_state=1).fit(trainX,trainY) for n in range(10,210,10)]
loss = np.array([1-(c.predict(testX)==testY).sum()/len(test) for c in clf])
with plt.style.context("default"):
    plt.plot([n for n in range(10, 210, 10)], loss, "bo-")
    plt.show()

np.random.shuffle(data)
train = data[0:int(len(data)*0.8)]
valid = data[int(len(data)*0.8):int(len(data)*0.9)]
test = data[int(len(data)*0.9):]
trainX = np.array([misc.imread(path,flatten=True,mode='L').flatten() for path in train[:,0]])
validX = np.array([misc.imread(path,flatten=True,mode='L').flatten() for path in valid[:,0]])
testX = np.array([misc.imread(path,flatten=True,mode='L').flatten() for path in test[:,0]])
trainY = train[:,2]
validY = valid[:,2]
testY = test[:,2]


clf = [RandomForestClassifier(n_estimators=n,random_state=1).fit(trainX,trainY) for n in range(10,210,10)]
loss = np.array([1-(c.predict(validX)==validY).sum()/len(test) for c in clf])
with plt.style.context("default"):
    plt.plot([n for n in range(10,210,10)],loss,"bo-")
    plt.show()

print("RF Loss:",1-(clf[np.argmin(loss)].predict(testX)==testY).sum()/len(test))

tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2)
tpot.fit(trainX, trainY)
print("TPOT Loss:",1-(tpot.predict(testX)==testY).sum()/len(test))
