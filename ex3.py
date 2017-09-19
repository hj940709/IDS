#cd ./data sets/HASY

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy import misc

raw = pd.read_csv("./hasy-data-labels.csv")
data = np.array(raw.loc[raw["symbol_id"]>=70 & raw["symbol_id"]<=80]])
misc.imread(path,flatten=True,mode='L').flatten()
np.random.shuffle(data)

train = data[0:int(len(data)*0.8)]
test = data[int(len(data)*0.8):]

im = Image.open('your/image/path')
im_grey = im.convert('L') # convert the image to *greyscale*
im_array = np.array(im_grey)

y=np.array(train["Survived"])
trainX=np.array(train.drop("Survived",axis=1))
testX=np.array(test)

lr_clf = LogisticRegression(random_state=1,multi_class="ovr")
lr_clf.fit(trainX,y)
