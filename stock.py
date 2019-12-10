import pandas as pd

df=pd.read_csv('Data.csv', encoding = "ISO-8859-1")


train = df[df.Date < '20150101']
test = df[df.Date  > '20141231']


df.head()

data1=train.drop('Date',1)
data1=train.drop('Label',1)
data1.replace("[^a-zA-Z]"," ",regex=True, inplace=True)
data1.shape
list=[i for i in range(26)]
new_Index2=[str(i) for i in list]
data1.columns= new_Index2
data1.head(5)

for index in new_Index2:
    data1[index]=data1[index].str.lower()
    
headlines = []
for row in range(0,len(data1.index)):
    headlines.append(' '.join(str(x) for x in data1.iloc[row,0:25]))


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier


countvector=CountVectorizer(ngram_range=(2,2))
traindataset=countvector.fit_transform(headlines)

randomclassifier=RandomForestClassifier(n_estimators=200,criterion='entropy')
randomclassifier.fit(traindataset,train['Label'])


## Predict for the Test Dataset
test_transform= []
for row in range(0,len(test.index)):
    test_transform.append(' '.join(str(x) for x in test.iloc[row,2:27]))
test_dataset = countvector.transform(test_transform)
predictions = randomclassifier.predict(test_dataset)

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

matrix=confusion_matrix(test['Label'],predictions)
print(matrix)
score=accuracy_score(test['Label'],predictions)
print(score)
report=classification_report(test['Label'],predictions)
print(report)







print(data1["1"])
data1.head(1)