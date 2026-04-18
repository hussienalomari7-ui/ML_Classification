import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support, precision_score, recall_score

rs = 123
# Ignore any deprecation warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

tumor = pd.read_csv("tumor.csv")
print(tumor.head())
print(tumor.columns)

x = tumor.iloc[: ,:-1]
y = tumor.iloc[: ,-1]
print( "features :" ,x)
print( "label:" , y )

print(x.describe())

print(y.value_counts(normalize=True))

y.value_counts().plot.bar(color=["green","red"])
# plt.show()

x_train , x_test, y_train ,y_test =train_test_split(x,y,test_size=0.2,stratify=y,random_state=rs)

knn_model = KNeighborsClassifier(n_neighbors=2)
knn_model.fit(x_train, y_train.values.ravel())

preds =knn_model.predict(x_test)


def evalute_metrics(yt ,yp ):
    results_pos ={}
    results_pos['accuracy'] =accuracy_score(yt,yp)
    precision , recall , f_beta , _ =precision_recall_fscore_support(yt,yp , average="binary")
    results_pos["recall"] = recall
    results_pos["precision"]= precision
    results_pos["f1score"] =f_beta
    return results_pos 

evalute_metrics(y_test ,preds)
print("this : " , evalute_metrics)

rslt ={'accuracy':[] ,'recall':[], "precision":[] ,"f1score":[] }

for k in range(1,51) : 
    knn_model = KNeighborsClassifier(n_neighbors= k)
    knn_model.fit(x_train,y_train.values.ravel())
    preds =knn_model.predict(x_test)
    rslt_ = evalute_metrics(y_test , preds)
    
    rslt['accuracy'].append(rslt_['accuracy'])
    rslt['recall'].append(rslt_['recall'])
    rslt['precision'].append(rslt_['precision'])
    rslt['f1score'].append(rslt_['f1score'])

result_df= pd.DataFrame(rslt)
best_k = result_df["f1score"].idxmax() +1 
print(best_k)


knn_model= KNeighborsClassifier(n_neighbors=best_k)
knn_model.fit(x_train, y_train.values.ravel())
preds = knn_model.predict(x_test)
evalute_metrics(y_test , preds)
print("******************************************************")

plt.figure(figsize=(10,6))
plt.plot(range(1, 51), result_df['f1score'])
plt.xlabel('K')
plt.ylabel('F1 Score')
plt.title('F1 Score for Different K Values')
plt.show()
