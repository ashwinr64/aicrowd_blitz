import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.neural_network import MLPClassifier
#from sklearn.svm import SVC
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score
from xgboost import XGBClassifier
#import lightgbm as lgb
from sklearn.feature_selection import SelectFromModel



all_data_path = "train.csv"
all_data = pd.read_csv(all_data_path)
final_test_path = "test.csv"
final_test = pd.read_csv(final_test_path)




X_train, X_val= train_test_split(all_data, test_size=0.2, random_state=42)

X_train,y_train = X_train.iloc[:,:-1],X_train.iloc[:,-1]
X_val,y_val = X_val.iloc[:,:-1],X_val.iloc[:,-1]


log_model = LogisticRegression(C=1,penalty = "l1",solver="liblinear",random_state = 7).fit(X_train,y_train)
model = SelectFromModel(log_model,prefit=True)
X_new = model.transform(X_train)


selected_features = pd.DataFrame(model.inverse_transform(X_new),index=X_train.index,columns=X_train.columns)
sel_col = selected_features.columns[selected_features.var()!=0]
#print(sel_col)


#clf = XGBClassifier()
clf = XGBClassifier(n_estimators = 100, learning_rate = 0.3)
clf.fit(X_train, y_train)

#prediction on the test set
y_pred=clf.predict(X_val)

# round float and convert to int
y_pred = y_pred.round(0)
y_pred= y_pred.astype(int)


# Calculating F1 Score
f1 = f1_score(y_val,y_pred,average='macro')
print("F1 score of the model is :" ,f1)



submission = clf.predict(final_test)
submission2=submission.round(0)

submission2=submission2.astype(int)
submission3 = pd.DataFrame(submission2)


submission3.to_csv('submission.csv',header=['class'],index=False)
