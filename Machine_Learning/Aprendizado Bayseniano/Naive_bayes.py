import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
import pickle

base_risco = pd.read_csv(r'C:\Users\aleph.alves\Desktop\class\Machine_Learning\Aprendizado Bayseniano\risco_credito.csv')

x_var = base_risco.iloc[:, 0:4].values
y_var = base_risco.iloc[:, 4].values

label_encoder_hist = LabelEncoder()
label_encoder_div = LabelEncoder()
label_encoder_garantia = LabelEncoder()
label_encoder_renda = LabelEncoder()

x_var[:,0] = label_encoder_hist.fit_transform(x_var[:,0])
x_var[:,1] = label_encoder_div.fit_transform(x_var[:,1])
x_var[:,2] = label_encoder_garantia.fit_transform(x_var[:,2])
x_var[:,3] = label_encoder_renda.fit_transform(x_var[:,3])

with open('risco_credito.pkl','wb') as f:
    pickle.dump([x_var,y_var],f)

# Algoritmo

naive_credit = GaussianNB()
naive_credit.fit(x_var,y_var)

previsao = naive_credit.predict([[0,0,1,2],[2,0,0,0]])

print(naive_credit.class_count_)