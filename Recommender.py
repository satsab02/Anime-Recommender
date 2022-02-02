from encodings import utf_8


import pandas as pd
def Readfile(file_path):
    data = {'User_ID':[], 'Anime_ID':[], 'Rating':[]}
    f = open(file_path, 'r', encoding= "utf_8")
    for line in f:
        
        User_id, Anime_id, score = line.split(",")
        data['User_ID'].append(User_id)
        data['Anime_ID'].append(Anime_id)
        data['Rating'].append(score.strip())
        
       
    f.close()
    return pd.DataFrame(data)
dataset = Readfile("Archive\Data_Reviewed.csv")
dataset['Rating']=dataset["Rating"].astype(float)
#print(dataset.head())

df_title = pd.read_csv("Archive\Data_Cleaned.csv", encoding="utf_8", header= None, names = ['Anime_ID', 'Anime_name','Rating'])
#print(df_title.head(10))

import surprise

from surprise import Reader, Dataset
from surprise import SVD
from surprise.model_selection import cross_validate
from scipy.sparse import csr_matrix
reader = Reader()
data = Dataset.load_from_df(dataset[['User_ID', 'Anime_ID','Rating']], reader)
svd = SVD()
# Run 5-fold cross-validation and print results
cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

trainset= data.build_full_trainset()
svd.fit(trainset)

df_0 = dataset[(dataset["Rating"]>=7) & (dataset["User_ID"]=="0")]
df_0 = df_0.set_index("Anime_ID")
#print(df_0.head())
df_0 = df_0.merge(df_title)["Anime_name"]
print(df_0.head(df_0.shape[0]))

titles = df_title.copy()

titles["Estimate_Score"] = titles["Anime_ID"].apply(lambda x: svd.predict(0, x).est)
titles = titles.sort_values(by=["Estimate_Score"], ascending=False)
print(titles["Anime_name"])

