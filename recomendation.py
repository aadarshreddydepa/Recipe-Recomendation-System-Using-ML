import pandas as pd 
import numpy as np
from sklearn.neighbors import NearestNeighbors #k-NN algoritm 
from sklearn.preprocessing import StandardScaler #used to Normalize Numerical Features
from sklearn.feature_extraction.text import TfidfVectorizer #used to convert a text data into numerical values
file="recipe_final _dataset.csv"
recipe_df=pd.read_csv(file)
print(recipe_df.head(3))
print(recipe_df['ingredients_list'][0]) #prints the first list of ingredents present in the dataset
vectorizer=TfidfVectorizer()
X_ingredients=vectorizer.fit_transform(recipe_df['ingredients_list'])
scaler=StandardScaler()
X_Numericals=scaler.fit_transform(recipe_df[['calories','fat','carbohydrates','protein','cholesterol','sodium','fiber']])
#Now we need to combine both the features together using Numpy through horizontalStack (hstack)
X_combined=np.hstack([X_Numericals,X_ingredients.toarray()])

# Training the KNN Model using the above data 
knn=NearestNeighbors(n_neighbors=3,metric='euclidean')
knn.fit(X_combined)#Now the model is trained on the dataset

#defining a fuction to predict the recipue based on the input features

def recommendation_system(input_features):
    input_features_scaled=scaler.transform([input_features[:7]])
    input_features_transformed=vectorizer.transform([input_features[7]])
    input_combined=np.hstack([input_features_scaled,input_features_transformed.toarray()])

    distances,indices=knn.kneighbors(input_combined)
    recommendation=recipe_df.iloc[indices[0]]

    return recommendation[['recipe_name','ingredients_list','image_url']]

#Example input
input_features=[23,56,34,74,24,546,34,'olive oil,peanuts,bread,egg,chicken']
recommend=recommendation_system(input_features)
print(recommend)