from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
from fuzzywuzzy import process


knn5Model = pickle.load(open('C:\\Users\\Aya khaled\\Desktop\\AyaSiaf_final\\knn5Model.pk1' , 'rb'))
recommendation_set = pd.read_csv('C:\\Users\\Aya khaled\\Desktop\\AyaSiaf_final\\recommendation_set.csv')
music_data = pd.read_csv('C:\\Users\\Aya khaled\\Desktop\\AyaSiaf_final\\music_data.csv')
X_test = pd.read_csv('C:\\Users\\Aya khaled\\Desktop\\AyaSiaf_final\\X_test.csv')

#print(music_data.head(3))
#print(recommendation_set.head(3))
recommendation_set.drop(X_test.columns[0], axis=1, inplace=True)
#print(recommendation_set.head(3))
X_test.drop(X_test.columns[0], axis=1, inplace=True)
#print(X_test.head(3))


def recommender(song_name, data,model):
    idx=process.extractOne(song_name, recommendation_set['song'])[2]
    requiredSongs = recommendation_set.select_dtypes(np.number).drop(columns = ['cat','cluster','year']).copy()
    distances, indices = model.kneighbors(requiredSongs.iloc[idx].values.reshape(1,-1))
    for i in indices:
        rec = music_data['song'][i] + "      " + music_data['artist'][i]
    return rec

def get_song_info(row_number):
    song_info = recommendation_set.loc[row_number, ["song", "artist"]]
    return song_info


app = Flask(__name__)


@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')


@app.route('/songs', methods=['POST'])
def songs():
    song_info = get_song_info(1)
    song_name = song_info[0]
    reco = recommender(song_name,X_test, knn5Model)
    recom = pd.DataFrame(reco)

    return render_template('songs.html', tables=[recom.to_html(classes='recom')], titles=recom.columns.values) 
 
if __name__ == "__main__":
    app.run(debug=True)
