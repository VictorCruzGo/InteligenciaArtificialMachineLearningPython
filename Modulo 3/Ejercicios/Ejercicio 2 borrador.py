'''
Nombre: VICTOR CRUZ GOMEZ
Curso: Machine Learning - Modulo III
'''
#Cuan positivo y negativo es un texto.
#Hay tres escalas negativo, neutral, positivo.

import os
import pandas as pd
import re
import string

path='C:/Users/vic/Documents/Victor Cruz Gomez Windows 10/CursoMachineLearning/Modulo 3/blog/'
access_rigths=0o755
fileName=''

anios=['2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015']

data={}

#Cargado de datos
for i, anio in enumerate(anios):
    #print(i)
    #print(anio)
    fileName=path+anio+'.txt'
    file = open(fileName,"r")    
    content=file.read()
    #print(content)
    data[anio]=content
    file.close()
    
#Exploracion inicial
#print(data.keys())
#print(data.values())
#print(data['2006'][:])

data_combined={key:[value] for (key,value) in data.items()} 

#print(type(data_combined))
#print(data_combined)


print('--------------------dataframe')
#pd.set_option('max_colwidth',150)

data_df=pd.DataFrame.from_dict(data_combined).transpose()
data_df.columns=['content']
data_df=data_df.sort_index()

#Limpieza de datos
def clean_content(content):
    content=content.lower()
    content=re.sub('\[.*?¿\]\%', ' ', content)
    content=re.sub('[%s]' % re.escape(string.punctuation), ' ', content)
    content = re.sub('\w*\d\w*', '', content)
    content = re.sub('[‘’“”…«»]', '', content)
    content = re.sub('\n', ' ', content)
    return content
    
clean_1=lambda x: clean_content(x)

data_clean= pd.DataFrame(data_df.content.apply(clean_1))
data_clean=data_clean.sort_index()


from sklearn.feature_extraction.text import CountVectorizer

nltk_spanish='C:/Users/vic/Documents/Victor Cruz Gomez Windows 10/CursoMachineLearning/Modulo 3/Ejercicios/spanish.txt'
 
with open(nltk_spanish) as f:
    lines = f.read().splitlines()
 
cv = CountVectorizer(stop_words=lines)
data_cv = cv.fit_transform(data_clean.content)
data_dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
data_dtm.index = data_clean.index
 
#data_dtm.to_pickle("dtm.pkl")
#data_clean.to_pickle('data_clean.pkl')

#pickle.dump(cv, open("cv.pkl", "wb"))
 
#data_dtm

#Analisis exploratorio
data=data_dtm.transpose()

#Palabras mas usadas por ano
top_dict = {}

for c in data.columns:
    top = data[c].sort_values(ascending=False).head(30)
    top_dict[c]= list(zip(top.index, top.values))
print(top_dict)
# Print the top 15 words by year
for anio, top_words in top_dict.items():
    print(anio)
    print(', '.join([word for word, count in top_words[0:14]]))


#Agregamos stop words
from collections import Counter

words = []
for anio in data.columns:
    top = [word for (word, count) in top_dict[anio]]
    for t in top:
        words.append(t)
print(Counter(words).most_common())
add_stop_words = [word for word, count in Counter(words).most_common() if count > 6]
add_stop_words

#Actualizamos nuestra Bag of Words
from sklearn.feature_extraction import text 
from sklearn.feature_extraction.text import CountVectorizer

# Add new stop words
with open(nltk_spanish) as f:
    stop_words = f.read().splitlines()
for pal in add_stop_words:
    stop_words.append(pal)
more_stop_words=['alex','lucas','andrés','mirta','tres','primer','primera','dos','uno','veces', 'así', 'luego', 'quizá','cosa','cosas','tan','asi','andres','todas','sólo','jesús','pablo','pepe']
for pal in more_stop_words:
    stop_words.append(pal)

# Recreate document-term matrix
cv = CountVectorizer(stop_words=stop_words)
data_cv = cv.fit_transform(data_clean.content)
data_stop = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
data_stop.index = data_clean.index

#Nueve de palabras
from wordcloud import WordCloud

wc = WordCloud(stopwords=stop_words, background_color="white", colormap="Dark2",
               max_font_size=150, random_state=42)

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [16,12]

# Create subplots for each anio
for index, anio in enumerate(data.columns):
    wc.generate(data_clean.content[anio])
    plt.subplot(4, 3, index+1)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(anios[index])
plt.show()

#Estadisticas de palabras por ano
# Find the number of unique words per Year
# Identify the non-zero items in the document-term matrix, meaning that the word occurs at least once

unique_list = []
for anio in data.columns:
    #uniques = data[anio].nonzero()[0].size
    uniques = data[anio].to_numpy().nonzero()[0].size
    unique_list.append(uniques)

# Create a new dataframe that contains this unique word count
data_words = pd.DataFrame(list(zip(anios, unique_list)), columns=['Anio', 'unique_words'])
#data_unique_sort = data_words.sort_values(by='unique_words')
data_unique_sort = data_words # sin ordenar
data_unique_sort
# ejecuta este si hicimos el webscrapping, o no tenemos los valores en la variable
posts_per_year=[]
try:
  enlaces
except NameError:
  # Si no hice, los tengo hardcodeados:
    posts_per_year = [50, 27, 18, 50, 42, 22, 50, 33, 31, 17, 33, 13]
else:
    for i in range(len(anios)):
        arts = enlaces[i]
        #arts = arts[0:10] #limito a maximo 10 por año
        print(anios[i],len(arts))
        posts_per_year.append(min(len(arts),MAX_POR_ANIO))

# Find the total number of words per Year
total_list = []
for anio in data.columns:
    totals = sum(data[anio])
    total_list.append(totals)
    
# Let's add some columns to our dataframe
data_words['total_words'] = total_list
data_words['posts_per_year'] = posts_per_year
data_words['words_per_posts'] = data_words['total_words'] / data_words['posts_per_year']

# Sort the dataframe by words per minute to see who talks the slowest and fastest
#data_wpm_sort = data_words.sort_values(by='words_per_posts')
data_wpm_sort = data_words #sin ordenar
data_wpm_sort


#Visualizacion de la tabla
import numpy as np
plt.rcParams['figure.figsize'] = [16, 6]

y_pos = np.arange(len(data_words))

plt.subplot(1, 3, 1)
plt.barh(y_pos,posts_per_year, align='center')
plt.yticks(y_pos, anios)
plt.title('Number of Posts', fontsize=20)


plt.subplot(1, 3, 2)
plt.barh(y_pos, data_unique_sort.unique_words, align='center')
plt.yticks(y_pos, data_unique_sort.Anio)
plt.title('Number of Unique Words', fontsize=20)

plt.subplot(1, 3, 3)
plt.barh(y_pos, data_wpm_sort.words_per_posts, align='center')
plt.yticks(y_pos, data_wpm_sort.Anio)
plt.title('Number of Words Per Posts', fontsize=20)

plt.tight_layout()
plt.show()

#Frecuencia de palabras
import nltk
from nltk.corpus import PlaintextCorpusReader
#corpus_root = './python_projects/blog' 
corpus_root=path
wordlists = PlaintextCorpusReader(corpus_root, '.*', encoding='latin-1')
#wordlists.fileids() # con esto listamos los archivos del directorio

cfd = nltk.ConditionalFreqDist(
        (word,genre)
        for genre in anios
        for w in wordlists.words(genre + '.txt')
        for word in ['casa','mundo','tiempo','vida']
        if w.lower().startswith(word) )
cfd.plot()


#(no funciona)
#Analisis de sentimientos 
'''
from classifier import SentimentClassifier
clf=SentimentClassifier()
sentiment = clf.predict(x)
'''

#(funciona, pero no hace una buena prediccion)
'''
from sentiment_analysis_spanish import sentiment_analysis
sentiment = sentiment_analysis.SentimentAnalysisSpanish()
text1 = "esta muy buena esa pelicula" #0.519
text2 = "Que horrible comida!!!"#-2.576
text3 = "Tuve una experiencia neutral"
print(sentiment.sentiment(text3))
'''

#(funcion, y hace una correcta prediccion)
#https://pypi.org/project/SentiLeak/
from sentileak import SentiLeak
sent_analysis = SentiLeak()
#text = "La decisión del árbitro fue muy perjudicial para el equipo local. El partido estaba empatado para ambos equipos. Al final, el portero hizo una gran intervención que salvó a su equipo."
#text1 = "esta muy buena esa pelicula"
#pred=sent_analysis.compute_sentiment(text1).get('global_sentiment') #2.0
#print(pred)

#text2 = "Que horrible comida!!!"
#sent_analysis.compute_sentiment(text2) #-4.0

#text3 = "Tuve una experiencia neutral"
#sent_analysis.compute_sentiment(text3) #0.0

#print (sentiment)

#data = pd.read_pickle('corpus.pkl')
data = data_df
    
lambda_predict = lambda x: sent_analysis.compute_sentiment(x).get('global_sentiment')

data['sentimient'] = data['content'].apply(lambda_predict)
data

