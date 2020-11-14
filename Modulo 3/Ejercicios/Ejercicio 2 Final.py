'''
EJERCICIO - 2

Nombre: VICTOR CRUZ GOMEZ
Curso: Machine Learning - Modulo III
'''

import os
import pandas as pd

############# CARGADO DE DATOS #############
#Definir la ruta de la capeta Blog, permisos de lectura/escritura.
path='C:/Users/vic/Documents/Victor Cruz Gomez Windows 10/CursoMachineLearning/Modulo 3/blog/'
access_rigths=0o755
fileName=''

anios=['2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015']

#Dataframe para almacenar el anio y contenido de cada cuento.
data={}

for i, anio in enumerate(anios):
    fileName=path+anio+'.txt'
    file = open(fileName,"r")    
    content=file.read()
    data[anio]=content
    file.close()

    
############# EXPLORACION INICIAL #############
#Diccionario para almacenar los registros del dataframe 'data'.  en clave (anio) y valor(contenido). 
data_combined={key:[value] for (key,value) in data.items()} 

#Guardar las tuplas del diccionario tranpuesto en el dataframe 'data_df'.
data_df=pd.DataFrame.from_dict(data_combined).transpose()
data_df.columns=['content']
data_df=data_df.sort_index()


############# LIMPIEZA DE DATOS #############
import re
import string

#Metodo - Efecutar la limpieza del contenido de los cuentos.
    #Convertir el texto en minusulas.
    #Eliminar texto entre corchetes.
    #Eliminar puntos.
    #Eliminar palabras que contienen numeros.
    #Eliminar signos de puntuacion.
    #Eliminar saltos de linea.
def clean_content(content):
    content=content.lower()
    content=re.sub('\[.*?¿\]\%', ' ', content)
    content=re.sub('[%s]' % re.escape(string.punctuation), ' ', content)
    content = re.sub('\w*\d\w*', '', content)
    content = re.sub('[‘’“”…«»]', '', content)
    content = re.sub('\n', ' ', content)
    return content

#Definir una funcion anonia para la limpieza de datos.    
lambda_clean_1=lambda x: clean_content(x)

#Efectuar la limpieza de los datos.
data_clean= pd.DataFrame(data_df.content.apply(lambda_clean_1))
data_clean=data_clean.sort_index()

#Crear bolsa de palabras
from sklearn.feature_extraction.text import CountVectorizer

nltk_spanish='C:/Users/vic/Documents/Victor Cruz Gomez Windows 10/CursoMachineLearning/Modulo 3/Ejercicios/spanish.txt'
 
#Obtener el corpus en espanol en una lista.
with open(nltk_spanish) as f:
    lines = f.read().splitlines()

#Eliminar los stop_words en espanol y contar la frecuencia de cada palabra.
cv = CountVectorizer(stop_words=lines)
data_cv = cv.fit_transform(data_clean.content)
data_dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
data_dtm.index = data_clean.index


############# ANALISIS EXPLORATORIO #############
data=data_dtm.transpose()
#Analisis - Palabras mas usadas por anio
top_dict = {}

for c in data.columns:
    top = data[c].sort_values(ascending=False).head(30)
    top_dict[c]= list(zip(top.index, top.values))
print(top_dict)
# Print the top 15 words by year
for anio, top_words in top_dict.items():
    print(anio)
    print(', '.join([word for word, count in top_words[0:14]]))

#Analisis - Mostrar las palabras mas comunes
from collections import Counter

words = []
for anio in data.columns:
    top = [word for (word, count) in top_dict[anio]]
    for t in top:
        words.append(t)
        
print(Counter(words).most_common())
add_stop_words = [word for word, count in Counter(words).most_common() if count > 6]
add_stop_words

#Analisis - Actualizamos nuestra Bag of Words
from sklearn.feature_extraction import text 
from sklearn.feature_extraction.text import CountVectorizer

#adicionar nuevos stop words
with open(nltk_spanish) as f:
    stop_words = f.read().splitlines()
for pal in add_stop_words:
    stop_words.append(pal)
more_stop_words=['alex','lucas','andrés','mirta','tres','primer','primera','dos','uno','veces', 'así', 'luego', 'quizá','cosa','cosas','tan','asi','andres','todas','sólo','jesús','pablo','pepe']
for pal in more_stop_words:
    stop_words.append(pal)

#Recrear el dataframe 'data_stop' sin los nuevos stop_words
cv = CountVectorizer(stop_words=stop_words)
data_cv = cv.fit_transform(data_clean.content)
data_stop = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
data_stop.index = data_clean.index

#Analisis - Grafico nueve de palabras
from wordcloud import WordCloud

wc = WordCloud(stopwords=stop_words, background_color="white", colormap="Dark2",
               max_font_size=150, random_state=42)

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [16,12]

#Crear subplots por cada anio
for index, anio in enumerate(data.columns):
    wc.generate(data_clean.content[anio])
    plt.subplot(4, 3, index+1)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(anios[index])
plt.show()

#Analisis - Estadisticas de palabras por anio.
#Encontrar el numero de palabras unicos por anio.
unique_list = []
for anio in data.columns:
    uniques = data[anio].to_numpy().nonzero()[0].size
    unique_list.append(uniques)

#Crear un nuevo dataframe que contenga las palabras unicas.
data_words = pd.DataFrame(list(zip(anios, unique_list)), columns=['Anio', 'unique_words'])
data_unique_sort = data_words #sin ordenar
data_unique_sort

posts_per_year=[]
try:
  enlaces
except NameError:
    posts_per_year = [50, 27, 18, 50, 42, 22, 50, 33, 31, 17, 33, 13]
else:
    for i in range(len(anios)):
        arts = enlaces[i]
        print(anios[i],len(arts))
        posts_per_year.append(min(len(arts),MAX_POR_ANIO))

#Encontrar el total de numero de palabras por anio.
total_list = []
for anio in data.columns:
    totals = sum(data[anio])
    total_list.append(totals)
    
#Adiciar columnas adicionales al dataframe 'data_words'.
data_words['total_words'] = total_list
data_words['posts_per_year'] = posts_per_year
data_words['words_per_posts'] = data_words['total_words'] / data_words['posts_per_year']

#Ordernar el dataframe por minuto para ver quien habla mas rapido o mas lento
data_wpm_sort = data_words #sin ordenar
data_wpm_sort

#Analisis - Visualizacion de la tabla
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


############# ANALISIS DE SENTIMIENTO ############# 
#https://pypi.org/project/SentiLeak/
'''
NOTA: Para el analisis de sentimiento del 'contenido' se utilizo la libreria Sentileak en espanol. 
      Despues de la primera ejecucion, el analizador el sentimiento, genera el siguiente error.
[E090] Extension 'stem' already exists on Token. To overwrite the existing extension, set `force=True` on `Token.set_extension`.

Solucion: Modificar el archivo analizer.py. En la linear 245, adicionar force=True, similar a lo siguiente: Token.set_extension("stem", default="", force=True)
'''

from sentileak import SentiLeak
#sent_analysis = SentiLeak()

data = data_df
#lambda_predict = lambda x: sent_analysis.compute_sentiment(x).get('global_sentiment')
lambda_predict = lambda x: SentiLeak().compute_sentiment(x).get('global_sentiment')

data['sentimient'] = data['content'].apply(lambda_predict)
data