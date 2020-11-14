'''
EJERCICIO - 1

Nombre: VICTOR CRUZ GOMEZ
Curso: Machine Learning - Modulo III
'''

import urllib.request
from bs4 import BeautifulSoup
import os

############# URL Y OBTENCION DE DATOS #############
#Obtener los cuentos de del sitio web Hernan Casciari
response=urllib.request.urlopen('https://editorialorsai.com/category/epocas/')
html=response.read()
soup = BeautifulSoup(html,'html5lib')

#Definir la ruta de la capeta Blog y permisos de lectura/escritura.
path='C:/Users/vic/Documents/Victor Cruz Gomez Windows 10/CursoMachineLearning/Modulo 3/blog'
access_rigths=0o755

#Metodo - Obtener el anio del texto.
def getNumberFromText(text):    
    try:
        return str(text.split()[1])
    except:
        print('Fallo la extraccion del año.')
    else:
        print('Año extraido correctamente')
try:
    os.mkdir(path,access_rigths)
except OSError:
    print('Fallo la creacion de la carpeta')
else:
    print('Carpeta creado exitosamente')    


############# CUENTOS POR GESTION #############
#Filtrar las etiquetas h2(titulos) y p(contenido) de cada cuento y escribir en archivo txt.
print('Obteniendo los cuentos de Hernan Casciari dentro de la carpeta blog...')
for tag in soup.find_all(['h2','p']):            
    if (tag.name=='h2'):        
        fileName=path+'/'+getNumberFromText(tag.text)+'.txt'
        print('-----------------tag h')
        print(tag.name)
        print(tag.text)        
        
        file = open(fileName,"w")    
        file.close()
    else:
        print('------------------tag p')
        print(tag.name)
        print(tag.text)
        
        file = open(fileName,"a+")    
        file.write(tag.text)         
        file.close()
    
print('Web Scrapping finalizado.')