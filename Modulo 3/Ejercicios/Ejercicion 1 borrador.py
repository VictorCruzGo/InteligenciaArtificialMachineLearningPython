# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 13:19:50 2020

@author: vic
"""

import nltk
#nltk.download()

import urllib.request
from bs4 import BeautifulSoup

response=urllib.request.urlopen('https://editorialorsai.com/category/epocas/')
html=response.read()

soup=BeautifulSoup(html,'html5lib')
text=soup.get_text(strip=True)

tokens=[t for t in text.split()]

#print(html)
#print(text)
#print(tokens)


from nltk.tokenize import sent_tokenize
#print(sent_tokenize(text))


import bs4 
import urllib.request
from bs4 import BeautifulSoup
import re

response=urllib.request.urlopen('https://editorialorsai.com/category/epocas/')
html=response.read()

soup = BeautifulSoup(html,'html5lib')

#print(soup.prettify())

#print(soup.title)

#print(soup.title.name)

#print(soup.title.string)

#print(soup.p)

#print(soup.p.parent.name)

#print(soup.p['class'])

#soup.find_all('h2')

#soup.find_all('p')

#soup.find(id='nav_right')

cadena='victor'
print(type(cadena))

print('despues---')
things=soup.find_all(['h2','p'])
#things

'''
t=things[0]
type(t)
print(t.name)
print(t.text)

tt=things[1]
type(tt)
print(tt.name)
print(tt.text)
'''

'''
for i in range(things):
    print(type(i))
    print(i.name)
    print(i.text)
'''    




for etiqueta in soup.find_all(['h2','p']):        
    
    if (etiqueta.name=='h2'):
        nombreArchivo=etiqueta.text+'.txt'
        print('-----------------etiqueta h')
        print(type(etiqueta))
        print(etiqueta.name)
        print(etiqueta.text)
        
        file1 = open(nombreArchivo,"w")    
        file1.close()
    else:
        print('------------------etiqueta p')
        print(type(etiqueta))
        print(etiqueta.name)
        print(etiqueta.text)
        
        file1 = open(nombreArchivo,"a+")    
        file1.write(etiqueta.text)         
        file1.close()
    
