# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 20:42:29 2020

@author: vic
"""

#Laboratorio 2 - Entornos
#Que son lo entornos?
#R.Breakout, un solo juego tiene 12 archivos.
#El entorno es el lugar donde se encuentra. 
from gym import envs #importar los entornos

#Almacenar todos los entornos
env_names=[env.id for env in env.registry.all()]

for name in sorted(env_names):
    #Imprimira una lista de todos los entornos de los juegos
    print(name)
    
    
