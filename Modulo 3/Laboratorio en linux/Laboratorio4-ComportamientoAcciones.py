# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 20:11:27 2020

@author: vic
"""

#Ver el comportamiento de los espacios de accion del juego
import gym
from gym.spaces import *
import sys

def print_spaces(space):
    print(space)
    #Cada juego tiene un conjunto de espacios de acciones. Movimientos.
    if isinstance(space, Box):
        print('\n cota inferior: ', space.low)
        print('\n cota inferior: ', space.high)
        
if __name___=='__main__':
    environment=gym.make(sys.argv[1])
    print('Espacion de observaciones')
    print_spaces(environment.observation_space)
    print('Espacion de acciones')
    print_spaces(environment.action_spaces)
    
    try:
        print('Descripcion de las acciones: ',environment.unwrapped.get_action_meannings)
    except:
        pass #Salga de la aplicacion

