# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#Guardar en la ruta root/Labs
import gym

#Cargando un juego de consolo de atari. No es lo mismo que renderizar
environment = gym.make("MountainCar-v0")
environment.reset()

for _ in range(2000):
    #Mostrar el juego en la pantalla del usuario
    environment.render()
    #Trabajo con los movimientos.
    environment.step(environment.action_space.sample())
#Cerrar el ambiente para que no permanesca en memoria
environment.close()


    
