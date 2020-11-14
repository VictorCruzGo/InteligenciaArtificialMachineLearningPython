# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 19:46:30 2020

@author: vic
"""

import gym
environment=gym.make('MountainCar-v0')
#Configurando cantidad maxima de episodios.
MAX_NUM_EPISODIOS=1000
for episode in range(MAX_NUM_EPISODIOS):
    #Informa si ya esta listo o no el juego
    done=False
    #En la variable de obs trabajar con el reseto del juego
    obs=environment.reset()
    #Por bellman siempre se asignas recompensan
    total_reward=0.0
    #Paso o accion que se va a realizar
    step=0

    #Verificacion aleatorio del agente    
    #Mientras no termine
    while not done:
        #Que se vaya visualizando en la pantalla
        environment.render()        
        action=environment.action_space.sample()    
        next_state,reward,done,info=environment.step(action)
        #Por cada accion o movimiento obtenemos una recompensa
        total_reward+=reward
        #Despues de realizar una accion    
        step+=1        
        obs=next_state
    print('\n episodio numero {} finalizado con {} iteraciones. Recompensa final = {}'.format(episode.step+1,total_reward))
environment.close()

#El algoritmo de Q-Learning tiene como finalidad de aprender.

        
        