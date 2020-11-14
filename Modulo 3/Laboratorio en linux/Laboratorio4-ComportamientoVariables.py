# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 19:58:27 2020

@author: vic
"""

import gym
environment=gym.make('Qbert-v0')
MAX_NUM_EPISODES=10
MAX_STEPS_PER_EPISODE=500#500 pasas por cada episodio

for episode in range(MAX_NUM_EPISODES):
    obs=environment.reset() #El reseto es una observacion, una accion en el juego.
    
    for step in range(MAX_STEPS_PER_EPISODE):
        environment.render()
        action=environment.action_space.sample() #El movimeinto del agente se guarda en la variable action
        #train_test_split
        #
        next_state,reward,done,info=environment.step(action)
        
        obs=next_state #la observacion tiene que ser almacenada
        
        if done is True:
            print('\n Episodio #{} terminado en {} steps'.format(episodio,step+1))
            break;
environment.close()