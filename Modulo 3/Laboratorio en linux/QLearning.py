# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 20:19:00 2020

@author: vic
"""

import gym
import numpy as np

#El aprendizaje esta garantizado esta entre 30k y 50k
#Son 50k episodios que tiene que recorrer el agente para aprender.
#Un agente recien al los 10k empieza aprender
MAX_NUM_EPISODES = 50000
#200 acciones que va realizando
#200 por cada episodio, 200 pasos.
STEPS_PER_EPISODES=200
#epsilon es para verificar el aprendizaje.  Desde -200 tiene que ir bajando hasta 0.005
#Tiene que ir bajando
EPSILON_MIN=0.005

max_num_steps=MAX_NUM_EPISODES*STEPS_PER_EPISODES
EPSILON_DECAY=500*EPSILON_MIN/max_num_steps
ALPHA=0.05
GAMMA=0.98
NUM_DISCRETE_BINS=30

#Estructura generica de Qlearning
class QLearner(object):
    def __init__(self,enviroment):
        #metodo de inicializacion del agente, variables
        #En el espacio de informacion teniamos la informacion de cota inferior, superior, limite, ancho
        self.obs_shape=enviroment.observation_space.shape #forma
        self.obs_high=enviroment.observation_space.high #alto
        self.obs_low=enviroment.observation_space.low #bajo
        #Limite donde se va a mover el agente
        self.obs_bins=NUM_DISCRETE_BINS
        self.bin_width=(self.obs_high-self.obs_low)/self.obs_bins
        
        self.action_shape=enviroment.action_space.n
        self.Q=np.zeros((self.obs_bins+1,self.obs_bins+1,self.action_shape))#matriz de 31x31
        self.alpha=ALPHA
        self.gamma=GAMMA
        self.epsilon=1.0
    
        
    def discretize(self, obs):
        #Movimiento en los limites permitodos
        #Movimiento que se esta realizando dentro de la matriz de 31x31, para que no se salga de los limites.
        return tuple((obs-self.obs_low)/self.bin_width).astype(int)
    
    def get_action(self, obs):
        #Obtener informacion de los movimientos permitidos
        #Obtener el movimiento del agente en un determinado tiempo
        discrete_obs=self.discretize(obs)
        if self.epsilon>EPSILON_MIN:
            #Si es mayor el agente no esta aprendiendo
            self.epsilon-=EPSILON_DECAY
        if np.random.random()>self.epsilon:
            #Trabajar con probabilidad
            #Si el valor aleatorio es mayor al epsilon, trabajar con ..
            return np.argmax(self.Q[discrete_obs])
        else:
            #Devolver un valor aleatorio
            return np.random.choice([a for a in range(self.action_shape)])        
        
    def learn(self,obs,action,reward,next_obs):
        #Aprendizaje por cada movimiento realizado.
        #Tiene que entender que accion esta realizando en cada movimiento
        discrete_obs=self.discretize(obs)
        discrete_next_obs=self.discretize(next_obs)
        self.Q[discrete_obs][action]+=self.alpha*(reward+self.gamma*np.max(self.Q[discrete_next_obs])-self.Q[discrete_obs][action])        
        
    def train(agent, enviroment):
        #Entrenamiento, devolver la mejor politica de entrenamiento.
        best_reward=float('inf')
        for episode in range(MAX_NUM_EPISODES):
            done=False
            obs=enviroment.reset()
            total_reward=0.0
            
            while not done:
                action = agent.get_action(obs)
                next_obs, reward, done, info = enviroment.step(action)
                agent.learn(obs, action, reward, next_obs)
                obs=next_obs
                total_reward=reward
            if total_reward>best_reward:
                best_reward=total_reward
            print('Episodio numero {} con recompensa:{}, mejor recompensa:{}, epsilo:{}'.format(episode, total_reward, best_reward, agente.epsilon))
        return np.argmax(agent.Q, axis=2)
        
    def test(agent enviroment, policy):
        #Testo
        done='False'
        obs=enviroment.reset()
        total_reward=0.0
        while not done:
            action=policy[agent.discretize(obs)]
            next_obs, reward, done, info=enviroment.step(action)
            obs=next_obs
            total_reward+=reward
        return total_reward
    
    
#Invocacion
if __name__=='__main__':
    enviroment=gym.make('MountainCar-v0')
    agent=QLearner(enviroment)
    learned_policy=train(agent, enviroment)
    monitor_path='./monitor_output'
    enviroment=gym.wrappers.Monitor(enviroment,monitor_path,force=True)
    for _ in range(1000):
        test(agent, environment, learned_policy)
    enviroment.close()
        
        
        
        

    