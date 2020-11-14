# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 19:51:04 2020

@author: vic
"""

import gym
import sys
def run_gym_environment(argv):
    environment=gym.make(arg[1])
    environment.reset()
    for _ in range(int(argv[2])):
        environment.render()
        environment.step(environment.action_space.sample())
    environment.close()
    
#Realizar la ejecucion normal del aplicativo
if __name__=="__main__":
    run_gym_environment(sys.argv)