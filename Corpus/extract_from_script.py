#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 17:44:29 2023

@author: alberto
"""

import re
import pandas as pd

def obtener_personajes(texto):
    
    expresion = "[A-Z]+:"
    
    personajes = []
    for i in re.findall(expresion,texto):
        if i not in personajes:
            personajes.append(i)
    
    return personajes



f  = open('dialogos.txt','r')
content = f.read() 
f.close()

print(obtener_personajes(content))
    




    