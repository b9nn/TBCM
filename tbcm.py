'''
The Triangular Body-Cover Model (TBCM) is a computational model used to simulate voice production, 
 in the context of vocal hyperfunction and phonotrauma. Expanding on the classic body-cover 
model (BCM), the TBCM incorporates a triangular glottal shape. This model allows for the simulation of
laryngal mechanisms. 

Coded by Benjamin Gladney
March 11th 2025

            --- References ---
[1] 



'''

import numpy as np
# https://numpy.org/doc/2.2/reference/generated/numpy.arange.html#numpy.arange

import matplotlib.pyplot as plt
# matplot crash course
# https://www.youtube.com/watch?v=3Xc3CA655Y4
# documentation ~ https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.pyplot.html

import scipy as sp

from scipy.integrate import solve_ivp 
# scipy resources:
# https://www.youtube.com/watch?v=MM3cBamj1Ms
# https://github.com/lukepolson/youtube_channel/blob/main/Python%20Tutorial%20Series/odes1.ipynb

# sample langrangian mechanics problem
# https://www.youtube.com/watch?v=rJaXxb_piGI


import pandas as pd
# https://pandas.pydata.org/docs/


import sympy as smp # allows for symbolic manipulation


#           --- Biomechanical Parameters ---
etau = 100*1e4; # [1/m^2] Non-linear spring coefficient for the upper mass
etal = 100*1e4; # [1/m^2] Non-linear spring coefficient for the lower mass
etab = 100*1e4; # [1/m^2] Non-linear spring coefficient for the body mass
zetau = 0.6; # 0.4; % [-] Basic damping factor for the upper mass
zetal = 0.1; # 0.4; % [-] Basic damping factor for the lower mass
zetab = 0.15; # 0.2; % [-] Damping factor for the body mass
zetauCol = 1.0; # 0.4; % [-] Collision damping factor for the upper mass
zetalCol = 1.0; # 0.4; % [-] Collision damping factor for the lower mass


mu = 0; # [kg] Mass of the upper cover mass
ml = 0; # [kg] Mass of the lower cover mass
mb = 0; # [kg] Mass of the body mass
xu0 = 0; # [m] Initial position of upper mass
xl0 = 0; # [m] Initial position of lower mass
xb0 = 3e-3; # [m] Initial position of body mass
ku = 0; # [N/m] Linear spring constant for the upper mass
kl = 0; # [N/m] Linear spring constant for the lower mass
kc = 0; # [N/m] Coupling spring constant for the cover layer
kb = 0; # [N/m] Linear spring constant for the body mass


