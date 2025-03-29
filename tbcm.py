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
'''
import numpy as np
# https://numpy.org/doc/2.2/reference/generated/numpy.arange.html#numpy.arange

import matplotlib.pyplot as plt
# matplot crash course
# https://www.youtube.com/watch?v=3Xc3CA655Y4
# documentation ~ https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.pyplot.html

import scipy as sp

from scipy.integrate import solve_ivp 
from scipy.integrate import quad
# scipy resources:
# https://www.youtube.com/watch?v=MM3cBamj1Ms
# https://github.com/lukepolson/youtube_channel/blob/main/Python%20Tutorial%20Series/odes1.ipynb

# sample langrangian mechanics problem
# https://www.youtube.com/watch?v=rJaXxb_piGI


import pandas as pd
# https://pandas.pydata.org/docs/


import sympy as smp # allows for symbolic manipulation

# Titze and Story (2002) https://watermark.silverchair.com/367_1_online.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAACAowgggGBgkqhkiG9w0BBwagggf3MIIH8wIBADCCB-wGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQMbQBDndSoHCeUCSLaAgEQgIIHvd5H8GVSXH69O9d_ZqF8vaFsygTMZj9IvMcoyz66Wz4QrwTUppaWwczoIf324Hb4JElmJkwwOWZDnEGd1uBcSJpzqFtHLy5GGHU3Y0hDxqe2Cu5tvipcfaBVDyjv1YWLdQSNlvX-8N8ZpSJRHYLdCP4ZbFj_vl_451pltlFe5TrFC3F-0lyICHFW6RHQqkzQGbYz_iXch3MOLFzco_-Ji-XC-PciuntRYUaf6VAuLOzZkbBl7OA1GvgHCptdOFJxITO2w-RowMjoQQNR_2IcVuUfONn2zBqG9qIym3_PnWyvuUfyK57MxzmbAaNDQuUyFLABraq1DNvxjE0usfyo3TcpgG-O96__2YQuBEgvYP_VWs7IsgfzOl4CLUEL0Rw_X2WcodpVjoXI_IKVl93EGLiH3vV-_1RAnIEyNVLughMeJ0tlYtfgTcCHlDU9M0bT_7vDZlL4nc3Zcq7jUa8MOqPrt5OI1IX48tyrk2zp1DwlpbONYl9gR8Yr-0fQ_KEbleqMtwz0kWn5oSnabf_559ewdxx_v5HENW8FneE2XnYKRrinuvCu1wghucuTRCaVaZKoTtnBcGldaDb04UupGwwD3V1WJimKYHupF_3ouGPl07VP8NSAQNbaa_m7WrNgWaFBO2d9zZipi7Nov3jHYKUTjrD8frFgv__FNjw3f-xmA-uvrCFhPR9KVz4qnlo5nPnB1I_SR8Eoqd6xO8XEJL1HeT5r3bBmE1Cly1TO-1K5A72kMo2aPTXaxAgIwG_FJtbnicF3T7svsVgsYBCmn-kMkUutsOESeNn6AAfjaCoeWaXKgPQRuhDiDbCS5996amJJN3Ppuqb1zBLw8iyXnGhFL41HU33qJ8Z38Rdwgu---8UhtUNNwtGdZvYuZGbhsjdZCI939fSnkndcAVcbRilWXVewko7RT9DPhTyOs8qILNDxbHrWRYpaAv4H73Asv9VcPse7dpdDbhg-cstcy5pPgMh4kdKTU1QTpoY3PmCEou0HgmTVRkS1uich0XRXQbTzgoFUUP_yfpPwpJJnC3_aFeSxe928h5IdrRkTzY58CqQL09gJtj6qjY1bOPJZ6vfiYm8p6fig1RD3ZUxKdQn7giyKcgBjdKXrRqANsTqfMIHEoLZYyRPzBfMfdoncv5SPxcOsTklNmZr2wLI0ucGajuYONP1Nu6HrBracK3wiB2y-VxZamgWMXOSOORnRp1zf8NW2XIkmQwJ-Wq14nhrLHyxegtTYvDoCUuF5dxQ9Jxxs8Pdo20MwOKJ3C-bITsfCY7J7ciVWtfKjWNs00WUVdR0_hzTrjgTweor7uKTXFfBc3pgN5qXI08mtLXAtUYbIc8ZOjZCKb84_nK9urn1U2vNhEOdLBeyY8VE5E1VDcyd-hgUl0zkCaMht40Gk_fKnl4qI9rBTsmj-LbnRF6AD3DUsD94EjY1be2kP4dpzv_oAGHBYO7BGsTmZX6_zAUV13kefubvs8Ptu0twi_VPhpE_PHQTnkMFcGTbqt0079R_SgorqYDDv82cd11XJFvsRBVkG3yZUMsCHF-6S26LkPg4c6YoV3uY7Bqgrk2FSoJr3kZBI365kFrxGT-7spoXJQ6fxkVGyWP5Kg3fQWd9-FKEpHOYqED24LugKlib2uEs6pdbJNBtHpyW7qHH9RdQNPczqi57KsbjuZWlYPOWvp4fFCg9Jzf9YlO9bO41nj9e_XmJc9OcQ4VV-bB2-pgcaWA-ajWSu_O4n7bRKJDXye9M-7n_W10dADdJzgm7fEbZt_zNdOa04aU288UCmLX5HXVBeki0eLi-YKy89Ch7lpszGbzHPFF6yVvNoCALmbOhWvtGdxwHM1D8B_m8u2z9JOUSnD4_Y3pNCb7IeIXuhQD3hbmhc0gjKmOzMA-UKtZMhrdTDaCUlt7ei5wYSKdAPTyoKQgqayDZl6oCPMapk4VL_o6kVnHL1NPvOPJDbUYfVA3g_vlPfcSS-5WeFJfyGNXyf5lyn2hFT1Ijw5NaiSAhYET9tZSDAA5Zzj8qJQ9cfCN8Cen1usg7ncd9wJrWkJZZ2LlpUsic6TzGOw7stqB0mNOWtAfI5V_bRsstQy1H_rbmNbiLLVeF4N9r9gPp96qEh_vAlmaeG3qKoXCkjyHkadDslszOQwbBHM8vkUhg3h5wbdBS_BouBlOWtmMAaYPmHgQM7uHBIDYS2P9IjAKcl0s_chPeWa8JasKeZnjYmKFue_hea9UVTQmlS3qY7g7Vf1afZ70ww3nPtw37QGYAvHivGGbA7kT8LcGXx9aVqbgYhM7bLuR2fSdL9cAZRrAKu_X8p4_VGYC6CRxlYopQPKtG13nTbTvlZIsmijuS8Ezo4bm0X3fqM7sxbJ6P8uDuZWSG5-C7GLwhVAuHfVFy_6MJ9mP955UHo2uD3OS9GVvwwCqw15bJ9BgX0i-GVvHDuuInjnKf8lkTPTqhnxe4dq4hhWs4RSjOlYOi_gNJJFYObOFsFGJMs6ryN752_gcnijADuQND0t9U-lwRp74JvsKWiRBJC-bMvPUgi_tk6fxspJ9m_YF-PNYtb2TGfpHz3ysrZNVSzaPkiZXbzKFC0GeZX1ykifDC_JxflySo4oL82-lcVWfF7GMceY55nfDsHtsAu12j7oE4

# questions


# P_s = 579 # Pa [N/m^2]

# Retrieved from same Titze and Hunter 2007
P_s = 800 # Pa [N/m^2] Lung Pressure ~ Driving Pressure

# what is that symbol above the variables, look at eqn a14
# Needs to be determined from contact stiffness of vocal folds
k_uCol = None
k_lCol = None
k_bCol = None

# Titze and Story 2002
T_u = 0.3 # Thickness of the upper layer of the cover
T_l = 0.3 # Thickness of the lower layer of the cover

eta_uCol = 1.0 # = eta_u? right above eqn 21
eta_lCol = None

# Computed from posture modelling (length from )
L_u_closed = None # how do we obtain
L_l_closed = None
L_g = 16 # Vibratory length of the glottis, obtained from muscle activation rules

# how to obtain
# 
L_d = None 
L_c = None

#           --- Biomechanical Parameters ---

#           --- Masses ---
m_u = None  # Mass of the upper block (m1)
m_l = None  # Mass of the lower block (m2)
m_b = None  # Mass of the body block (M)

#           --- Spring Parameters ---
k_u = None  # Spring constant between the upper mass and the base (k1)
k_l = None  # Spring constant between the lower mass and the base (k2)
k_c = None  # Spring constant between the upper and lower masses (kc) (coupling)
k_b = None




#           --- Intraglottal Pressures ---
PDG = 0.480 # [mm] Posterior Glottal Disaplcement
a_r = 0.2 # [degrees]  Arytenoid Rotation
# consult dr.peterson, what is this angle a fucntion of



# Rest positions
x_u_0 = None  # Initial position of the upper mass
x_l_0 = None  # Initial position of the lower mass
x_b_0 = None  # Initial position of the body mass

x_u_1 = None  # Final position of the upper mass
x_l_1 = None  # Final position of the lower mass
x_b_1 = None  # Final position of the body mass

x_u_d = None  # First order derivative of position of upper mass wrt time
x_l_d = None  # First order derivative of position of lower mass wrt time
x_b_d = None  # First order derivative of position of body mass wrt time

x_u_dd = None  # Second order derivative of position of upper mass wrt time
x_l_dd = None  # Second order derivative of position of lower mass wrt time
x_b_dd = None  # Second  order derivative of position of body mass wrt time

u_displacement = x_u_1 - x_u_0 # Displacement of upper mass
l_displacement = x_l_1 - x_l_0 # Displacement of lower mass
b_displacement = x_b_1 - x_b_0 # Displacement of body mass

v_u_0 = 0.0  # Initial velocity of the upper mass
v_l_0 = 0.0  # Initial velocity of the lower mass
v_b_0 = 0.0  # Initial velocity of the body mass

#           ---  Non-Linear Spring Coefficients ---
eta_u = 100*1e4; # [1/m^2] Non-linear spring coefficient for the upper mass
eta_l = 100*1e4; # [1/m^2] Non-linear spring coefficient for the lower mass
eta_b = 100*1e4; # [1/m^2] Non-linear spring coefficient for the body mass

zeta_u = 0.6; # 0.4; % [-] Basic damping factor for the upper mass
zeta_l = 0.1; # 0.4; % [-] Basic damping factor for the lower mass
zeta_b = 0.15; # 0.2; % [-] Damping factor for the body mass

zetau_Col = 1.0; # 0.4; % [-] Collision damping factor for the upper mass
zetal_Col = 1.0; # 0.4; % [-] Collision damping factor for the lower mass

#           --- Proportionality Factors ---
alpha_u = L_u_closed / L_g
alpha_l = L_l_closed / L_g

#            --- Anatomical Parameters---
LREST_MALE = 1.6e-2; # [m] Vocal fold length at rest for male subject
TREST_MALE = 0.3e-2; # [m] Vocal fold thickness at rest for male subject
DEPTH_MUC_MALE = 0.2e-2; # [m] Depth of mucosa (0.2 [cm] in males and 0.15 [cm] in females)
DEPTH_LIG_MALE = 0.2e-2; # [m] Depth of ligament (0.2 [cm] in males and 0.15 [cm] in females)
DEPTH_MUS_MALE = 0.4e-2; # [m] Depth of TA muscle (0.4 [cm] in males and 0.3 [cm] in females)
      
LREST_FEMALE = 1.0e-2; # [m] Vocal fold length at rest for female subject
TREST_FEMALE = 0.2e-2; # [m] Vocal fold thickness at rest for female subject
DEPTH_MUC_FEMALE = 0.15e-2; # [m] Depth of mucosa (0.2 [cm] in males and 0.15 [cm] in females)
DEPTH_LIG_FEMALE = 0.15e-2; # [m] Depth of ligament (0.2 [cm] in males and 0.15 [cm] in females)
DEPTH_MUS_FEMALE = 0.3e-2; # [m] Depth of TA muscle (0.4 [cm] in males and 0.3 [cm] in females)

#           --- Incomplete Glottal Closure ---


#           --- Muscle Activation ---
# Retreived from file:///C:/Users/bglad/Downloads/17_1_online%20(1).pdf
a_TA = 0.899  # Thyroarytenoid
a_CT = 0.1  # Cricothyroid
a_LCA = 0.5  # Lateral Cricoarytenoid

def x_upper(z):
   return (x_u_0 + (u_displacement / L_g) * z) # is this inital x_u position?? refer eqn a17
# additionally whats the difference between the expressions of x_u(z) in A17 and A18

def x_lower(z):
   return (x_l_0 + (l_displacement / L_g) * z) # clairify this function

def fkuCol(z):
   return (-k_uCol * (x_upper(z) - + eta_uCol * (x_upper(z)) ** 3))

def fklCol(z):
   return (-k_lCol * (x_lower(z) + eta_lCol * (x_lower(z)) ** 3))

#           --- All the equations for F_u ----
F_ku = -k_u * ((u_displacement - b_displacement) + eta_u * (u_displacement - b_displacement) ** 3)
F_du = -2 * zeta_u * np.sqrt(m_u * k_u) * (x_u_d - x_b_d)
F_kc = -k_c * (l_displacement - u_displacement) 
# F_eu =  Not completed, talk to Dr. Peterson
F_kuCol = quad(fkuCol(z), L_u_closed, np.inf)[0] # extracts only integral result (not error)
# clairify that the upper bound is infinite
F_duCol = -2 * zetau_Col * alpha_u * np.sqrt(m_u * k_u)(x_u_d - x_b_d)

#           --- The equations for F_l ---
F_kl = -k_l * ((l_displacement - b_displacement) + eta_u * (l_displacement - b_displacement) ** 3)
F_dl = -2 * zeta_l * np.sqrt(m_l * k_l) * (x_l_d - x_b_d)
# F_kc already defined
F_el = T_l (P_s * (1 - alpha_l) * L_g - (P_s - P_l)(L_d + quad((x_upper(z) ** 2) / (x_lower(z) ** 2)), L_c, np.inf)) # clairify integral bounds
F_klCol = quad(fkuCol(z), L_l_closed, np.inf) # clairify upper bound
F_dlCol = - 2 * zetal_Col * alpha_l * np.sqrt(m_l * k_l) * (x_l_d - x_b_d)

#           --- The equations for F_b ---
F_kb = -k_b * (b_displacement + eta_b * (b_displacement) ** 3)
F_db = - 2 * zeta_b * np.sqrt(m_b * k_b) * (x_b_d)
# F_ku, F_du, F_kl, F_dl, F_duCol and F_dlCol defined

def dSdz(z, S):
   # m_u * x_u_dd = 
   return 0
   
# Upper mass
    q_u_d = (-k_u * (du + eta_u * du ** 3) 
             - 2*zeta_u*np.sqrt(m_u*k_u)*(q_u - q_b)
             + k_c * ((x_l_1 - x_l_0) - (x_u_1 - x_u_0))) / m_u
    
    # Lower mass
    q_l_d = (-k_l * (dl + eta_l * dl ** 3) 
             - 2 * zeta_l * np.sqrt(m_l * k_l) * (q_l - q_b)
             + k_c * ((x_l_1 - x_l_0) - (x_u_1 - x_u_0))) / m_l
    
    # Body mass
    q_b_d = (-k_b * (db + eta_b * db ** 3) 
             - 2 * zeta_b * np.sqrt(m_b * k_b)*q_b
             + k_u * (du + eta_u * du ** 3) 
             + 2 * zeta_u * np.sqrt(m_u * k_u) * (q_u - q_b)
             + k_l * (dl + eta_l * dl ** 3) 
             + 2 * zeta_l * np.sqrt(m_l * k_l) * (q_l - q_b)) / m_b

'''

import numpy as np
# https://numpy.org/doc/2.2/reference/generated/numpy.arange.html#numpy.arange

import matplotlib.pyplot as plt
# matplot crash course
# https://www.youtube.com/watch?v=3Xc3CA655Y4
# documentation ~ https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.pyplot.html

import scipy as sp

from scipy.integrate import odeint 
from scipy.integrate import solve_ivp
from scipy.integrate import quad
# scipy resources:
# https://www.youtube.com/watch?v=MM3cBamj1Ms
# https://github.com/lukepolson/youtube_channel/blob/main/Python%20Tutorial%20Series/odes1.ipynb

# sample langrangian mechanics problem
# https://www.youtube.com/watch?v=rJaXxb_piGI


import pandas as pd
# https://pandas.pydata.org/docs/


import sympy as smp # allows for symbolic manipulation

# Titze and Story (2002) https://watermark.silverchair.com/367_1_online.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAACAowgggGBgkqhkiG9w0BBwagggf3MIIH8wIBADCCB-wGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQMbQBDndSoHCeUCSLaAgEQgIIHvd5H8GVSXH69O9d_ZqF8vaFsygTMZj9IvMcoyz66Wz4QrwTUppaWwczoIf324Hb4JElmJkwwOWZDnEGd1uBcSJpzqFtHLy5GGHU3Y0hDxqe2Cu5tvipcfaBVDyjv1YWLdQSNlvX-8N8ZpSJRHYLdCP4ZbFj_vl_451pltlFe5TrFC3F-0lyICHFW6RHQqkzQGbYz_iXch3MOLFzco_-Ji-XC-PciuntRYUaf6VAuLOzZkbBl7OA1GvgHCptdOFJxITO2w-RowMjoQQNR_2IcVuUfONn2zBqG9qIym3_PnWyvuUfyK57MxzmbAaNDQuUyFLABraq1DNvxjE0usfyo3TcpgG-O96__2YQuBEgvYP_VWs7IsgfzOl4CLUEL0Rw_X2WcodpVjoXI_IKVl93EGLiH3vV-_1RAnIEyNVLughMeJ0tlYtfgTcCHlDU9M0bT_7vDZlL4nc3Zcq7jUa8MOqPrt5OI1IX48tyrk2zp1DwlpbONYl9gR8Yr-0fQ_KEbleqMtwz0kWn5oSnabf_559ewdxx_v5HENW8FneE2XnYKRrinuvCu1wghucuTRCaVaZKoTtnBcGldaDb04UupGwwD3V1WJimKYHupF_3ouGPl07VP8NSAQNbaa_m7WrNgWaFBO2d9zZipi7Nov3jHYKUTjrD8frFgv__FNjw3f-xmA-uvrCFhPR9KVz4qnlo5nPnB1I_SR8Eoqd6xO8XEJL1HeT5r3bBmE1Cly1TO-1K5A72kMo2aPTXaxAgIwG_FJtbnicF3T7svsVgsYBCmn-kMkUutsOESeNn6AAfjaCoeWaXKgPQRuhDiDbCS5996amJJN3Ppuqb1zBLw8iyXnGhFL41HU33qJ8Z38Rdwgu---8UhtUNNwtGdZvYuZGbhsjdZCI939fSnkndcAVcbRilWXVewko7RT9DPhTyOs8qILNDxbHrWRYpaAv4H73Asv9VcPse7dpdDbhg-cstcy5pPgMh4kdKTU1QTpoY3PmCEou0HgmTVRkS1uich0XRXQbTzgoFUUP_yfpPwpJJnC3_aFeSxe928h5IdrRkTzY58CqQL09gJtj6qjY1bOPJZ6vfiYm8p6fig1RD3ZUxKdQn7giyKcgBjdKXrRqANsTqfMIHEoLZYyRPzBfMfdoncv5SPxcOsTklNmZr2wLI0ucGajuYONP1Nu6HrBracK3wiB2y-VxZamgWMXOSOORnRp1zf8NW2XIkmQwJ-Wq14nhrLHyxegtTYvDoCUuF5dxQ9Jxxs8Pdo20MwOKJ3C-bITsfCY7J7ciVWtfKjWNs00WUVdR0_hzTrjgTweor7uKTXFfBc3pgN5qXI08mtLXAtUYbIc8ZOjZCKb84_nK9urn1U2vNhEOdLBeyY8VE5E1VDcyd-hgUl0zkCaMht40Gk_fKnl4qI9rBTsmj-LbnRF6AD3DUsD94EjY1be2kP4dpzv_oAGHBYO7BGsTmZX6_zAUV13kefubvs8Ptu0twi_VPhpE_PHQTnkMFcGTbqt0079R_SgorqYDDv82cd11XJFvsRBVkG3yZUMsCHF-6S26LkPg4c6YoV3uY7Bqgrk2FSoJr3kZBI365kFrxGT-7spoXJQ6fxkVGyWP5Kg3fQWd9-FKEpHOYqED24LugKlib2uEs6pdbJNBtHpyW7qHH9RdQNPczqi57KsbjuZWlYPOWvp4fFCg9Jzf9YlO9bO41nj9e_XmJc9OcQ4VV-bB2-pgcaWA-ajWSu_O4n7bRKJDXye9M-7n_W10dADdJzgm7fEbZt_zNdOa04aU288UCmLX5HXVBeki0eLi-YKy89Ch7lpszGbzHPFF6yVvNoCALmbOhWvtGdxwHM1D8B_m8u2z9JOUSnD4_Y3pNCb7IeIXuhQD3hbmhc0gjKmOzMA-UKtZMhrdTDaCUlt7ei5wYSKdAPTyoKQgqayDZl6oCPMapk4VL_o6kVnHL1NPvOPJDbUYfVA3g_vlPfcSS-5WeFJfyGNXyf5lyn2hFT1Ijw5NaiSAhYET9tZSDAA5Zzj8qJQ9cfCN8Cen1usg7ncd9wJrWkJZZ2LlpUsic6TzGOw7stqB0mNOWtAfI5V_bRsstQy1H_rbmNbiLLVeF4N9r9gPp96qEh_vAlmaeG3qKoXCkjyHkadDslszOQwbBHM8vkUhg3h5wbdBS_BouBlOWtmMAaYPmHgQM7uHBIDYS2P9IjAKcl0s_chPeWa8JasKeZnjYmKFue_hea9UVTQmlS3qY7g7Vf1afZ70ww3nPtw37QGYAvHivGGbA7kT8LcGXx9aVqbgYhM7bLuR2fSdL9cAZRrAKu_X8p4_VGYC6CRxlYopQPKtG13nTbTvlZIsmijuS8Ezo4bm0X3fqM7sxbJ6P8uDuZWSG5-C7GLwhVAuHfVFy_6MJ9mP955UHo2uD3OS9GVvwwCqw15bJ9BgX0i-GVvHDuuInjnKf8lkTPTqhnxe4dq4hhWs4RSjOlYOi_gNJJFYObOFsFGJMs6ryN752_gcnijADuQND0t9U-lwRp74JvsKWiRBJC-bMvPUgi_tk6fxspJ9m_YF-PNYtb2TGfpHz3ysrZNVSzaPkiZXbzKFC0GeZX1ykifDC_JxflySo4oL82-lcVWfF7GMceY55nfDsHtsAu12j7oE4


'''    if du != dl: # Avoid division by zero
      z_intersect = L_g * (x_l_1 - x_u_1) / (du - dl)
      
      # Check if intersection is within glottal length
      if z_intersect < L_g:
         L_d = z_intersect  # Divergent portion length of overlap (the length from 0 - z)
         L_c = L_g - z_intersect  # Convergent portion length (the total length - length from 0 - z )

         if (x_u_1 < x_l_1):
            Convergent = True
         else:
            Convergent = False

      else:
         # Fully divergent (intersection greater than length of the glottis)
         L_d = L_g
         L_c = 0
         Convergent = False

    else:
    # Case when du = dl
     if x_u_1 < x_l_1:
        L_c = L_g
        L_d = 0
        Convergent = True
     else:
        L_d = L_g
        L_c = 0
        Convergent = False


    def x_upper(z):
      return (x_u_0 + (du / L_g) * z) 

    def x_lower(z):
      return (x_l_0 + (dl / L_g) * z) # clairify this function
    
    def fkuCol(z):
        x_u_z = x_upper(z)
        if x_u_z < x_col:  # Checking if collision force exstence
            return k_u_col * (x_u_z - x_col) + eta_u_col * (x_u_z - x_col)**3
        return 0
    
    def fklCol(z):
        x_l_z = x_lower(z)
        if x_l_z < x_col: 
            return k_l_col * (x_l_z - x_col) + eta_l_col * (x_l_z - x_col)**3
        return 0




    if A_min > 0:
        P_u = P_s - (P_s - P_i) * (A_min / A_l) ** 2

        def pressure_integrand(z):
                x_u_z = x_upper(z)
                x_l_z = x_lower(z)
                if x_u_z <= 0 or x_l_z <= 0:
                    return 0
                return (x_u_z**2 / x_l_z**2)
       
       # Integrate pressure effect along convergent portion only (0 -> L_c)
        if L_c > 0:  # Only integrate if there's a convergent portion
          pressure_integral, _ = quad(pressure_integrand, 0, L_c)
        else:
          pressure_integral = 0
        
        # Integrate pressure effect along convergent portion only (0 -> L_c)
        F_eu = T_u * P_i * (1 - alpha_u) * L_g
        F_el = T_l * (P_s * (1 - alpha_l) * L_g - (P_s - P_i) * (L_d + pressure_integral))
    
    else: # When the glottis is closed, no pressure forces exist, so set them all to 0
       P_u = 0
       F_eu = 0
       F_el = 0

      # Calculate collision forces by integrating along the glottis
    # These represent the contact forces when masses collide with the midplane
   # F_ku_col, _ = quad(fkuCol, 0, L_c)
   # F_kl_col, _ = quad(fklCol, 0, L_c)
    F_ku_col = 0
    F_kl_col = 0
'''



'''
For clairty, I'm defining these hear
x_u_dd = None  # Second order derivative of position of upper mass wrt time
x_l_dd = None  # Second order derivative of position of lower mass wrt time
x_b_dd = None  # Second  order derivative of position of body mass wrt time
'''

'''
#           --- All the equations for F_u ----
F_ku = -k_u * ((u_displacement - b_displacement) + eta_u * (u_displacement - b_displacement) ** 3)
F_du = -2 * zeta_u * np.sqrt(m_u * k_u) * (x_u_d - x_b_d)
F_kc = -k_c * (l_displacement - u_displacement) 

#           --- The equations for F_l ---
F_kl = -k_l * ((l_displacement - b_displacement) + eta_u * (l_displacement - b_displacement) ** 3)
F_dl = -2 * zeta_l * np.sqrt(m_l * k_l) * (x_l_d - x_b_d)
# F_kc already defined

#           --- The equations for F_b ---
F_kb = -k_b * (b_displacement + eta_b * (b_displacement) ** 3)
F_db = - 2 * zeta_b * np.sqrt(m_b * k_b) * (x_b_d)
# F_ku, F_du, F_kl, F_dl, F_duCol and F_dlCol defined'



#           --- Creating 1st order differential equations ---

# Let q_u = d(x_u)/dt
q_u = x_u_d
q_u_d = (-k_u * ((u_displacement - b_displacement) + eta_u * (u_displacement - b_displacement) ** 3) + 
            (-2 * zeta_u * np.sqrt(m_u * k_u) * (q_u - q_b)) - 
            (-k_c * (l_displacement - u_displacement))) / m_u

# Let q_l = d(x_l)/dt
q_l = x_l_d
q_l_d = (-k_l * ((l_displacement - b_displacement) + eta_u * (l_displacement - b_displacement) ** 3) + 
            (-2 * zeta_l * np.sqrt(m_l * k_l) * (q_l - x_b_d)) + 
            (-k_c * (l_displacement - u_displacement))) / m_l

# Let q_b = d(x_b)/dt
q_b = x_b_d
q_b_d = (-k_b * (b_displacement + eta_b * (b_displacement) ** 3) + 
            (-2 * zeta_b * np.sqrt(m_b * k_b) * q_b) - 
            (-k_u * ((u_displacement - b_displacement) + eta_u * (u_displacement - b_displacement) ** 3)) + 
            (-2 * zeta_u * np.sqrt(m_u * k_u) * (q_u - q_b)) + 
            (-k_l * ((l_displacement - b_displacement) + eta_u * (l_displacement - b_displacement) ** 3)) + 
            -2 * zeta_l * np.sqrt(m_l * k_l) * (q_u - q_b)) / m_b


return [q_u,
        q_l, 
        q_b,  
        (-k_u * (((x_u_1 - x_u_0) - (x_b_1 - x_b_0)) + eta_u * ((x_u_1 - x_u_0) - (x_b_1 - x_b_0)) ** 3) + 
        (-2 * zeta_u * np.sqrt(m_u * k_u) * (q_u - q_b)) - 
        (-k_c * ((x_l_1 - x_l_0) - (x_u_1 - x_u_0)))) / m_u, 
        (-k_l * (((x_l_1 - x_l_0) - (x_b_1 - x_b_0)) + eta_l * ((x_l_1 - x_l_0) - (x_b_1 - x_b_0)) ** 3) + 
        (-2 * zeta_l * np.sqrt(m_l * k_l) * (q_l - q_b)) + 
        (-k_c * ((x_l_1 - x_l_0) - (x_u_1 - x_u_0)))) / m_l,
        (-k_b * ((x_b_1 - x_b_0) + eta_b * ((x_b_1 - x_b_0)) ** 3) + 
        (-2 * zeta_b * np.sqrt(m_b * k_b) * q_b) - 
        (-k_u * (((x_u_1 - x_u_0) - (x_b_1 - x_b_0)) + eta_u * ((x_u_1 - x_u_0) - (x_b_1 - x_b_0)) ** 3)) + 
        (-2 * zeta_u * np.sqrt(m_u * k_u) * (q_u - q_b)) + 
        (-k_l * (((x_l_1 - x_l_0) - (x_b_1 - x_b_0)) + eta_l * ((x_l_1 - x_l_0) - (x_b_1 - x_b_0)) ** 3)) + 
        -2 * zeta_l * np.sqrt(m_l * k_l) * (q_u - q_b)) / m_b]
'''
global Convergent



#           --- Biomechanical Parameters ---

m_u = 0.5  # Mass of upper block (m1)
m_l = 0.5  # Mass of lower block (m2)
m_b = 1.0  # Mass of body block (M)

k_u = 25  # [N/m] Spring constant for upper mass
k_l = 25  # [N/m] Spring constant for lower mass
k_c = 50  # [N/m] Spring constant for coupling between masses
k_b = 200 # [N/m] Spring constant for body cover mass

# Rest positions relative to midpoint line (symmetric system)

x_u_0 = 0.003 # [m] Initial position of the upper mass
x_l_0 = 0.002 # [m] Initial position of the lower mass
x_b_0 = 0.005  # [m] Initial position of the body mass

#           ---  Non-Linear Spring Coefficients ---
eta_u = 100*1e4  # [1/m^2] Non-linear spring coefficient for upper mass
eta_l = 100*1e4  # [1/m^2] Non-linear spring coefficient for lower mass
eta_b = 100*1e4  # [1/m^2] Non-linear spring coefficient for body mass

zeta_u = 0.1  # Basic damping factor for upper mass
zeta_l = 0.1  # Basic damping factor for lower mass
zeta_b = 0.1  # Damping factor for body mass

#           --- Collision Parameters ---
k_u_col = 25  # [N/m] Linear collision spring constant for upper mass
k_l_col = 25  # [N/m] Linear collision spring constant for lower mass
eta_u_col = 100*1e4  # [1/m^2] Non-linear collision spring constant for upper mass
eta_l_col = 100*1e4  # [1/m^2] Non-linear collision spring constant for lower mass

zeta_u_col = 0.5  # Collision damping factor for upper mass
zeta_l_col = 0.5  # Collision damping factor for lower mass
#          --- Glottal Parameters ---
a_t = 0.4 # [rad] Angle of the triangular glottis (maintained constant for now)
L_a = 2e-3 # [m] Arytenoid length 
L_g = 16e-2 # [m] Resting glottal length
PGD = 2 * (a_t + L_a * np.sin(a_t)) # Posterior glottal distance, kept constant for now
glottal_angle = 2 * np.arctan(PGD / L_g) # Since it's consant, works out to be ~ 2.7468
    
T_u = 3e-3 # [m] Thickness of upper layer 
T_l = 3e-3 # [m] Thickness of lower layer 
x_col = 0 # [m] Glottal-midline ~ used to turn on and off collision forces
Convergent = True # The lower block starts closer to the midplane line, leading to convergent flow

#        --- Pressures ---
P_s = 579 # [Pa] Subglottal pressure
P_i = 0 # needs to be adjusted - [Pa] Supraglottal pressure


def dSdx(t, S):
    x_u_1, x_l_1, x_b_1, q_u, q_l, q_b = S
    
    # Displacements relative to body mass
    du = (x_u_1 - x_u_0) - (x_b_1 - x_b_0)  # Upper mass displacement
    dl = (x_l_1 - x_l_0) - (x_b_1 - x_b_0)  # Lower mass displacement
    db = (x_b_1 - x_b_0)                   # Body mass displacement


    ''' area is enclosed if the masses travel mass the glottal midpoint line

    the x pos of the mass is given by:
            x(z) = current_pos + displacement/L_g * z

    setting x(z) to 0, we see
            0 = current_pos + displacement/L_g * z

    solving for z
            z = -current_pos * L_g / displacement
            if du != 0: # Avoid div by zero
        z_closed_u = -x_u_1 * (L_g / du) 
    else:
        z_closed_u = 0
    
    if dl != 0:
        z_closed_l = -x_l_1 * (L_g / dl)
    else:
        z_closed_l = 0
    '''

     



    

    
    # Use the smaller z_closed value to determine L_closed
    z_closed = min(z_closed_u, z_closed_l)
    
    # Ensure z_closed is within the valid range
    z_closed = min(1, max(0, min(z_closed, L_g)))
    
    # Calculate L_closed
    L_closed = L_g - z_closed

    # Propotionality constants for upper and lower masses
    alpha_u = L_closed / L_g 
    alpha_l = L_closed / L_g

    # Glottal areas (max to ensure non-negatives)
    A_u = max(0, (1 - alpha_u) * L_g * (x_u_1 + du))
    A_l = max(0, (1 - alpha_l) * L_g * (x_l_1 + dl))
    
    # Determine minimum area (used in pressure calculations)
    if min(A_u, A_l) > 0:
      A_min = min(A_u, A_l)
    else:
      A_min = 0
        
    
    
    # Calculate forces
    
    # Normal spring forces 
    F_ku = -k_u * (du + eta_u * du**3)  # Spring force for upper mass
    F_kl = -k_l * (dl + eta_l * dl**3)  # Spring force for lower mass
    F_kb = -k_b * (db + eta_b * db**3)  # Spring force for body mass
    F_kc = -k_c * ((x_l_1 - x_l_0) - (x_u_1 - x_u_0))  # Coupling spring force
    
    # Normal damping forces 
    F_du = -2 * zeta_u * np.sqrt(m_u * k_u) * (q_u - q_b)
    F_dl = -2 * zeta_l * np.sqrt(m_l * k_l) * (q_l - q_b)
    F_db = -2 * zeta_b * np.sqrt(m_b * k_b) * q_b
    
    # Activating Collision forces if and only if collision
    if alpha_u > 0:
        F_ku_col = -k_u_col * L_g * alpha_u * (x_u_1 + 0.5 * du + alpha_u) + eta_u_col * (x_u_1**3 + 1.5 * x_u_1 ** 2 * du * alpha_u + x_u_1 * du ** 2 + alpha_u ** 2 + 0.25 * du ** 3 * alpha_u ** 3)
    else:
        F_ku_col = 0
        
    if alpha_l > 0:
        F_kl_col = -k_l_col * L_g * alpha_l * (x_l_1 + 0.5 * dl + alpha_l) + eta_l_col * (x_l_1**3 + 1.5 * x_l_1 ** 2 * dl * alpha_l + x_l_1 * dl ** 2 + alpha_l ** 2 + 0.25 * dl ** 3 * alpha_l ** 3)
    else:
        F_kl_col = 0
    
    
    # Collision damping forces
    F_du_col = -2 * zeta_u_col * alpha_u * np.sqrt(m_u * k_u) * (q_u - q_b)
    F_dl_col = -2 * zeta_l_col * alpha_l * np.sqrt(m_l * k_l) * (q_l - q_b) 
    

    # Solving for inraglottal pressures
    if A_min > 0:

        P_u = P_s - (P_s - P_i) * (A_min / A_l) **2
        
        # Fluid forces on upper and lower masses
        F_eu = T_u * P_u * (1 - alpha_u) * L_g
        F_el = T_l * P_u * (1 - alpha_l) * L_g - (P_s - P_i) * (L_g + 0.5 * (x_l_1**2) / (L_g * a_t))

    else:
        F_eu = 0
        F_el = 0    

    
    # Total forces (Eqs. A1-A3)
    F_u = F_ku + F_du + F_kc + F_eu + F_ku_col + F_du_col
    F_l = F_kl + F_dl + F_kc + F_el + F_kl_col + F_dl_col
    #F_u = F_ku + F_du + F_kc + F_eu
    #F_l = F_kl + F_dl + F_kc + F_el
    F_b = F_kb + F_db - (F_ku + F_du + F_kl + F_dl) - (F_du_col + F_dl_col)
    
    # Accelerations
    q_u_d = F_u / m_u
    q_l_d = F_l / m_l
    q_b_d = F_b / m_b
    
    return [q_u, q_l, q_b, q_u_d, q_l_d, q_b_d]



S_0 = [x_u_0, x_l_0, x_b_0 + 1e-5, 0, 0, 0] 


# Create timesteps
#t = np.linspace(0, 10, 5000)

# Solve the system of ODEs

# Create timesteps within a 10 second interval with 5000 points
t = np.linspace(0, 2, 1000) 

# Solve the system of ODEs
sol = odeint(dSdx, y0=S_0, t=t, tfirst=True)

#print("Solver status:", sol.status)
#print("Solver message:", sol.message)
#print("Final time:", sol.t[-1])  # Should be close to tspan[1] (5 seconds)


# Extract solutions
x_u_sol = sol[:, 0]
x_l_sol = sol[:, 1]
x_b_sol = sol[:, 2]
v_u_sol = sol[:, 3]
v_l_sol = sol[:, 4]
v_b_sol = sol[:, 5]


# Displacements
plt.subplot(2, 1, 2)
plt.plot(t, x_u_sol, label='x_u (Upper Mass)')
plt.plot(t, x_l_sol, label='x_l (Lower Mass)')
plt.plot(t, x_b_sol, label='x_b (Body Mass)')
plt.xlabel('Time (s)')
plt.ylabel('Displacement (m)')
plt.legend()
plt.title('Displacement vs Time')

# Velocities
plt.subplot(2, 2, 2)
plt.plot(t, v_u_sol, label='v_u (Upper Mass)')
plt.plot(t, v_l_sol, label='v_l (Lower Mass)')
plt.plot(t, v_b_sol, label='v_b (Body Mass)')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.legend()
plt.title('Velocity vs Time')

plt.show()
'''
# Plot glottal area vs time
plt.subplot(2, 1, 2)
plt.plot(t, A_g * 1e6, label='Glottal Area')  
plt.xlabel('Time (s)')
plt.ylabel('Glottal Area (mm²)')
plt.title('Glottal Area vs Time')
plt.grid(True)

plt.tight_layout()
plt.show()


'''

'''
# Initialize the system
S_0 = [x_u_0, x_l_0, x_b_0 + 1e-3, 0, 0, 0] 

# Create 5000 timesteps within a 5 second interval
t = np.linspace(0, 10, 5000) 

# Solve the system of ODEs
sol = odeint(dSdx, y0=S_0, t=t, tfirst=True)

x_u_sol = sol[:, 0]
x_l_sol = sol[:, 1]
x_b_sol = sol[:, 2]

plt.plot(t, x_u_sol, label='x_u (Upper Mass)')
plt.plot(t, x_l_sol, label='x_l (Lower Mass)')
plt.plot(t, x_b_sol, label='x_b (Body Mass)')
plt.xlabel('Time (s)')
plt.ylabel('Displacement (m)')
plt.legend()
plt.title('Displacement vs Time for Upper, Lower, and Body Masses')
plt.show()


'''



    

