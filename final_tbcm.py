'''
The Triangular Body-Cover Model (TBCM) is a computational model used to simulate voice production, 
 in the context of vocal hyperfunction and phonotrauma. Expanding on the classic body-cover 
model (BCM), the TBCM incorporates a triangular glottal shape. This model allows for the simulation of
laryngal mechanisms. 

Coded by Benjamin Gladney
March 11th 2025

            --- References ---
[1] https://pmc.ncbi.nlm.nih.gov/articles/PMC5831616/
[2] https://pmc.ncbi.nlm.nih.gov/articles/PMC8727069/



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



#           --- Biomechanical Parameters ---


# Declare convergent as a global variable
global Convergent

m_u = 0.5  # Mass of upper block (m1) [100 mg]
m_l = 0.5  # Mass of lower block (m2) [100 mg]
m_b = 1.0  # Mass of body block (M) [250 mg]

k_u = 25  # [N/m] Spring constant for upper mass
k_l = 25  # [N/m] Spring constant for lower mass
k_c = 50  # [N/m] Spring constant for coupling between masses
k_b = 200 # [N/m] Spring constant for body cover mass

# Rest positions relative to midpoint line (symmetric system)

x_u_0 = 0.001 # [m] Initial position of the upper mass
x_l_0 = 0.003 # [m] Initial position of the lower mass
x_b_0 = 0.005  # [m] Initial position of the body mass

#           ---  Non-Linear Spring Coefficients ---
eta_u = 100e4  # [1/m^2] Non-linear spring coefficient for upper mass
eta_l = 100e4  # [1/m^2] Non-linear spring coefficient for lower mass
eta_b = 100e4  # [1/m^2] Non-linear spring coefficient for body mass

zeta_u = 0.1  # Basic damping factor for upper mass
zeta_l = 0.1  # Basic damping factor for lower mass
zeta_b = 0.1  # Damping factor for body mass

#           --- Collision Parameters ---
k_u_col = 25  # [N/m] Linear collision spring constant for upper mass
k_l_col = 25  # [N/m] Linear collision spring constant for lower mass
eta_u_col = 100*1e4  # [1/m^2] Non-linear collision spring constant for upper mass
eta_l_col = 100*1e4  # [1/m^2] Non-linear collision spring constant for lower mass

zeta_u_col = 0.4  # Collision damping factor for upper mass
zeta_l_col = 0.4  # Collision damping factor for lower mass
#          --- Glottal Parameters ---
a_t = 0.4 # [rad] Angle of the triangular glottis (maintained constant for now)
L_a = 2e-3 # [m] Arytenoid length 
L_g = 1.5e-2 # [m] Resting glottal length
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
    """
    Solving for positions and velocities of upper, lower, and body mass within the TBCM
    
    t: time
    S: state vector [x_u, x_l, x_b, v_u, v_l, v_b] where x is position v is velocity
    u -> upper mass, l-> lower mass, b-> body mass

    """
    x_u_1, x_l_1, x_b_1, v_u, v_l, v_b = S
    
    # Displacements relative to body mass
    du = (x_u_1 - x_u_0) - (x_b_1 - x_b_0)  # Upper mass displacement
    dl = (x_l_1 - x_l_0) - (x_b_1 - x_b_0)  # Lower mass displacement
    db = (x_b_1 - x_b_0)                    # Body mass displacement

    # Lengths of closed glottal portions - absolute value as positions are relative to midline (midline = 0 -> negative x means crossing midline)
    L_u_closed = abs(x_u_1) * np.tan(glottal_angle) if x_u_1 < 0 else 0
    L_l_closed = abs(x_l_1) * np.tan(glottal_angle) if x_l_1 < 0 else 0
    
    # Determine flow configuration (convergent or divergent)
    if L_l_closed > L_u_closed:
        L_c = L_l_closed - L_u_closed
        L_d = 0
        Convergent = True
    else:
        L_c = 0
        L_d = L_u_closed - L_l_closed
        Convergent = False
    
    # Porportionality constants for vocal fold posturing
    alpha_u = L_u_closed / L_g
    alpha_l = L_l_closed / L_g
    
    # Glottal areas 
    A_u = max(0, (1 - alpha_u) * L_g * (x_u_1 - du))
    A_l = max(0, (1 - alpha_l) * L_g * (x_l_1 - dl))
    
    # Minimum area (used in pressure calculations)
    A_min = min(A_u, A_l) if min(A_u, A_l) > 0 else 0
    
    # Expressing x positions as a function of z
    def x_upper(z):
        return x_u_0 + (du / L_g) * z
    
    def x_lower(z):
        return x_l_0 + (dl / L_g) * z
    
    # Normal spring forces
    F_ku = -k_u * (du + eta_u * du**3)  # Spring force for upper mass
    F_kl = -k_l * (dl + eta_l * dl**3)  # Spring force for lower mass
    F_kb = -k_b * (db + eta_b * db**3)  # Spring force for body mass
    F_kc = -k_c * ((x_l_1 - x_l_0) - (x_u_1 - x_u_0))  # Coupling spring force
    
    # Normal damping forces
    F_du = -2 * zeta_u * np.sqrt(m_u * k_u) * (v_u - v_b)
    F_dl = -2 * zeta_l * np.sqrt(m_l * k_l) * (v_l - v_b)
    F_db = -2 * zeta_b * np.sqrt(m_b * k_b) * v_b
    
    # Collision spring forces - triggered when mass crosses midline (x < 0)
    if alpha_u > 0:  # Upper mass collision occurred
        F_ku_col = -k_u_col * L_g * alpha_u * (x_u_1 + 0.5 * du + alpha_u) + \
                   eta_u_col * (x_u_1**3 + 1.5 * x_u_1**2 * du * alpha_u + \
                               x_u_1 * du**2 + alpha_u**2 + 0.25 * du**3 * alpha_u**3)
    else:
        F_ku_col = 0
        
    if alpha_l > 0:  # Lower mass collision occurred
        F_kl_col = -k_l_col * L_g * alpha_l * (x_l_1 + 0.5 * dl + alpha_l) + \
                   eta_l_col * (x_l_1**3 + 1.5 * x_l_1**2 * dl * alpha_l + \
                               x_l_1 * dl**2 + alpha_l**2 + 0.25 * dl**3 * alpha_l**3)
    else:
        F_kl_col = 0
    
    # Collision damping forces - also triggered by collision
    F_du_col = -2 * zeta_u_col * alpha_u * np.sqrt(m_u * k_u) * (v_u - v_b)
    F_dl_col = -2 * zeta_l_col * alpha_l * np.sqrt(m_l * k_l) * (v_l - v_b)
    
    # Aerodynamic forces from pressure differences
    if A_min > 0:  # Only if glottis is open
        #  find use for this
        P_u = P_s - (P_s - P_i) * (A_min / A_l)**2
        
        integral_result = 0
        
        # Pressure forces only exist in convergent conditions
        if L_c > 0 and Convergent:
            # Applying manual integration as built in was a little bugg ~ something to look into
            z_points = np.linspace(0, L_c, 10000)
            dz = z_points[1] - z_points[0]
            
            for z in z_points:
                integral_result += ((x_upper(z)**2) / (x_lower(z)**2)) * dz
            
        # Fluid forces on each mass
        F_eu = T_u * P_i * (1 - alpha_u) * L_g
        F_el = T_l * (P_s * (1 - alpha_l) * L_g - (P_s - P_i) * (L_d + integral_result))
    else:
        F_eu = 0
        F_el = 0
    
    # Total forces on each mass
    F_u = F_ku + F_du - F_kc + F_eu + F_ku_col + F_du_col
    F_l = F_kl + F_dl + F_kc + F_el + F_kl_col + F_dl_col
    F_b = F_kb + F_db - (F_ku + F_du + F_kl + F_dl) - (F_du_col + F_dl_col)
    
    # Throughout the calculatings, converted to first oder ODEs
    v_u_d = F_u / m_u
    v_l_d = F_l / m_l
    v_b_d = F_b / m_b
    
    return [v_u, v_l, v_b, v_u_d, v_l_d, v_b_d]

# Initial state [x_u, x_l, x_b, v_u, v_l, v_b]
S_0 = [x_u_0, x_l_0, x_b_0, 0.1, -0.1, 0.5] # small velocities to get the system moving


# Solve the system of ODEs

# Create timesteps within a 2 second interval with 5000 points
t = np.linspace(0, 2, 5000) 

# Solve the system of ODEs
sol = odeint(dSdx, y0=S_0, t=t, tfirst=True)


# Extract solutions
x_u_sol = sol[:, 0]
x_l_sol = sol[:, 1]
x_b_sol = sol[:, 2]
v_u_sol = sol[:, 3]
v_l_sol = sol[:, 4]
v_b_sol = sol[:, 5]

# Syntax derived completely from manual linked above

# Positions
plt.subplot(2, 1, 2)

plt.plot(t, x_u_sol, label='x_u (Upper Mass)')
plt.plot(t, x_l_sol, label='x_l (Lower Mass)')
plt.plot(t, x_b_sol, label='x_b (Body Mass)')
plt.axhline(y=0, color='k', linestyle='--', alpha=0.5, label='Midline')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.title('Mass Positions vs Time')
plt.legend()
plt.grid(True, alpha=0.3)

# Velocities
plt.subplot(2, 2, 2)
plt.plot(t, v_u_sol, label='v_u (Upper Mass)')
plt.plot(t, v_l_sol, label='v_l (Lower Mass)')
plt.plot(t, v_b_sol, label='v_b (Body Mass)')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.title('Velocity vs Time')
plt.legend()
plt.grid(True, alpha=0.3)

plt.show()