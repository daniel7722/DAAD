# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 11:19:38 2022

@author: s8913814


https://www.johannes-strommer.com/formeln/auflagerreaktionen-durchbiegung-winkel/
Feste Einspannung - Einzellast - Loslager

"""
#%%
import numpy as np
import random
import pickle

#input parameter
h = 10 
b = 5
l = 1000
E = 210000
F = 1000
a = 10
sigma_max = 200
#output
f = 0


def f(h,b,l,a,E,F,sigma_max):
    I = b*h**3/12
    W = I/(h/2)
    M_a = F*a*(l-a)/l*(a/(2*l)-1)
    sigma_b = M_a/W
    f = np.sin(F) * M_a
    if sigma_max < abs(sigma_b):
        return np.inf
    else:
        return f#,sigma_b


f(10,5,1000,100,210000,1000,200)

def create_random_para():
    h_bound = [20,40]
    b_bound = [5,30]
    l_bound = [1000]
    E_bound = [210000]
    F_bound = [5000]
    a_bound = [100,200]
    sigma_max_bound = [200]
    
    h = random.randint(min(h_bound),max(h_bound))
    b = random.randint(min(b_bound),max(b_bound))
    l = random.randint(min(l_bound),max(l_bound))
    E = random.randint(min(E_bound),max(E_bound))
    F = random.randint(min(F_bound),max(F_bound))
    a = random.randint(min(a_bound),max(a_bound))
    sigma_max = random.randint(min(sigma_max_bound),max(sigma_max_bound))
    
    return h,b,l,a,E,F,sigma_max

def create_list_load_substeps(F):
    n = 50
    list_return = [0]
    while max(list_return)<F:
        if random.random()<0.1:
            list_return.append(list_return[-1]+F/n + random.random()*500)
        else:
            list_return.append(round(list_return[-1]+F/n))
    list_return[-1] = F
    return list_return


list_input = []
list_output = []
for i in range(50):
    h,b,l,a,E,F,sigma_max = create_random_para()
    list_input.append([h,b,l,a,E,F,sigma_max])
    _F = create_list_load_substeps(F)
    _out = []
    for _ in _F:
        _out.append(f(h,b,l,a,E,_,sigma_max))
    list_output.append([_F,_out])


with open(r"802_data.pkl","wb") as file:
    pickle.dump(list_input,file)
    pickle.dump(list_output,file)
    
    
    
    
#os.chdir(r"C:\LokaleDaten\s8913814\Documents\03_Studenten\03_DAAD_Huang")
with open(r"802_data.pkl","rb") as file:
    list_input  = pickle.load(file)
    list_output = pickle.load(file)
    
