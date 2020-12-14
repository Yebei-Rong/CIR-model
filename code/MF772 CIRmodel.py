#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 21:19:54 2020

@author: Huitong Li
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as si
import math
from datetime import datetime
import xlrd
from sklearn.linear_model import LinearRegression
from scipy import stats
from scipy.optimize import minimize

length = 60
citigroupCDS = pd.read_excel('/Users/samantha/Desktop/MF772/CitigroupCDS.xlsx')
citigroupCDS = citigroupCDS.iloc[::-1]
UStrea = pd.read_excel('/Users/samantha/Desktop/MF772/US treasury.xlsx')
UStrea = UStrea.iloc[::-1]
citi60 = citigroupCDS[len(citigroupCDS)-length:].iloc[:,1]/10000
UStrea60 = list(UStrea[len(UStrea)-length:].iloc[:,2]/100)
citiCDS_diffT = pd.read_excel('/Users/samantha/Desktop/MF772/CDS spread compared diff matur.xlsx')
citiCDS_diffT = citiCDS_diffT.iloc[::-1]
recover = 0.4
citi_hazard_rate_diff = [list(citiCDS_diffT[len(citiCDS_diffT)-length:].iloc[:,i]/10000/(1-recover)) for i in range(1,6)]
test = list(UStrea[len(UStrea)-length:]['Date'])
#approximation to hazard rate by S = hazard_rate/(1-recover)
hazard_rate60 = list(citi60/(1-recover))

def rf_calibr(data):
    deltaT=0.1
    theta_est_lr = [0,0,0]
    n = len(data)
    y = np.array(data[1:])
    x = np.array(data[:n-1]).reshape(-1, 1)
    reg = LinearRegression().fit(x, y)
    slope = reg.coef_[0]
    intercept = reg.intercept_
    residuals = y-reg.predict(x)
    theta_est_lr[0] = intercept/deltaT
    theta_est_lr[1] = (1.0 - slope)/deltaT
    theta_est_lr[2] = np.std(residuals/np.sqrt(x))/(deltaT**0.5)
    theta1, theta2, theta3 = theta_est_lr[0],theta_est_lr[1],theta_est_lr[2]#0.042829665,2.212663284,0.009564792#1.0639895,0.3306534,0.1025292
    alpha = theta1
    mu = theta1/theta2 #theta2 
    sigma = theta3
    return [alpha, mu, sigma]
def hazard(r0,t_init,t_end,dt,mu,sigma):
    np.random.seed(5)
    ts = np.arange(t_init, t_end, dt)
    dW = np.random.normal(loc=0.0, scale=np.sqrt(dt),size = ts.size)
    r = []
    r.append(r0)
    for i in range(1, ts.size):
        r.append(r[i-1] + (mu)*dt+ sigma*dW[i-1])
    return r

def hazard_calibr(data):
    guess=[0.001,0.005,0.05]
    def spe(guess,data):
        mu=guess[0]
        sigma=guess[1]
        r0=data[0]
        # For Euler method
        sim=hazard(r0,0,len(data)/252,1/252,mu,sigma)
        temp= np.array(sim)-np.array(data) #list(set(sim)-set(data))
        sse=sum(temp*temp)
        return sse
    res = minimize(spe,guess,args=data,method='SLSQP')
    return res

#simulation CIR process
class CIR(object):
    def __init__(self, alpha, mu, sigma, r0):
         self.alpha = alpha # mean-reverted speed
         self.mu = mu # mean-reverted level
         self.sigma = sigma # vol of interest rate
         self.r0 = r0 # initial interest rate
        
    def Euler(self,t_init,t_end,dt):
        np.random.seed(5)
        ts = np.arange(t_init, t_end, dt)
        dW = np.random.normal(loc=0.0, scale=np.sqrt(dt),size = ts.size)
        r = []
        r.append(self.r0)
        for i in range(1, ts.size):
            r.append(r[i-1] + self.alpha*(self.mu-r[i-1])*dt 
                     + self.sigma*np.sqrt(r[i-1])*dW[i-1])
        return r
    
    def Milstein(self,t_init,t_end,dt):
        np.random.seed(5)
        ts = np.arange(t_init, t_end, dt)
        dW = np.random.normal(loc=0.0, scale=np.sqrt(dt),size = ts.size)
        r = []
        r.append(self.r0)
        for i in range(1, ts.size):
            dw = dW[i-1]
            r.append(r[i-1] + self.alpha*(self.mu-r[i-1])*dt 
                     + self.sigma*np.sqrt(r[i-1])*dw + 
                     (self.sigma**2/2)*(dw**2 - dt))
        return r
    
N = 60#len(UStrea60)

rs = []
hs = []
N_rol = 5
CIR_par = []
N_rol1 = 20
for i in range(N-N_rol):
    UStre = UStrea60[i:N_rol+i]
    citih = hazard_rate60[i:N_rol+i]
    alpha, mu, sigma = rf_calibr(UStre)
    CIR_par.append([alpha, mu, sigma])
#    mu2,sigma2 = hazard_calibr(citih).x[:2]
#    C = CIR(alpha, mu, sigma,UStre[-1])
#    r = C.Euler(0,2/252,1/252)[0]
#    Ht = hazard(citih[-1],0,2/252,1/252,mu2,sigma2)[0]
#    rs.append(r)
#    hs.append(Ht)
alpha = []
mu = []
sigma = []
'''-------------------------------------------------------------------------------------'''
hs_all = []
para_all = []
for i in np.arange(0,N,N_rol):
    UStre = UStrea60[i:i+N_rol]
    alpha, mu, sigma = rf_calibr(UStre)
#    CIR_par.append([alpha, mu, sigma])
    C = CIR(alpha, mu, sigma,UStre[0])
    r = C.Euler(0,N_rol/252,1/252)
    rs.append(r)
for j in range(len(citi_hazard_rate_diff)):
    hs = []
    CIR_par = []
    for i in np.arange(0,N,N_rol):
        citi_CDS = citi_hazard_rate_diff[j][i:i+N_rol]
        alpha1, mu1, sigma1 = rf_calibr(citi_CDS)
        CIR_par.append([alpha1, mu1, sigma1])
        C1 = CIR(alpha1, mu1, sigma1,citi_CDS[0])
        Ht = C1.Euler(0,N_rol/252,1/252)
        hs.append(Ht)
    hs_all.append(hs)
    para_all.append(CIR_par)
column_name =['5yr','4yr','3yr','2yr','1yr']
a = np.array([np.array(hs_all[i]).reshape(1,60)[0] for i in range(5)])
b = pd.DataFrame(columns = column_name)
for i in range(5):
    b[column_name[i]] = np.array([np.array(hs_all[i]).reshape(1,60)[0] for i in range(5)])[i]
def_bond_all = []
for j in range(1,6):
    def_bond = np.array([np.exp(-(rs[i]+b[column_name[j-1]][i])*(6-j)) for i in range(len(hs))])
    def_bond_all.append(def_bond)
c = np.array([np.array(def_bond_all[i]).reshape(1,60)[0] for i in range(5)])
d = pd.DataFrame(columns = column_name)
for i in range(5):
    d[column_name[i]] = np.array([np.array(def_bond_all[i]).reshape(1,60)[0] for i in range(5)])[i]

#alp = 
#.reshape(12,5),columns = column_name
alpha_change = pd.DataFrame(np.array([np.array(para_all[i]).reshape(3,12)[0] for i in range(5)]))
mu_change = pd.DataFrame(np.array([np.array(para_all[i]).reshape(3,12)[1] for i in range(5)]))
sigma_change = pd.DataFrame(np.array([np.array(para_all[i]).reshape(3,12)[2] for i in range(5)]))
e = pd.DataFrame(para_all)
hazard_rate_and_def_bond = pd.concat([b,d],axis=1)
CIR_para_change = pd.concat([alpha_change,mu_change,sigma_change],axis=0)
hazard_rate_and_def_bond.index = test
period = np.arange(4,N,N_rol)
CIR_para_change.index = [test[i] for i in period]

hazard_rate_and_def_bond.to_excel('/Users/samantha/Desktop/MF772/hazard_rate_and_def_bond12.13.xlsx')
CIR_para_change.to_excel('/Users/samantha/Desktop/MF772/CIR_para_change12.13.xlsx')
'''-------------------------------------------------------------------------------------'''
for i in range(1,6):
    plt.plot(c[i])
#for j in range(len(citi_hazard_rate_diff)):
#    def_bond = np.array([np.exp(-(rs[i]+hs_all[j][i])) for i in range(len(hs))])
#    def_bond_all.append(def_bond)
#for i in range(len(CIR_par)):
#    alpha.append(CIR_par[i][0])
#    mu.append(CIR_par[i][1])
#    sigma.append(CIR_par[i][2])
#CIR_par1 = []
#
#for i in range(N-N_rol):
##    UStre = UStrea60[i:N_rol+i]
#    citih = hazard_rate60[i:N_rol+i]
#    alpha1, mu1, sigma1 = rf_calibr(citih)
#    CIR_par1.append([alpha1, mu1, sigma1])
#    
alpha1 = []
mu1 = []
sigma1 = []
#for i in range(len(CIR_par1)):
#    alpha1.append(CIR_par1[i][0])
#    mu1.append(CIR_par1[i][1])
#    sigma1.append(CIR_par1[i][2])
CIR_par = []
CIR_par1 = []
for i in np.arange(0,N,N_rol):
    UStre = UStrea60[i:i+N_rol]
    alpha, mu, sigma = rf_calibr(UStre)
    citi_CDS = hazard_rate60[i:i+N_rol]
    alpha1, mu1, sigma1 = rf_calibr(citi_CDS)
    CIR_par.append([alpha, mu, sigma])
    CIR_par1.append([alpha1, mu1, sigma1])
alpha1 = []
mu1 = []
sigma1 = []
alpha = []
mu = []
sigma = []
for i in range(len(CIR_par1)):
    alpha.append(CIR_par[i][0])
    mu.append(CIR_par[i][1])
    sigma.append(CIR_par[i][2])
    alpha1.append(CIR_par1[i][0])
    mu1.append(CIR_par1[i][1])
    sigma1.append(CIR_par1[i][2])

    
for i in np.arange(0,N,N_rol):
    UStre = UStrea60[i:i+N_rol]
    alpha, mu, sigma = rf_calibr(UStre)
    citi_CDS = hazard_rate60[i:i+N_rol]
    alpha1, mu1, sigma1 = rf_calibr(citi_CDS)
#    CIR_par.append([alpha, mu, sigma])
    C = CIR(alpha, mu, sigma,UStre[0])
    C1 = CIR(alpha1, mu1, sigma1,citi_CDS[0])
    r = C.Euler(0,N_rol/252,1/252)
    Ht = C1.Euler(0,N_rol/252,1/252)
    rs.append(r)
    hs.append(Ht)
    
#for i in np.arange(0,N,20):
#    citih = hazard_rate60[i:20+i]
#    alpha1, mu1, sigma1 = rf_calibr(citi_CDS)
#    C1 = CIR(alpha1, mu1, sigma1,citi_CDS[0])
#    Ht = C1.Euler(0,20/252,1/252)
#    hs.append(Ht)
    
for i in np.arange(0,N,20):
    citih = hazard_rate60[i:20+i]
    mu2,sigma2 = hazard_calibr(citih).x[:2]
    Ht = hazard(citih[0],0,20/252,1/252,mu2,sigma2)
    hs.append(Ht)
rs = np.array(rs).reshape(1,60)[0]
hs = np.array(hs).reshape(1,60)[0]

def_bond = np.array([np.exp(-(rs[i]+hs[i])) for i in range(len(hs))])
real_def_bond = np.array([np.exp(-(UStrea60[i]+hazard_rate60[i])) for i in range(len(hazard_rate60))])
#citih = hazard_rate60
#mu2,sigma2 = hazard_calibr(citih).x[:2]
#hs = hazard(citih[0],0,60/252,1/252,mu2,sigma2)
#alpha, mu, sigma = rf_calibr(UStrea60)
#mu2,sigma2 = hazard_calibr(hazard_rate60).x[:2]
#C = CIR(alpha, mu, sigma,UStrea60[-1])
#r = C.Euler(0,2/252,1/252)[1]
#H = hazard(hazard_rate60[-1],0,2/252,1/252,mu2,sigma2)


plt.plot(rs)
plt.plot(UStrea60)#[N-50:]
plt.plot(hs)
plt.plot(hazard_rate60)
plt.plot(def_bond)
plt.plot(real_def_bond)
error = real_def_bond-def_bond
plt.plot(abs(error))
compara_data = pd.DataFrame()
compara_data['estimated interest rate'] = rs
compara_data['US treasury yield'] = UStrea60
compara_data['estimated hazard rate'] = hs
compara_data['hazard rate calculated by Citigroup CDS'] = hazard_rate60
compara_data['estimated defaultable zero coupon bond price'] = def_bond
compara_data['defaultable zero coupon bond price'] = real_def_bond
compara_data.index = test['Date']
compara_data.to_csv('/Users/samantha/Desktop/MF772/compara_data_citigroup.csv')

data2 = pd.DataFrame()
data2['interest rate'] = rs
data2['hazard rate'] = hs
data2['defaultable bond']=def_bond
data2.index = test['Date']

data2.to_csv('/Users/samantha/Desktop/MF772/citigroup_data2.csv')

#para = pd.DataFrame()
#para['alpha'] = alpha
#para['mu'] = mu
#para['sigma']=sigma
#para.index = test['Date'][5:]
#para.to_csv('/Users/samantha/Desktop/MF772/CIR_parameter.csv')

para = pd.DataFrame()
period = np.arange(4,N,N_rol)
para['alpha'] = alpha
para['mu'] = mu
para['sigma']=sigma
para['real risk free'] = [UStrea60[i] for i in period]
para['alpha1'] = alpha1
para['mu1'] = mu1
para['sigma1']=sigma1
para['real hazard rate'] = [hazard_rate60[i] for i in period]
para.index = [test[i] for i in period]
para.to_csv('/Users/samantha/Desktop/MF772/CIR_parameter2.csv')
#
plt.plot(alpha1)
print(max(alpha1),min(alpha1),max(alpha1)-min(alpha1))
plt.plot(mu1)
print(max(mu1),min(mu1),max(mu1)-min(mu1))
plt.plot(sigma1)
print(max(sigma1),min(sigma1),max(sigma1)-min(sigma1))
plt.plot([hazard_rate60[i] for i in period])
plt.plot(alpha)
plt.plot(mu)
plt.plot(sigma)
plt.plot([UStrea60[i] for i in period])
print(max(alpha),min(alpha),max(alpha)-min(alpha))
print(max(mu),min(mu),max(mu)-min(mu))
print(max(sigma),min(sigma),max(sigma)-min(sigma))
plt.figure()
l1,=plt.plot(def_bond)
l2,=plt.plot(real_def_bond)
plt.ylabel('def bond price',fontsize = 10)
plt.legend(handles=[l1, l2], labels=['estimtae','real'], fontsize = 10)







#for calibration
Data = pd.read_excel('/Users/samantha/Desktop/MF772/citigroupbond.xlsx')
Data = Data.iloc[::-1]
realreturn = list(Data.iloc[:10,2]/100)
theta1, theta2, theta3 = 0.062601034,3.228052898,0.006406258#0.042829665,2.212663284,0.009564792#1.0639895,0.3306534,0.1025292
alpha = theta1
mu = theta1/theta2 #theta2 
sigma = theta3
C = CIR(alpha, mu, sigma,0.01914)
# For Euler method
r = C.Euler(0,1/252,1/252) # start time = 0, end time = 1, dt = 0.01
# For Milstern method
r1 = C.Milstein(0,1/252,1/252)
# Plot and compare
plt.figure(figsize=(20,10))
l1,=plt.plot(r)
l2,=plt.plot(r1,'b*',markersize=4)
l3,=plt.plot(realreturn)
plt.ylabel('interest rate r',fontsize = 10)
plt.legend(handles=[l1, l2], labels=['Euler', 'Milstein','real'], fontsize = 10)

############################
theta_est_lr[0] = intercept/deltaT
theta_est_lr[1] = (1.0 - slope)/deltaT
theta_est_lr[2] = np.std(residuals/np.sqrt(x))/(deltaT**0.5)
#realreturn = list(Data.iloc[:10,2]/100)
deltaT=0.1
#using linear regression for calibration 
theta_est_lr = [0,0,0]
n = len(realreturn)
y = np.array(realreturn[1:])
x = np.array(realreturn[:n-1]).reshape(-1, 1)
reg = LinearRegression().fit(x, y)
slope = reg.coef_[0]
intercept = reg.intercept_
residuals = y-reg.predict(x)
np.std(residuals/np.sqrt(x))/(deltaT**0.5)

theta_est_lr[0] = intercept/deltaT
theta_est_lr[1] = (1.0 - slope)/deltaT
theta_est_lr[2] = np.std(residuals/np.sqrt(x))/(deltaT**0.5)
