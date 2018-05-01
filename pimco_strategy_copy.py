# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 23:26:06 2018

@author: rbgud
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 16:44:39 2018

@author: 이규형
"""

import pandas as pd
import numpy as np
import scipy as sp
import scipy.stats as sps
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.optimize import minimize

plt.close('all')
direc="C:/Users/mjh/Documents/commodity_pimco/"

oilprice = pd.read_csv(direc+'oilprice.csv',header =0 ,encoding='CP949',engine='python').set_index('Date',drop=True)[:-2]
oilprice.index = pd.to_datetime(oilprice.index,format='%Y-%m-%d') 
goldprice = pd.read_csv(direc+'goldprice.csv',header =0,encoding='CP949',engine='python').set_index('Date',drop=True)[:-2]
goldprice.index = pd.to_datetime(goldprice.index,format='%Y-%m-%d') 

oilcap = pd.read_csv(direc+'oilcap.csv',header =0,encoding='CP949',engine='python').set_index('Date',drop=True)[:-2]
oilcap.index = pd.to_datetime(oilcap.index,format='%Y-%m-%d') 
goldcap = pd.read_csv(direc+'goldcap.csv',header =0,encoding='CP949',engine='python').set_index('Date',drop=True)[:-2]
goldcap.index = pd.to_datetime(goldcap.index,format='%Y-%m-%d') 

origin = pd.read_csv(direc+'origin2.csv',header =0,encoding='CP949',engine='python').set_index('Date',drop=True)[:-2]
origin.index = pd.to_datetime(origin.index,format='%Y-%m-%d') 


pet_CI = pd.DataFrame(origin['SPXGS_PTRM']).reset_index(drop=True)


gold_CI = pd.DataFrame(origin['SPXGS_GTRM']).reset_index(drop=True)

Spx = pd.DataFrame(origin['S&P500']).reset_index(drop=True)[:235]

rf = pd.DataFrame(origin['LIBOR_3M'].fillna(0)/1200)[:235].reset_index(drop=True)
#%% Gold Index 모멘텀 스코어 

#모멘텀 1~12개월 롤링 스코어 (골드)
M_gold = pd.DataFrame(np.zeros((235,12)))
for j in range(12):
    for i in range(j+1,len(gold_CI)):
        if gold_CI.loc[i].values-gold_CI.loc[i-j-1].values>0:
           M_gold.iloc[i,j]=1
        else:
           M_gold.iloc[i,j]=0
       

#모멘텀 1~12개월 롤링 스코어 (오일)
M_oil = pd.DataFrame(np.zeros((235,12)))
for j in range(12):
    for i in range(j+1,len(gold_CI)):
        if pet_CI.loc[i].values-pet_CI.loc[i-j-1].values>0:
           M_oil.iloc[i,j]=1
        else:
           M_oil.iloc[i,j]=0      
           
aa = M_gold.sum(axis=1)  

bb = M_oil.sum(axis=1)  

cc = pd.concat([aa,bb],axis=1)    

cc['w_gold'] = (cc[0]/(cc[0]+cc[1])).fillna(0.5)   

cc['w_oil'] =(cc[1]/(cc[0]+cc[1])).fillna(0.5)

w = cc.loc[:235]
      
#       
w_gold = (M_gold.mean(axis=0).sum()/12)/(M_gold.mean(axis=0).sum()/12+M_oil.mean(axis=0).sum()/12)  
w_oil = (M_oil.mean(axis=0).sum()/12)/(M_gold.mean(axis=0).sum()/12+M_oil.mean(axis=0).sum()/12)  


wg = pd.DataFrame(w['w_gold'])      
wo = pd.DataFrame( w['w_oil'])

ret_goldCI = (gold_CI/gold_CI.shift(1)-1)   
ret_petCI = (pet_CI/pet_CI.shift(1)-1)

#ret_CI_PF = ret_goldCI.values*w_gold+ret_petCI.values*w_oil
ret_CI_PF =ret_goldCI.values*wg.values+ret_petCI.values*wo.values

ret_CI_PF[0]=0

ret_CI_PF = pd.DataFrame(ret_CI_PF)[:235]

#%%

oil_ret = (oilprice/oilprice.shift(1)-1)
gold_ret = (goldprice/goldprice.shift(1)-1)

oil_weightsumM = oilcap.sum(axis=1)

gold_weightsumM = goldcap.sum(axis=1)

oil_MonthlyW = pd.DataFrame()
gold_MonthlyW = pd.DataFrame()

for i in range(235):
    W=oilcap[i:i+1]/oil_weightsumM[i]
    oil_MonthlyW=pd.concat([oil_MonthlyW,W])
for i in range(235):
    W=goldcap[i:i+1]/gold_weightsumM[i]
    gold_MonthlyW=pd.concat([gold_MonthlyW,W])

## NRE 구성
vw_oilret = pd.DataFrame((oil_ret*oil_MonthlyW.values).sum(axis=1))
vw_goldret = pd.DataFrame((gold_ret*gold_MonthlyW.values).sum(axis=1))

nre = vw_oilret*0.5+vw_goldret*0.5

#Spx_ret = (-1)*Spx.pct_change(-1)
Spx_ret = (Spx/Spx.shift(1)-1)


Ci=ret_CI_PF[1:]#x1
Spx_Index=Spx_ret[1:]   #x2
nre=pd.DataFrame(nre.values)[1:]


#sharpe_PF = (ret_CI_PF-rf.values).mean()*12/(ret_CI_PF.std()*np.sqrt(12))

#Rolling 116 window - vw_vw_nre
def reg_m(y, x):
    ones = np.ones(len(x[0]))
    X = sm.add_constant(np.column_stack((x[0], ones)))
    for ele in x[1:]:
        X = sm.add_constant(np.column_stack((ele, X)))
    results = sm.OLS(y, X).fit()
    return results

numberOfwindow = 114
numberOfmonthIn10Y=120

ret_goldCI = ret_goldCI[121:]
ret_petCI = ret_petCI[121:]

#################################################
####### VW CI
#################################################

text_file = open("Mimic_PF_result.txt", "w")
Mimic_PF_result=pd.DataFrame(index=np.arange(0, numberOfwindow), columns=('x1', 'x2','c') )
for i in range(numberOfwindow):
    x = []
    x.append(Ci[0+i:120+i])
    x.append(Spx_Index[0 + i:120+i])
    result = reg_m(nre[0 + i:120+i], x)
    text_file.write(result.summary().as_text())
    Mimic_PF_result.loc[i] = [result.params[0], result.params[1], result.params[2]]
text_file.close()

Ci=pd.DataFrame(Ci[120:]).reset_index(drop=True)
Spx_Index = pd.DataFrame(Spx_Index[120:]).reset_index(drop=True)
rf = pd.DataFrame(rf[121:].values).reset_index(drop=True)
x1 =pd.DataFrame(Mimic_PF_result['x1'])
x2 = pd.DataFrame(Mimic_PF_result['x2'])
nre=pd.DataFrame(nre[120:])

Mimic_PF_ret=pd.DataFrame(Ci.values*x1.values+Spx_Index.values*x2.values+rf*(1-x1.values-x2.values))
excess_ret=pd.DataFrame(nre.values-Mimic_PF_ret.values)

sharpe_PF = (Mimic_PF_ret-rf.values).mean()*12/(Mimic_PF_ret.std()*np.sqrt(12))

sharpe_nre = (nre-rf.values).mean()*12/(nre.std().mean()*np.sqrt(12))


print('sh_PF= %.2f' %(sharpe_PF))
print('sh_NRE= %.2f' %(sharpe_nre))
print('-----------------------------------------')
#################################################
####### Equally Weighted GOLD NRE
#################################################
ew_gold_nre=gold_ret.mean(axis=1)[121:]
EW_GOLD_result=pd.DataFrame(index=np.arange(0, numberOfwindow), columns=('x1', 'x2','c') )
for i in range(numberOfwindow):
    x = []
    x.append(Ci[0+i:120+i])
    x.append(Spx_Index[0 + i:120+i])
    result = reg_m(ew_gold_nre[0 + i:120+i], x)
    EW_GOLD_result.loc[i] = [result.params[0], result.params[1], result.params[2]]
    
x1 =pd.DataFrame(EW_GOLD_result['x1'])
x2 = pd.DataFrame(EW_GOLD_result['x2'])
ew_gold_nre = pd.DataFrame(ew_gold_nre).values
EW_GOLD_result=pd.DataFrame(ret_goldCI.values*x1.values+Spx_Index.values*x2.values+rf*(1-x1.values-x2.values))
#excess_ret_1=pd.DataFrame(ew_gold_nre.values-EW_GOLD_result.values)

sharpe_PF_1 = (EW_GOLD_result-rf.values).mean()*12/(EW_GOLD_result.std()*np.sqrt(12))
sharpe_nre_1 = (ew_gold_nre-rf.values).mean()*12/(ew_gold_nre.std().mean()*np.sqrt(12))

print('sh_PF_EW_GOLD= %.2f' %(sharpe_PF_1))
print('sh_NRE_EW_GOLD= %.2f' %(sharpe_nre_1))
print('-----------------------------------------')
#################################################
####### Value Weighted TOTAL NRE
#################################################
vw_oil=pd.DataFrame()
vw_gold=pd.DataFrame()

for i in range(235):
    r=oilcap[i:i+1]/oilcap.sum(axis=1)[i]
    vw_oil=pd.concat([vw_oil,r])
for i in range(235):
    r=goldcap[i:i+1]/goldcap.sum(axis=1)[i]
    vw_gold=pd.concat([vw_gold,r])


oil_vw_nre = vw_oil.multiply(oil_ret).sum(axis=1)
gold_vw_nre = vw_gold.multiply(gold_ret).sum(axis=1)
total_vw_nre=list(pd.Series((oil_vw_nre*(54.68/59.41)).values+(gold_vw_nre*(4.73/59.41)).values))[121:]

vw_Ci=pd.DataFrame((ret_petCI*(54.68/59.41)).values+(ret_goldCI*(4.73/59.41)).values)

VW_TOTAL_result=pd.DataFrame(index=np.arange(0, numberOfwindow), columns=('x1', 'x2','c') )
for i in range(numberOfwindow):
    x = []
    x.append(vw_Ci[0+i:120+i])
    x.append(Spx_Index[0 + i:120+i])
    result = reg_m(total_vw_nre[0 + i:120+i], x)
    VW_TOTAL_result.loc[i] = [result.params[0], result.params[1], result.params[2]]
    
x1 =pd.DataFrame(VW_TOTAL_result['x1'])
x2 = pd.DataFrame(VW_TOTAL_result['x2'])
total_vw_nre = pd.DataFrame(total_vw_nre).values
VW_TOTAL_result=pd.DataFrame(vw_Ci.values*x1.values+Spx_Index.values*x2.values+rf*(1-x1.values-x2.values))

sharpe_PF_2 = (VW_TOTAL_result-rf.values).mean()*12/(VW_TOTAL_result.std()*np.sqrt(12))
sharpe_nre_2 = (total_vw_nre-rf.values).mean()*12/(total_vw_nre.std().mean()*np.sqrt(12))

print('sh_PF_VW_TOTAL= %.2f' %(sharpe_PF_2))
print('sh_NRE_VW_TOTAL= %.2f' %(sharpe_nre_2))
print('-----------------------------------------')

#################################################
####### Value Weighted GOLD NRE
#################################################
vw_gold=pd.DataFrame()
for i in range(235):
    r=goldcap[i:i+1]/goldcap.sum(axis=1)[i]
    vw_gold=pd.concat([vw_gold,r])
gold_vw_nre = vw_gold.multiply(gold_ret).sum(axis=1)[121:]

VW_GOLD_result=pd.DataFrame(index=np.arange(0, numberOfwindow), columns=('x1', 'x2','c') )
for i in range(numberOfwindow):
    x = []
    x.append(ret_goldCI[0+i:120+i])
    x.append(Spx_Index[0 + i:120+i])
    result = reg_m(gold_vw_nre[0 + i:120+i], x)
    VW_GOLD_result.loc[i] = [result.params[0], result.params[1], result.params[2]]
    
x1 =pd.DataFrame(VW_GOLD_result['x1'])
x2 = pd.DataFrame(VW_GOLD_result['x2'])
gold_vw_nre = pd.DataFrame(gold_vw_nre).values
VW_GOLD_result=pd.DataFrame(ret_goldCI.values*x1.values+Spx_Index.values*x2.values+rf*(1-x1.values-x2.values))

sharpe_PF_3 = (VW_GOLD_result-rf.values).mean()*12/(VW_GOLD_result.std()*np.sqrt(12))
sharpe_nre_3 = (gold_vw_nre-rf.values).mean()*12/(gold_vw_nre.std().mean()*np.sqrt(12))

print('sh_PF_VW_GOLD= %.2f' %(sharpe_PF_3))
print('sh_NRE_VW_GOLD= %.2f' %(sharpe_nre_3))
print('-----------------------------------------')
#################################################
####### Value Weighted OIL NRE
#################################################
vw_oil=pd.DataFrame()

for i in range(235):
    r=oilcap[i:i+1]/oilcap.sum(axis=1)[i]
    vw_oil=pd.concat([vw_oil,r])
    
oil_vw_nre = vw_oil.multiply(oil_ret).sum(axis=1)[121:]

VW_OIL_result=pd.DataFrame(index=np.arange(0, numberOfwindow), columns=('x1', 'x2','c') )
for i in range(numberOfwindow):
    x = []
    x.append(ret_petCI[0+i:120+i])
    x.append(Spx_Index[0 + i:120+i])
    result = reg_m(oil_vw_nre[0 + i:120+i], x)
    VW_OIL_result.loc[i] = [result.params[0], result.params[1], result.params[2]]
    
x1 =pd.DataFrame(VW_OIL_result['x1'])
x2 = pd.DataFrame(VW_OIL_result['x2'])
oil_vw_nre = pd.DataFrame(oil_vw_nre).values
VW_OIL_result=pd.DataFrame(ret_petCI.values*x1.values+Spx_Index.values*x2.values+rf*(1-x1.values-x2.values))

sharpe_PF_4 = (VW_OIL_result-rf.values).mean()*12/(VW_OIL_result.std()*np.sqrt(12))
sharpe_nre_4 = (oil_vw_nre-rf.values).mean()*12/(oil_vw_nre.std().mean()*np.sqrt(12))

print('sh_PF_VW_OIL= %.2f' %(sharpe_PF_4))
print('sh_NRE_VW_OIL= %.2f' %(sharpe_nre_4))
print('-----------------------------------------')
#VW_OIL_result_VWCI=pd.DataFrame(index=np.arange(0, numberOfwindow), columns=('x1', 'x2','c') )
#for i in range(116):
#    x = []
#    wa=list((pd.Series(ret_petCI[121:])*(54.68/59.41))+(pd.Series(ret_goldCI[121:])*4.73/59.41))
#    x.append(wa[0+i:120+i])    
#    x.append(Spx_Index[0 + i:120+i])
#    result = reg_m(oil_vw_nre[0 + i:120+i], x)
#    VW_OIL_result_VWCI.loc[i] = [result.params[0], result.params[1], result.params[2]]
#    
#vw_vw_nre1=pd.Series(vw_vw_nre1) #여기서부터 잘 못됨!!!!!! CI도 합쳐야함..!!
#vw_ci=list((pd.Series(oci)*(54.68/59.41))+(pd.Series(gci)*4.73/59.41))
#vw_vw_return1=vw_ci[120:]*vw_vw_result1['x1']+mi[120:]*vw_vw_result1['x2']+lb.values[121:]*(1-vw_vw_result1['x1']-vw_vw_result1['x2'])
#vw_vw_exret1=pd.Series(vw_vw_nre1[120:].values-vw_vw_return1.values, index=vw_vw_return1.index)















# 그래프 보기!!
#plt.figure(1)
#plt.plot(vw_vw_result['x1'],label = 'oil')
#plt.plot(vw_vw_result['x2'],label = 'gold')
#plt.plot(vw_vw_result['x3'],label = 'index')
#plt.plot(1-(vw_vw_result['x1']+vw_vw_result['x2']+vw_vw_result['x3']),label='cash')
#plt.legend()
#plt.show()


##Rolling 116 window- vw_ew_nre
#vw_ew_nre=list(pd.Series((a['port_ret']*(0.5)).values+(b['port_ret']*(0.5)).values-lb.values, index=lb.index))[1:]
#text_file = open("vw_ew_nre_result.txt", "w")
#vw_ew_result=pd.DataFrame(index=np.arange(0, numberOfwindow), columns=('x1', 'x2', 'x3','c') )
#for i in range(116):
#    x = []
#    x.append(oci[0+i:120+i])
#    x.append(gci[0 + i:120+i])
#    x.append(mi[0 + i:120+i])
#    result = reg_m(vw_ew_nre[0 + i:120+i], x)
#    text_file.write(result.summary().as_text())
#    vw_ew_result.loc[i] = [result.params[0], result.params[1], result.params[2], result.params[3]]
#text_file.close()
#
#vw_ew_nre=pd.Series(vw_ew_nre)
#vw_ew_return=oci[120:]*vw_ew_result['x1']+gci[120:]*vw_ew_result['x2']+mi[120:]*vw_ew_result['x3']+lb.values[121:]*(1-vw_ew_result['x1']-vw_ew_result['x2']-vw_ew_result['x3'])
#vw_ew_exret=pd.Series(vw_ew_nre[120:].values-vw_ew_return.values, index=vw_ew_return.index)
#
#
#



































#price = oilprice
#
#rets = price.pct_change()
#covmat = pd.DataFrame.cov(rets)
#
#def RC(weight, covmat) :
#    weight = np.array(weight)
#    variance = weight.T @ covmat @ weight
#    sigma = variance ** 0.5
#    mrc = 1/sigma * (covmat @ weight)
#    rc = weight * mrc
#    rc = rc / rc.sum()
#    rc = pd.DataFrame(rc).set_index(covmat.index)
#    return(rc)
#
#    
#def RiskParity_objective(x) :
#    
#    variance = x.T @ covmat @ x
#    sigma = variance ** 0.5
#    mrc = 1/sigma * (covmat @ x)
#    rc = x * mrc
#    a = np.reshape(rc, (len(rc), 1))
#    risk_diffs = a - a.T
#    sum_risk_diffs_squared = np.sum(np.square(np.ravel(risk_diffs)))
#    return (sum_risk_diffs_squared)
#
#def weight_sum_constraint(x) :
#    return(x.sum() - 1.0 )
#
#
#def weight_longonly(x) :
#    return(x)
#
#def RiskParity(covmat) :
#    
#    x0 = np.repeat(1/covmat.shape[1], covmat.shape[1]) 
#    constraints = ({'type': 'eq', 'fun': weight_sum_constraint},
#                  {'type': 'ineq', 'fun': weight_longonly})
#    options = {'ftol': 1e-20, 'maxiter': 800}
#    
#    result = minimize(fun = RiskParity_objective,
#                      x0 = x0,
#                      method = 'SLSQP',
#                      constraints = constraints,
#                      options = options)
#    return(result.x)
#
#wt_erc = RiskParity(covmat)
#
#wt_ew = np.repeat(1/rets.shape[1], rets.shape[1]) 
#rc_ew = RC(wt_ew, covmat)
#col = list(oilprice.columns)
#col =np.array(col)
#
#fig = plt.figure(1)
#plt.rcParams["figure.figsize"] = (15,8)
#plt.rcParams['lines.linewidth'] = 2
#plt.rcParams['axes.grid'] = True
#
#plt.subplot(211)
#plt.bar(rc_ew.index,rc_ew[0])
#
#plt.subplot(212)
#pd.DataFrame(wt_ew).plot(kind = 'bar',width=30)
#
#plt.tight_layout()
#
#oilprice.columns
#


#rc_ew = pd.DataFrame(rc_ew).set_index(PF_stocks.iloc[-1,:])   

#oilprice = oilprice.T
#rc_ew = pd.concat([rc_ew,oilprice.loc[oilprice.index.isin(rc_ew.index)]],axis=1) 
#
#wts = rc_ew.pop(0)
#PFW_price = rc_ew
#PFW_daily = PFW_price.pct_change(axis=1)
#
#PFW_daily =  PFW_daily.mul(wts,axis=0).T                           
#PFW_daily.iloc[0,:] = 0
#PFW_dailycum= np.cumprod(PFW_daily.mean(axis=1)+1,axis=0)   
