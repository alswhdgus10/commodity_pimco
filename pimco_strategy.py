# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
#import scipy as sp
#import scipy.stats as sps
import statsmodels.api as sm
import matplotlib.pyplot as plt
#from scipy.optimize import minimize

plt.close('all')
direc="C:/Users/mjh/Documents/commodity_pimco/"

oilprice = pd.read_csv(direc+'oilprice.csv',header =0 ,encoding='CP949',engine='python').set_index('Date',drop=True)
oilprice.index = pd.to_datetime(oilprice.index,format='%Y-%m-%d') 
goldprice = pd.read_csv(direc+'goldprice.csv',header =0,encoding='CP949',engine='python').set_index('Date',drop=True)
goldprice.index = pd.to_datetime(goldprice.index,format='%Y-%m-%d') 

oilcap = pd.read_csv(direc+'oilcap.csv',header =0,encoding='CP949',engine='python').set_index('Date',drop=True)
oilcap.index = pd.to_datetime(oilcap.index,format='%Y-%m-%d') 
goldcap = pd.read_csv(direc+'goldcap.csv',header =0,encoding='CP949',engine='python').set_index('Date',drop=True)
goldcap.index = pd.to_datetime(goldcap.index,format='%Y-%m-%d') 

origin = pd.read_csv(direc+'origin2.csv',header =0,encoding='CP949',engine='python').set_index('Date',drop=True)
origin.index = pd.to_datetime(origin.index,format='%Y-%m-%d') 


pet_CI = pd.DataFrame(origin['SPXGS_PTRM']).reset_index(drop=True)


gold_CI = pd.DataFrame(origin['SPXGS_GTRM']).reset_index(drop=True)

Spx = pd.DataFrame(origin['S&P500']).reset_index(drop=True)

rf = pd.DataFrame(origin['LIBOR_3M'].fillna(0)/1200).reset_index(drop=True)

date_index = pd.DataFrame(goldprice.index).set_index('Date',drop=False).loc['20080630':'20180228']
#%% Gold Index 모멘텀 스코어 

#모멘텀 1~12개월 롤링 스코어 (골드)
n = 237
M_gold = pd.DataFrame(np.zeros((n,12)))
for j in range(12):
    for i in range(j+1,len(gold_CI)):
        if gold_CI.loc[i].values-gold_CI.loc[i-j-1].values>0:
           M_gold.iloc[i,j]=1
        else:
           M_gold.iloc[i,j]=0
       

#모멘텀 1~12개월 롤링 스코어 (오일)
M_oil = pd.DataFrame(np.zeros((n,12)))
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
w = cc.loc[:n+1]
  
w_gold = (M_gold.mean(axis=0).sum()/12)/(M_gold.mean(axis=0).sum()/12+M_oil.mean(axis=0).sum()/12)  
w_oil = (M_oil.mean(axis=0).sum()/12)/(M_gold.mean(axis=0).sum()/12+M_oil.mean(axis=0).sum()/12)  

wg = pd.DataFrame(w['w_gold'])      
wo = pd.DataFrame( w['w_oil'])

ret_goldCI = (gold_CI/gold_CI.shift(1)-1)   
ret_petCI = (pet_CI/pet_CI.shift(1)-1)

#ret_CI_PF = ret_goldCI.values*w_gold+ret_petCI.values*w_oil
ret_CI_PF =ret_goldCI.values*wg.values+ret_petCI.values*wo.values

ret_CI_PF[0]=0

ret_CI_PF = pd.DataFrame(ret_CI_PF)

#%%

oil_ret = (oilprice/oilprice.shift(1)-1)
gold_ret = (goldprice/goldprice.shift(1)-1)

oil_weightsumM = oilcap.sum(axis=1)

gold_weightsumM = goldcap.sum(axis=1)

oil_MonthlyW = pd.DataFrame()
gold_MonthlyW = pd.DataFrame()

for i in range(n):
    W=oilcap[i:i+1]/oil_weightsumM[i]
    oil_MonthlyW=pd.concat([oil_MonthlyW,W])
for i in range(n):
    W=goldcap[i:i+1]/gold_weightsumM[i]
    gold_MonthlyW=pd.concat([gold_MonthlyW,W])

## NRE 구성
vw_oilret = pd.DataFrame((oil_ret*oil_MonthlyW.values).sum(axis=1))
vw_goldret = pd.DataFrame((gold_ret*gold_MonthlyW.values).sum(axis=1))

nre = vw_oilret*(54.68/59.41)+vw_goldret*(4.73/59.41)

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

numberOfwindow = 116
numberOfmonthIn10Y=120

ret_goldCI_buff = ret_goldCI[121:]
ret_petCI_buff = ret_petCI[121:]
rsqured_buff = pd.DataFrame(columns=['ewnre_momentumci','vwnre_ewgoldci','vwoilnre'])
#%%
#################################################
####### 동일비중 NRE ~  CI 모멘텀 회귀
#################################################

text_file = open("result/Mimic_PF_result.txt", "w")
Mimic_PF_result=pd.DataFrame(index=np.arange(0, numberOfwindow), columns=('x1', 'x2','c') )
rsquare_list = []
for i in range(numberOfwindow):
    x = []
    x.append(Ci[0+i:numberOfmonthIn10Y+i])
    x.append(Spx_Index[0 + i:numberOfmonthIn10Y+i])
    result = reg_m(nre[0 + i:numberOfmonthIn10Y+i], x)
    rsquare_list.append(result.rsquared)
    text_file.write(result.summary().as_text())
    Mimic_PF_result.loc[i] = [result.params[0], result.params[1], result.params[2]]
rsqured_buff['ewnre_momentumci'] = pd.Series(rsquare_list).values
text_file.close()

Ci_buff=pd.DataFrame(Ci[numberOfmonthIn10Y:]).reset_index(drop=True)
Spx_Index_buff = pd.DataFrame(Spx_Index[numberOfmonthIn10Y:]).reset_index(drop=True)
rf_buff = pd.DataFrame(rf[121:].values).reset_index(drop=True)
Broad_CI_beta =pd.DataFrame(Mimic_PF_result['x1'])
ME_beta = pd.DataFrame(Mimic_PF_result['x2'])
Cash_beta = pd.DataFrame(1-Broad_CI_beta.values-ME_beta.values)

nre_buff=pd.DataFrame(nre[numberOfmonthIn10Y:])

Mimic_PF_ret=pd.DataFrame(Ci_buff.values*Broad_CI_beta.values+Spx_Index_buff.values*ME_beta.values+rf_buff*(1-Broad_CI_beta.values-ME_beta.values))
excess_ret=pd.DataFrame(-nre_buff.values+Mimic_PF_ret.values)

sharpe_PF = (Mimic_PF_ret-rf_buff.values).mean()*12/(Mimic_PF_ret.std()*np.sqrt(12))

sharpe_nre = (nre_buff-rf_buff.values).mean()*12/(nre_buff.std().mean()*np.sqrt(12))

df = pd.DataFrame()
df['a'] = Mimic_PF_ret[0].values.astype(float)
df['b'] = nre_buff[0].reset_index(drop=True)
Model_corr = df['a'].corr(df['b'])

print('####### 동일비중 NRE ~  CI 모멘텀 회귀 #######')
print("Rsquare Mean: %.3f" %(sum(rsquare_list) / float(len(rsquare_list))))
print('Model correlation = %.2f' %(Model_corr))
print('sh_PF= %.2f' %(sharpe_PF))
print('sh_NRE= %.2f' %(sharpe_nre))
print('Annualized return PF = %.2f%%' %(((1+Mimic_PF_ret.mean())**12 - 1)*100 ))
print('Volatility PF = %.2f%%' %(Mimic_PF_ret.std()*100))
print('Annualized return NRE = %.2f%%' %(((1+nre_buff.mean())**12-1)*100))
print('Volatility NRE = %.2f%%' %(nre_buff.std()*100))
print('-----------------------------------------')


fig = plt.figure(1)
plt.rcParams["figure.figsize"] = (10,6)
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.grid'] = True

plt.plot(date_index.index,ME_beta,'r-',label='Broad commodities composite beta')
plt.plot(date_index.index,Broad_CI_beta,'b-',label = 'S&P beta')
plt.plot(date_index.index,Cash_beta,'g-',label = 'Cash beta')
plt.legend(loc=2, prop={'size': 7})
plt.title('Replicating NRE : rolling betas using 120 monthly observations')
plt.ylabel('beta')
plt.xlabel('time(Y)')
plt.savefig('result/1.jpg')
plt.close()


fig = plt.figure(2)
plt.rcParams["figure.figsize"] = (10,6)
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.grid'] = True

Mimic_PF_ret_cum = (Mimic_PF_ret+1).cumprod()-1
nre_buff_cum = (nre_buff+1).cumprod().reset_index(drop=True)-1
excess_ret_cum = (excess_ret+1).cumprod()-1

plt.plot(date_index.index,nre_buff_cum,'r-',label='NRE')
plt.plot(date_index.index,Mimic_PF_ret_cum,'b-',label = 'Replicating portfolio')
plt.plot(date_index.index,excess_ret_cum,'g-',label = 'Excess return')
plt.legend(loc=2, prop={'size': 10})
plt.title('Growth of a dollar for replicating portfolio and NRE portfolio')
plt.ylabel('Growth of a dollar($)')
plt.xlabel('time(Y)')
plt.savefig('result/2.jpg')
plt.close()

fig = plt.figure(3)
plt.rcParams["figure.figsize"] = (10,6)
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.grid'] = True

plt.plot(date_index.index,nre_buff,'r-',label='NRE')
plt.plot(date_index.index,Mimic_PF_ret,'b-',label = 'Replicating portfolio')
plt.legend(loc=2, prop={'size': 7})
plt.title('Monthly Return of NRE and replicating portfolio')
plt.ylabel('return')
plt.xlabel('time(Y)')
plt.savefig('result/3.jpg')
plt.close()
#%%
#################################################
####### 시총비중 Gold NRE ~  가중평균 Gold CI 회귀
#################################################
vw_gold=pd.DataFrame()
for i in range(n):
    r=goldcap[i:i+1]/goldcap.sum(axis=1)[i]
    vw_gold=pd.concat([vw_gold,r],axis=0)
    
gold_vw_nre = vw_gold.multiply(gold_ret).sum(axis=1)
gold_vw_nre_buff = gold_vw_nre[121:]

VW_GOLD_result=pd.DataFrame(index=np.arange(0, numberOfwindow), columns=('x1', 'x2','c') )
rsquare_list = []
text_file = open("result/vw_gold.txt", "w")
for i in range(numberOfwindow):
    x = []
    x.append(ret_goldCI[1+i:121+i])
    x.append(Spx_Index[0 + i:numberOfmonthIn10Y+i])
    result = reg_m(gold_vw_nre[1 + i:121+i], x)
    rsquare_list.append(result.rsquared)
    text_file.write(result.summary().as_text())
    VW_GOLD_result.loc[i] = [result.params[0], result.params[1], result.params[2]]
rsqured_buff['vwnre_ewgoldci'] = pd.Series(rsquare_list).values
text_file.close()
vw_goldCI_beta =pd.DataFrame(VW_GOLD_result['x1'])
vw_ME_betaG = pd.DataFrame(VW_GOLD_result['x2'])
gold_vw_nre = pd.DataFrame(gold_vw_nre).reset_index(drop=True)
VW_GOLD_result_beta = VW_GOLD_result
VW_GOLD_result=pd.DataFrame(ret_goldCI_buff.values*vw_goldCI_beta.values+Spx_Index_buff.values*vw_ME_betaG.values+rf_buff*(1-vw_goldCI_beta.values-vw_ME_betaG.values))

sharpe_PF_3 = (VW_GOLD_result-rf_buff.values).mean()*12/(VW_GOLD_result.std()*np.sqrt(12))
sharpe_nre_3 = (gold_vw_nre_buff.values-rf_buff.values).mean()*12/(gold_vw_nre_buff.values.std()*np.sqrt(12))
#excess_ret2 = (-gold_vw_nre_buff.reset_index(drop=True)+VW_GOLD_result.reset_index(drop=True)).dropna(axis=0,how='all')[0]

df2 = pd.DataFrame()
df2['a'] = VW_GOLD_result[0].values.astype(float)
df2['b'] = pd.DataFrame(gold_vw_nre_buff)[0].reset_index(drop=True)
Model_corr2 = df2['a'].corr(df2['b'])

print('####### 시총비중 Gold NRE ~  가중평균 Gold CI 회귀  #######')
print("Rsquare Mean: %.3f" %(sum(rsquare_list) / float(len(rsquare_list))))  
print('Model correlation = %.2f' %(Model_corr2))
print('sh_PF_VW_GOLD= %.2f' %(sharpe_PF_3))
print('sh_NRE_VW_GOLD= %.2f' %(sharpe_nre_3))
print('Annualized return PF = %.2f%%' %(((1+VW_GOLD_result.mean())**12 - 1)*100 ))
print('Volatility PF = %.2f%%' %(VW_GOLD_result.std()*100))
print('Annualized return NRE = %.2f%%' %(((1+gold_vw_nre_buff.mean())**12-1)*100))
print('Volatility NRE = %.2f%%' %(gold_vw_nre_buff.std()*100))
print('-----------------------------------------')

fig = plt.figure(7)
plt.rcParams["figure.figsize"] = (10,6)
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.grid'] = True

plt.plot(date_index.index,vw_goldCI_beta,'r-',label='Broad commodities composite beta')
plt.plot(date_index.index,vw_ME_betaG,'b-',label = 'S&P beta')
plt.plot(date_index.index,1-vw_goldCI_beta.values-vw_ME_betaG.values,'g-',label = 'Cash beta')
plt.legend(loc=2, prop={'size': 7})
plt.savefig('result/7.jpg')
plt.close()

fig = plt.figure(8)
plt.rcParams["figure.figsize"] = (10,6)
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.grid'] = True

VW_GOLD_result_cum = (VW_GOLD_result+1).cumprod()-1
gold_vw_nre_buff_cum = (gold_vw_nre_buff+1).cumprod().reset_index(drop=True)-1
#excess_ret_cum2 = (excess_ret2+1).cumprod()-1

plt.plot(date_index.index,gold_vw_nre_buff_cum,'r-',label='NRE')
plt.plot(date_index.index,VW_GOLD_result_cum,'b-',label = 'Replicating portfolio')
#plt.plot(date_index.index,excess_ret_cum2,'g-',label = 'Excess return')
plt.legend(loc=2, prop={'size': 7})
plt.savefig('result/8.jpg')
plt.close()

fig = plt.figure(9)
plt.rcParams["figure.figsize"] = (10,6)
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.grid'] = True

plt.plot(date_index.index,gold_vw_nre_buff,'r-',label='NRE')
plt.plot(date_index.index,VW_GOLD_result,'b-',label = 'Replicating portfolio')
plt.legend(loc=2, prop={'size': 7})
plt.savefig('result/9.jpg')
plt.close()
#%%
#################################################
####### Value Weighted OIL NRE 
#################################################
vw_oil=pd.DataFrame()

for i in range(n):
    r=oilcap[i:i+1]/oilcap.sum(axis=1)[i]
    vw_oil=pd.concat([vw_oil,r])
    
oil_vw_nre = vw_oil.multiply(oil_ret).sum(axis=1)
oil_vw_nre_buff = oil_vw_nre[121:]

VW_OIL_result=pd.DataFrame(index=np.arange(0, numberOfwindow), columns=('x1', 'x2','c') )
text_file = open("result/vw_oil.txt", "w")
rsquare_list = []
for i in range(numberOfwindow):
    x = []
    x.append(ret_petCI[1+i:121+i])
    x.append(Spx_Index[0+i:numberOfmonthIn10Y+i])
    result = reg_m(oil_vw_nre[1 + i:121+i], x)
    rsquare_list.append(result.rsquared)
    text_file.write(result.summary().as_text())
    VW_OIL_result.loc[i] = [result.params[0], result.params[1], result.params[2]]
rsqured_buff['vwoilnre'] = pd.Series(rsquare_list).values
text_file.close()
x1 =pd.DataFrame(VW_OIL_result['x1'])
x2 = pd.DataFrame(VW_OIL_result['x2'])
oil_vw_nre = pd.DataFrame(oil_vw_nre).values
VW_OIL_result_beta = VW_OIL_result
VW_OIL_result=pd.DataFrame(ret_petCI_buff.values*x1.values+Spx_Index_buff.values*x2.values+rf_buff*(1-x1.values-x2.values))

sharpe_PF_4 = (VW_OIL_result-rf_buff.values).mean()*12/(VW_OIL_result.std()*np.sqrt(12))
sharpe_nre_4 = (pd.DataFrame(oil_vw_nre_buff)-rf_buff.values).mean()*12/(pd.DataFrame(oil_vw_nre_buff).std().mean()*np.sqrt(12))

df3 = pd.DataFrame()
oil_vw_nre_buff_df = oil_vw_nre_buff.to_frame()
df3['a'] = VW_OIL_result[0].values.astype(float)
df3['b'] = oil_vw_nre_buff_df[0].reset_index(drop=True)
Model_corr3 = df3['a'].corr(df3['b'])

print('####### 시총비중 Oil NRE ~  가중평균 Oil CI 회귀  #######')
print("Rsquare Mean: %.3f" %(sum(rsquare_list) / float(len(rsquare_list))))  
print('Model correlation = %.2f' %(Model_corr3))
print('sh_PF_VW_OIL= %.2f' %(sharpe_PF_4))
print('sh_NRE_VW_OIL= %.2f' %(sharpe_nre_4))
print('Annualized return PF = %.2f%%' %(((1+VW_OIL_result.mean())**12 - 1)*100 ))
print('Volatility PF = %.2f%%' %(VW_OIL_result.std()*100))
print('Annualized return NRE = %.2f%%' %(((1+oil_vw_nre_buff.mean())**12-1)*100))
print('Volatility NRE = %.2f%%' %(oil_vw_nre_buff.std()*100))
print('-----------------------------------------')
print('oil ci, oil nre corr = %.5f' %(ret_petCI['SPXGS_PTRM'].corr(pd.DataFrame(oil_vw_nre)[0])))
print('gold ci, gold nre corr = %.5f' %(ret_goldCI['SPXGS_GTRM'].corr(gold_vw_nre[0])))

date_index_cum = pd.DataFrame(goldprice.index).set_index('Date',drop=False).loc['19980430':'20180228']

fig = plt.figure(10)
plt.rcParams["figure.figsize"] = (10,6)
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.grid'] = True

plt.plot(date_index_cum.index,(ret_petCI['SPXGS_PTRM']+1).cumprod()-1,'r-',label='S&P GSCI PETROLEUM')
plt.plot(date_index_cum.index,(pd.DataFrame(oil_vw_nre)[0]+1).cumprod()-1,'b-',label = 'Oil NRE')
plt.legend(loc=2, prop={'size': 10  })
plt.title('S&P GSCI PETROLEUM & Oil NRE Cumulative Return')
plt.ylabel('return')
plt.xlabel('time(Y)')
plt.savefig('result/10.jpg')
plt.close()
## RSquare
fig = plt.figure(11)
plt.rcParams["figure.figsize"] = (10,6)
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.grid'] = True

plt.plot(date_index.index,rsqured_buff['ewnre_momentumci'],'r-',label='ewnre_momentumci')
plt.plot(date_index.index,rsqured_buff['vwnre_ewgoldci'],'b-',label = 'vwnre_ewgoldci')
plt.plot(date_index.index,rsqured_buff['vwoilnre'],'g-',label = 'vwoilnre')
plt.legend(loc=2, prop={'size': 10  })
plt.ylabel('RSquare')
plt.xlabel('time(Y)')
plt.savefig('result/11.jpg')
plt.close()

## Mimic_PF_result Beta
fig = plt.figure(12)
plt.rcParams["figure.figsize"] = (10,6)
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.grid'] = True

plt.plot(date_index.index,Mimic_PF_result['x1'],'r-',label='CI')
plt.plot(date_index.index,Mimic_PF_result['x2'],'b-',label = 'MI')
#plt.plot(date_index.index,Mimic_PF_result['c'],'g-',label = 'c')
plt.legend(loc=2, prop={'size': 10  })
plt.title('Mimic_PF_result Beta')
plt.xlabel('time(Y)')
plt.savefig('result/12.jpg')
plt.close()
## VW_OIL_result_beta
fig = plt.figure(13)
plt.rcParams["figure.figsize"] = (10,6)
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.grid'] = True

plt.plot(date_index.index,VW_OIL_result_beta['x1'],'r-',label='CI')
plt.plot(date_index.index,VW_OIL_result_beta['x2'],'b-',label = 'MI')
#plt.plot(date_index.index,VW_OIL_result_beta['c'],'g-',label = 'c')
plt.legend(loc=2, prop={'size': 10  })
plt.title('VW_OIL_result Beta')
plt.xlabel('time(Y)')
plt.savefig('result/13.jpg')
plt.close()
## VW_GOLD_result_beta
fig = plt.figure(14)
plt.rcParams["figure.figsize"] = (10,6)
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.grid'] = True

plt.plot(date_index.index,VW_GOLD_result_beta['x1'],'r-',label='CI')
plt.plot(date_index.index,VW_GOLD_result_beta['x2'],'b-',label = 'MI')
#plt.plot(date_index.index,VW_GOLD_result_beta['c'],'g-',label = 'c')
plt.legend(loc=2, prop={'size': 10  })
plt.title('VW_GOLD_result Beta')
plt.xlabel('time(Y)')
plt.savefig('result/14.jpg')
plt.close()  