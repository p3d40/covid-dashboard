import pandas as pd
import io
import requests
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from scipy.integrate import odeint
import math
from scipy.stats import gamma
import datetime
import matplotlib.dates as mdates
from statistics import mean


RESULTS_DIR='results/'
VALIDATION_DIR='validation/'
HTML_DIR='html/'
IMG_DIR='html/img/'

inc, incMin, incMax= 5.1, 4.3, 5.8
rec=14
M=17000000

def round10(x,d):
    m=math.floor(math.log10(x))
    r=(math.floor(x/10**(m-d)+1/2))*(10**(m-d))
    return '{:,}'.format(r)

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def deriv(y, t, Rp, l, N,mu, gammap, a, beta):
    S, E, I, R, D = y
    dSdt = - beta * S * E / N
    dEdt = beta*S * E / N - a*E
    dIdt = a*E - gammap*I
    dRdt = gammap*(1-mu) * I
    dDdt = gammap*mu*I
    return dSdt, dEdt, dIdt, dRdt, dDdt

def seird(Rp,m,k,Tt,l):
    t = np.linspace(0, Tt-1, Tt)
    N=M
    gammap=1/rec
    a=1/l
    beta=Rp*a
    I0=confirmedTs[k]-recoveredTs[k]-deathsTs[k]
    E0=beta*I0
    R0=recoveredTs[k]
    D0=deathsTs[k]
    S0=N-E0-I0-R0-D0
    y0 = S0, E0, I0, R0, D0
    ret = odeint(deriv, y0, t, args=(Rp,l,N,m, gammap, a, beta))
    S, E, I, R, D = ret.T
    return S,E,I,R,D

def rep(k,b):
    k=k+len(confirmedTs0)-len(confirmedTs)
    mu=inc
    s=(incMax-incMin)/4
    a=(mu**2)/(s**2)
    th=(s**2)/mu
    n=[gamma.pdf(x=k, a=a, scale = th) for k in range(b+1)]
    rTs=[1 / sum( [n[i+1]*confirmedTs0[j-i-1]/confirmedTs0[j] for i in range(b) ]) for j in range(k-14,k+1)]
    rp,rpmin,rpmax = mean_confidence_interval(rTs)
    return rp,rpmin,rpmax

def mort(k):
    ratios=deathsTs[:k]/confirmedTs[:k]
    m,mMin,mMax=mean_confidence_interval(ratios)
    return m, mMin, mMax

def doubling(k,b):
    confidence=0.95
    window=confirmedTs[k-b:k]
    x=range(len(window))
    y=[math.log(f) for f in window.values]
    slope, intercept, r_value, p_value, se = scipy.stats.linregress(x,y)
    n=len(x)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    DT=math.log(2)/slope
    DTMin=math.log(2)/(slope+h)
    DTMax=math.log(2)/(slope-h)
    fig6 = plt.figure(facecolor='w')
    ax6 = fig6.add_subplot(111, axisbelow=True)
    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    formatter = mdates.ConciseDateFormatter(locator)
    ax6.xaxis.set_major_locator(locator)
    ax6.xaxis.set_major_formatter(formatter)
    ax6.plot(confirmedTs,'-' ,color='r', alpha=0.5, lw=2)
    ax6.set_xlabel('Fecha')
    ax6.set_ylabel('Casos')
    date_time = current_date.strftime("%m/%d/%Y")
    strTitle='Casos COVID19 (Guatemala)\nEscala Logarítmica'
    ax6.set_title(strTitle)
    ax6.yaxis.set_tick_params(length=0)
    ax6.xaxis.set_tick_params(length=0)
    ax6.set_yscale('log')
    ax6.grid(b=True, which='both', c='0.75', alpha=0.2, lw=2, ls='-')
    for spine in ('top', 'right', 'bottom', 'left'):
        ax6.spines[spine].set_visible(False)
    filename='doubling.png'
    plt.savefig(RESULTS_DIR+filename)
    plt.savefig(IMG_DIR+filename)
    plt.close()
    return DT, DTMin, DTMax, r_value**2

def forecast(k,b,T):
    S,E,I,R,D=seird( rep(k,b)[0], mort(k)[0], k, T, inc)
    SMin,EMin,IMin,RMin,DMin=seird( rep(k,b)[1], mort(k)[1], k, T, incMax)
    SMax,EMax,IMax,RMax,DMax=seird( rep(k,b)[2], mort(k)[2], k, T, incMin)
    cts=I+R+D
    ctsMin=IMin+RMin+DMin
    ctsMax=IMax+RMax+DMax
    dts=D
    dtsMin=DMin
    dtsMax=DMax
    base = confirmedTs.index[k]
    date_list = [base + datetime.timedelta(days=x) for x in range(T)]
    fig1 = plt.figure(facecolor='w')
    ax1 = fig1.add_subplot(111, axisbelow=True)
    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    formatter = mdates.ConciseDateFormatter(locator)
    ax1.xaxis.set_major_locator(locator)
    ax1.xaxis.set_major_formatter(formatter)
    ax1.plot(confirmedTs, 'b', alpha=0.5, lw=2, label='Casos')
    ax1.plot(date_list, cts, '--' ,color='r', alpha=0.5, lw=2, label='Proyección')
    ax1.fill_between(date_list, ctsMin, ctsMax, alpha=0.2, color='r')
    ax1.set_xlabel('Fecha')
    ax1.set_ylabel('Casos')
    date_time = current_date.strftime("%m/%d/%Y")
    strTitle='Proyección de 30 días al '+date_time+'\n Casos COVID19 (Guatemala)'
    ax1.set_title(strTitle)
    ax1.yaxis.set_tick_params(length=0)
    ax1.xaxis.set_tick_params(length=0)
    ax1.grid(b=True, which='major', c='0.75', alpha=0.2, lw=2, ls='-')
    legend1 = ax1.legend()
    legend1.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax1.spines[spine].set_visible(False)
    filename='forecast_confirmed.png'
    plt.savefig(RESULTS_DIR+filename)
    plt.savefig(IMG_DIR+filename)
    plt.close()
    fig2 = plt.figure(facecolor='w')
    ax2 = fig2.add_subplot(111, axisbelow=True)
    ax2.xaxis.set_major_locator(locator)
    ax2.xaxis.set_major_formatter(formatter)
    ax2.plot(deathsTs, 'b', alpha=0.5, lw=2, label='Muertes')
    ax2.plot(date_list, dts, '--' ,color='r', alpha=0.5, lw=2, label='Proyección')
    ax2.fill_between(date_list, dtsMin, dtsMax, alpha=0.2, color='r')
    ax2.set_xlabel('Fecha')
    ax2.set_ylabel('Muertos')
    date_time = current_date.strftime("%m/%d/%Y")
    strTitle='Proyección de 30 días al '+date_time+'\n Muertes COVID19 (Guatemala)'
    ax2.set_title(strTitle)
    ax2.yaxis.set_tick_params(length=0)
    ax2.xaxis.set_tick_params(length=0)
    ax2.grid(b=True, which='major', c='0.75', alpha=0.2, lw=2, ls='-')
    legend2 = ax2.legend()
    legend2.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax2.spines[spine].set_visible(False)
    filename='forecast_deaths.png'
    plt.savefig(RESULTS_DIR+filename)
    plt.savefig(IMG_DIR+filename)
    plt.close()
    return cts[-1], ctsMin[-1], ctsMax[-1], dts[-1], dtsMin[-1],dtsMax[-1]

def rsquared(y,f):
    ym=mean(y)
    sst=sum([(y[i]-ym)**2 for i in range(len(y))])
    ssr=sum([(y[i]-f[i])**2 for i in range(len(y))])
    rsq=1-ssr/sst
    return rsq

def validation(k,b):
    T=len(confirmedTs)-k
    S,E,I,R,D=seird( rep(k,b)[0], mort(k)[0], k, T, inc)
    SMin,EMin,IMin,RMin,DMin=seird( rep(k,b)[1], mort(k)[1], k, T, incMax)
    SMax,EMax,IMax,RMax,DMax=seird( rep(k,b)[2], mort(k)[2], k, T, incMin)
    cts=I+R+D
    ctsMin=IMin+RMin+DMin
    ctsMax=IMax+RMax+DMax
    dts=D
    dtsMin=DMin
    dtsMax=DMax
    ctsrs=rsquared(confirmedTs[k:now+1],cts)
    dtsrs=rsquared(deathsTs[k:now+1],dts)
    base = confirmedTs.index[k]
    date_list = [base + datetime.timedelta(days=x) for x in range(T)]
    fig1 = plt.figure(facecolor='w')
    ax1 = fig1.add_subplot(111, axisbelow=True)
    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    formatter = mdates.ConciseDateFormatter(locator)
    ax1.xaxis.set_major_locator(locator)
    ax1.xaxis.set_major_formatter(formatter)
    ax1.plot(confirmedTs, 'b', alpha=0.5, lw=2, label='Casos', marker="o",linestyle="")
    ax1.plot(date_list, cts, '--' ,color='g', alpha=0.5, lw=2, label='Interpolación')
    ax1.fill_between(date_list, ctsMin, ctsMax, alpha=0.2, color='g')
    ax1.set_xlabel('Fecha')
    ax1.set_ylabel('Casos')
    date_time = base.strftime("%m/%d/%Y")
    strTitle='Validación de Casos desde '+date_time+'\nCOVID19 (Guatemala) R^2='+'{:.2f}'.format(ctsrs)
    ax1.set_title(strTitle)
    ax1.yaxis.set_tick_params(length=0)
    ax1.xaxis.set_tick_params(length=0)
    ax1.grid(b=True, which='major', c='0.75', alpha=0.2, lw=2, ls='-')
    legend1 = ax1.legend()
    legend1.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax1.spines[spine].set_visible(False)
    filename='validation_cases_'+'{0:03d}'.format(k)+'.png'
    plt.savefig(VALIDATION_DIR+filename)
    plt.savefig(IMG_DIR+filename)
    plt.close()
    fig2 = plt.figure(facecolor='w')
    ax2 = fig2.add_subplot(111, axisbelow=True)
    ax2.xaxis.set_major_locator(locator)
    ax2.xaxis.set_major_formatter(formatter)
    ax2.plot(deathsTs, 'r', alpha=0.5, lw=2, label='Muertes', marker="o",linestyle="")
    ax2.plot(date_list, dts, '--' ,color='y', alpha=0.5, lw=2, label='Interpolación')
    ax2.fill_between(date_list, dtsMin, dtsMax, alpha=0.2, color='y')
    ax2.set_xlabel('Fecha')
    ax2.set_ylabel('Muertos')
    strTitle='Validación de Muertes desde '+date_time+'\nCOVID19 (Guatemala) R^2='+'{:.2f}'.format(dtsrs)
    ax2.set_title(strTitle)
    ax2.yaxis.set_tick_params(length=0)
    ax2.xaxis.set_tick_params(length=0)
    ax2.grid(b=True, which='major', c='0.75', alpha=0.2, lw=2, ls='-')
    legend2 = ax2.legend()
    legend2.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax2.spines[spine].set_visible(False)
    filename='validation_deaths_'+'{0:03d}'.format(k)+'.png'
    plt.savefig(VALIDATION_DIR+filename)
    plt.savefig(IMG_DIR+filename)
    plt.close()
    return ctsrs, dtsrs


print("Inicialndo processo...")
url_confirmed="https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
url_recovered="https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv"
url_deaths="https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"

print("Obteniendo datos...")
s=requests.get(url_confirmed).content
c=pd.read_csv(io.StringIO(s.decode('utf-8')))
c=c.loc[c['Country/Region']=='Guatemala']
df=c.drop(['Province/State','Country/Region','Lat','Long'], axis=1)
confirmedTs=df.iloc[0]
confirmedTs0=confirmedTs.loc[confirmedTs>0]
confirmedTs0.index = pd.to_datetime(confirmedTs0.index)
initial_date0=confirmedTs0.index[0]
confirmedTs=confirmedTs0.loc[confirmedTs0>100]
initial_date=confirmedTs.index[0]
current_date=confirmedTs.index[-1]

s=requests.get(url_recovered).content
c=pd.read_csv(io.StringIO(s.decode('utf-8')))
c=c.loc[c['Country/Region']=='Guatemala']
df=c.drop(['Province/State','Country/Region','Lat','Long'], axis=1)
recoveredTs0=df.iloc[0]
recoveredTs0.index = pd.to_datetime(recoveredTs0.index)
recoveredTs0=recoveredTs0.loc[initial_date0:]
recoveredTs=recoveredTs0.loc[initial_date:]

s=requests.get(url_deaths).content
c=pd.read_csv(io.StringIO(s.decode('utf-8')))
c=c.loc[c['Country/Region']=='Guatemala']
df=c.drop(['Province/State','Country/Region','Lat','Long'], axis=1)
deathsTs0=df.iloc[0]
deathsTs0.index = pd.to_datetime(deathsTs0.index)
deathsTs0=deathsTs0.loc[initial_date0:]
deathsTs=deathsTs0.loc[initial_date:]

#latest
now=len(confirmedTs)-1

#new
print("Casos nuevos...")
nconfirmedTs=confirmedTs.diff()
nrecoveredTs=recoveredTs.diff()
ndeathsTs=deathsTs.diff()
nmconfirmedTs=nconfirmedTs.rolling(window=7).mean()
fig5 = plt.figure(facecolor='w')
ax5 = fig5.add_subplot(111, axisbelow=True)
locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
formatter = mdates.ConciseDateFormatter(locator)
ax5.xaxis.set_major_locator(locator)
ax5.xaxis.set_major_formatter(formatter)
ax5.plot(nmconfirmedTs, 'b', alpha=0.8, lw=2, label='Promedios Móviles')
ax5.bar(nconfirmedTs.index, nconfirmedTs.values,color='r', alpha=0.2, lw=2, label='Casos Nuevos')
ax5.set_xlabel('Fecha')
ax5.set_ylabel('Casos Nuevos')
date_time = current_date.strftime("%m/%d/%Y")
strTitle='Casos Nuevos al '+date_time+'\n COVID19 (Guatemala)'
ax5.set_title(strTitle)
ax5.yaxis.set_tick_params(length=0)
ax5.xaxis.set_tick_params(length=0)
ax5.grid(b=True, which='major', c='0.75', alpha=0.2, lw=2, ls='-')
legend5 = ax5.legend()
legend5.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax5.spines[spine].set_visible(False)
filename='new.png'
plt.savefig(RESULTS_DIR+filename)
plt.savefig(IMG_DIR+filename)
plt.close()

#forecast
print("Propyecciones...")
cts, ctsMin, ctsMax, dts, dtsMin,dtsMax = forecast(now,10,30)
future_time=confirmedTs.index[now]+datetime.timedelta(days=30)
future_date=future_time.strftime("%m/%d/%Y")

#current
print("Estado actual...")
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, axisbelow=True)
locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
formatter = mdates.ConciseDateFormatter(locator)
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)
ax.plot(confirmedTs0, 'b', alpha=0.5, lw=2, label='Confirmados')
ax.plot(recoveredTs0, 'g', alpha=0.5, lw=2, label='Recuperados')
ax.plot(deathsTs0, 'r', alpha=0.5, lw=2, label='Muertos')
ax.set_xlabel('Fecha')
ax.set_ylabel('Personas')
date_time = current_date.strftime("%m/%d/%Y")
strTitle='COVID19 (Guatemala) al '+date_time
ax.set_title(strTitle)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='0.75', alpha=0.2, lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
filename='current.png'
plt.savefig(RESULTS_DIR+filename)
plt.savefig(IMG_DIR+filename)
plt.close()

#mortality
print("Letalidad...")
base = confirmedTs.index[0]
date_list = [base + datetime.timedelta(days=x) for x in range(1,now+1)]
mTs=[100*mort(i+1)[0] for i in range(1,now+1)]
mTsMin=[100*mort(i+1)[1] for i in range(1,now+1)]
mTsMax=[100*mort(i+1)[2] for i in range(1,now+1)]
fig3 = plt.figure(facecolor='w')
ax3 = fig3.add_subplot(111, axisbelow=True)
locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
formatter = mdates.ConciseDateFormatter(locator)
ax3.xaxis.set_major_locator(locator)
ax3.xaxis.set_major_formatter(formatter)
ax3.plot_date(date_list,mTs,'--' ,color='r', alpha=0.5, lw=2)
ax3.fill_between(date_list, mTsMin, mTsMax, alpha=0.2, color='r')
ax3.set_xlabel('Fecha')
ax3.set_ylabel('Porcentaje')
date_time = current_date.strftime("%m/%d/%Y")
strTitle='Letalidad al'+date_time+'\nCOVID19 (Guatemala)'
ax3.set_title(strTitle)
ax3.yaxis.set_tick_params(length=0)
ax3.xaxis.set_tick_params(length=0)
ax3.grid(b=True, which='major', c='0.75', alpha=0.2, lw=2, ls='-')
for spine in ('top', 'right', 'bottom', 'left'):
    ax3.spines[spine].set_visible(False)
filename='mortality.png'
plt.savefig(RESULTS_DIR+filename)
plt.savefig(IMG_DIR+filename)
plt.close()

#reproductive
print("Número de reproducción...")
be=10
offset=max(be,2*math.floor(incMax))
base = confirmedTs.index[0]
date_list = [base + datetime.timedelta(days=x) for x in range(offset,now)]
repTs=[rep(i+1,be)[0] for i in range(offset,now)]
repTsMin=[rep(i+1,be)[1] for i in range(offset,now)]
repTsMax=[rep(i+1,be)[2] for i in range(offset,now)]
fig4 = plt.figure(facecolor='w')
ax4 = fig4.add_subplot(111, axisbelow=True)
locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
formatter = mdates.ConciseDateFormatter(locator)
ax4.xaxis.set_major_locator(locator)
ax4.xaxis.set_major_formatter(formatter)
ax4.plot_date(date_list,repTs,'--' ,color='b', alpha=0.5, lw=2)
ax4.fill_between(date_list, repTsMin, repTsMax, alpha=0.2, color='b')
ax4.set_xlabel('Fecha')
ax4.set_ylabel('Número de Reproducción')
date_time = current_date.strftime("%m/%d/%Y")
strTitle='Número de Reproducción al '+date_time+'\nCOVID19 (Guatemala)'
ax4.set_title(strTitle)
ax4.yaxis.set_tick_params(length=0)
ax4.xaxis.set_tick_params(length=0)
ax4.grid(b=True, which='major', c='0.75', alpha=0.2, lw=2, ls='-')
for spine in ('top', 'right', 'bottom', 'left'):
    ax4.spines[spine].set_visible(False)
filename='reproductive.png'
plt.savefig(RESULTS_DIR+filename)
plt.savefig(IMG_DIR+filename)
plt.close()

#doubling
print("Período de duplicación...")
dt,dtmin,dtmax, rsq=doubling(now,14)

#validation
print("Validando modelo...")
rsqTs=[validation(k,14) for k in range(2,now-2)]
#rscts=mean(list(zip(*rsqTs))[0])
#rsdts=mean(list(zip(*rsqTs))[1])
rscts=mean_confidence_interval(list(zip(*rsqTs))[0])
rsdts=mean_confidence_interval(list(zip(*rsqTs))[1])

#HTML
print("Generando HTML Dashboard...")
html="""
<!doctype html>
<html lang="en">
 <head>
  <meta charset="UTF-8">
  <meta content="IE=edge" http-equiv="X-UA-Compatible">
  <meta content="width=device-width, initial-scale=1" name="viewport">
  <title>COVID19-Guatemala</title>
  <!-- CSS only -->
<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.0/css/bootstrap.min.css" integrity="sha384-9aIt2nRpC12Uk9gS9baDl411NQApFmC26EwAOH8WgZl5MYYxFfc+NcPb1dKGj7Sk" crossorigin="anonymous">
<link rel="stylesheet" href="css/custom_dash.css">

<!-- JS, Popper.js, and jQuery -->
<script src="https://code.jquery.com/jquery-3.5.1.slim.min.js" integrity="sha384-DfXdz2htPH0lsSSs5nCTpuj/zy4C+OGpamoFVy38MVBnE+IbbVYUew+OrCXaRkfj" crossorigin="anonymous"></script>
<script src="https://cdn.jsdelivr.net/npm/popper.js@1.16.0/dist/umd/popper.min.js" integrity="sha384-Q6E9RHvbIyZFJoft+2mJbHaEWldlvI9IOYy5n3zV9zzTtmI3UksdQRVvoxMfooAo" crossorigin="anonymous"></script>
<script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.0/js/bootstrap.min.js" integrity="sha384-OgVRvuATP1z7JjHLkuOU7Xw704+h835Lr+6QL9UvYjZE3Ipu6Tp75j7Bh/kR0JKI" crossorigin="anonymous"></script>
<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML">
</script>


 </head>
 <body>

  <div class="container">
    <div class="row text-center">
      <div class="col-12">
        <h1>COVID-19 Guatemala</h1>
      </div>
    </div>

    <div class="row text-center">
      <div class="col-12">
        <h3>Proyecciones</h3>
      </div>
    </div>

    <div class="row">
        <div class="col-sm-6">
            <img src="img/forecast_confirmed.png " class="img-fluid center-block forecast-img">
        </div>
        <div class="col-sm-6">
          <img src="img/forecast_deaths.png" class="img-fluid center-block forecast-img">
        </div>

    </div>


    <div class="row">
      <div class="col-sm-6 vertical-middle-text">
        <h4>
          Proyección para el
          """
html+=future_date
html+="""
        </h4>
        <p>
          Casos positivos: entre
          """
html+=round10(ctsMin,1) +" y "+ round10(ctsMax,1)+" (est. "+round10(cts,1)+ ")"
html+="""
        <br/>
          Muertes: entre
          """
html+= round10(dtsMin,1) +" y "+ round10(dtsMax,1)+" (est. "+round10(dts,1)+ ")"
html+="""
        </p>
        <p>
          Las bandas de confianza del modelo se obtienen por medio de considerar los
          intervalos de confianza de los parámetros al 95%.
        </p>
        <p><a href="model.html">Detalles del modelo...</a></p>
      </div>
      <div class="col-sm-6">
        <p>
          El modelo se valida por medio del coeficiente de determinación de proyecciones pasadas.
          Los intervalos de confianza para los valores de \(R^2\) para casos reportados y muertes
           respectivamente son:</p>
           <p class="text-center">
           \(R^2_c\) entre """
html+='{:0.3f}'.format(rscts[1])+" y "+'{:0.3f}'.format(rscts[2])+" (promedio "+'{:0.3f}'.format(rscts[0])+")"
html+="""<br/>
\(R^2_{m}\) entre """
html+='{:0.3f}'.format(rsdts[1])+" y "+'{:0.3f}'.format(rsdts[2])+" (promedio "+'{:0.3f}'.format(rsdts[0])+")"
html+="""</p>
        <p><a href="validation.html">Detalles de validación...</a></p>
      </div>
    </div>

    <div class="row">
      <div class="col-12 text-center">
        <h2>
          Estado actual
        </h2>
      </div>
    </div>

    <div class="row">
      <div class="col-sm-6 vertical-middle-text">
        <h4>
          Estadísticas para el
          """
html+=date_time
html+="""
        </h4>
        <p>
          Casos positivos:
          """
html+='{:,}'.format(confirmedTs[-1])+" (+"+'{:,.0f}'.format(nconfirmedTs[-1])
html+=""")
          <br/>
          Recuperados:
          """
html+='{:,}'.format(recoveredTs[-1])+" (+"+'{:,.0f}'.format(nrecoveredTs[-1])
html+=""")
          <br/>
          Muertos:
          """
html+='{:,}'.format(deathsTs[-1])+" (+"+'{:,.0f}'.format(ndeathsTs[-1])
html+=""")
          <br/><br/>
          Letalidad: entre
          """
html+='{:.2f}'.format(mTsMin[-1])+"% y "+'{:.2f}'.format(mTsMax[-1])
html+="""%
          <br/>
          Período de duplicación: entre
          """
html+='{:,.2f}'.format(dtmin)+" días y "+'{:,.2f}'.format(dtmax)
html+="""
          días
          <br/>
          Número de reproducción: entre
          """
html+='{:,.2f}'.format(repTsMin[-1]) +" y "+ '{:,.2f}'.format(repTsMax[-1])
html+="""
        </p>
      </div>
      <div class="col-sm-6">
          <img src="img/current.png " class="img-fluid center-block forecast-img">
      </div>
    </div>

    <div class="row">
      <div class="col-sm-6">
          <img src="img/new.png " class="img-fluid center-block forecast-img">
      </div>
      <div class="col-sm-6 vertical-middle-text">
        <h4>Casos Nuevos</h4>
        <p>
          Para analizar de mejor manera el número de casos nuevos es importante considerar
           los promedios móviles. De esta forma se minimiza el impacto de los errores presentes
           en los datos y se pueden observar mejor las tendiencias presentes.
        </p>
      </div>
    </div>

    <div class="row">
      <div class="col-sm-6 vertical-middle-text">
        <h4>Letalidad</h4>
        <p>
          La letalidad se calcula como el porcentaje de infectados que mueren debido a la enfermedad.
          Para esto se toma la distribución de pprcentajes durante los días de la epidemia y se
          calculan los intervalos de confianza al 95%.
        </p>
      </div>
      <div class="col-sm-6">
          <img src="img/mortality.png " class="img-fluid center-block forecast-img">
      </div>
    </div>

    <div class="row">
      <div class="col-sm-6">
          <img src="img/reproductive.png " class="img-fluid center-block forecast-img">
      </div>
      <div class="col-sm-6 vertical-middle-text">
        <h4>Número de Reproducción</h4>
        <p>
          Este número mide la cantidad promedio de contagios secundarios ocacionadas por
          un infectado en un intervalo de tiempo. Este se calcula utilizando los datos y obteniendo
          los intervalos de confianza al 95%.
        </p>
      </div>
    </div>

    <div class="row">
      <div class="col-sm-6 vertical-middle-text">
        <h4>Período de Duplicación</h4>
        <p>
          Entre """
html+='{:,.2f}'.format(dtmin)+" días y "+'{:,.2f}'.format(dtmax)
html+="""
           días
        </p>
        <p>
          Este parámetro estima en cuando tiempo se espera que el número de casos se duplique.
          Actualmente el modelo exponencial presenta un coeficiente de determinación de
          $$R^2=
          """
html+='{:,.3f}'.format(rsq)
html+="\,,$$"
if (rsq>0.80):
    strex="por lo que el brote presenta un crecimiento <strong>exponencial</strong>."
else:
    strex="por lo que el brote presenta un crecimiento <strong>sub-exponencial</strong>."
html+=strex
html+="""
        </p>
      </div>
      <div class="col-sm-6">
          <img src="img/doubling.png " class="img-fluid center-block forecast-img">
      </div>
    </div>


  </div>
 </body>
</html>
"""

filename='dashboard.html'
f = open(HTML_DIR+filename,'w')
f.write(html)
f.close()


print("Generando HTML Validación...")
html="""
<!doctype html>
<html lang="en">
 <head>
  <meta charset="UTF-8">
  <meta content="IE=edge" http-equiv="X-UA-Compatible">
  <meta content="width=device-width, initial-scale=1" name="viewport">
  <title>COVID19 Validacion Guatemala</title>
  <!-- CSS only -->
<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.0/css/bootstrap.min.css" integrity="sha384-9aIt2nRpC12Uk9gS9baDl411NQApFmC26EwAOH8WgZl5MYYxFfc+NcPb1dKGj7Sk" crossorigin="anonymous">
<link rel="stylesheet" href="css/custom_dash.css">

<!-- JS, Popper.js, and jQuery -->
<script src="https://code.jquery.com/jquery-3.5.1.slim.min.js" integrity="sha384-DfXdz2htPH0lsSSs5nCTpuj/zy4C+OGpamoFVy38MVBnE+IbbVYUew+OrCXaRkfj" crossorigin="anonymous"></script>
<script src="https://cdn.jsdelivr.net/npm/popper.js@1.16.0/dist/umd/popper.min.js" integrity="sha384-Q6E9RHvbIyZFJoft+2mJbHaEWldlvI9IOYy5n3zV9zzTtmI3UksdQRVvoxMfooAo" crossorigin="anonymous"></script>
<script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.0/js/bootstrap.min.js" integrity="sha384-OgVRvuATP1z7JjHLkuOU7Xw704+h835Lr+6QL9UvYjZE3Ipu6Tp75j7Bh/kR0JKI" crossorigin="anonymous"></script>
<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML">
</script>


 </head>
 <body>

  <div class="container">
    <div class="row text-center">
      <div class="col-12">
        <h1>COVID-19 Guatemala</h1><br/>
      </div>
    </div>

    <div class="row text-center">
      <div class="col-12">
        <h3>Validación</h3>
      </div>
    </div>"""
for k in range(2,now-2):
    html+="""<div class="row">
        <div class="col-sm-6">
            <img src="img/"""
    html+='validation_cases_'+'{0:03d}'.format(k)+'.png'
    html+="""" class="img-fluid center-block forecast-img">
        </div>
        <div class="col-sm-6">
          <img src="img/"""
    html+='validation_deaths_'+'{0:03d}'.format(k)+'.png'
    html+="""" class="img-fluid center-block forecast-img">
        </div>
    </div>"""
html+="""
  </div>
 </body>
</html>
"""

filename='validation.html'
f = open(HTML_DIR+filename,'w')
f.write(html)
f.close()

print("Proceso completado.")
