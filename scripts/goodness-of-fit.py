from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import numpy as np

state = "MP"
df = pd.read_csv("data/MP/MP.csv")
df = df.sample(n=10000, random_state=2)
speed = df['Wind Speed'].to_numpy()
list_of_dists = ['alpha','anglit','arcsine','beta','betaprime','bradford','burr','burr12','cauchy','chi','chi2','cosine','dgamma','dweibull','erlang','expon','exponnorm','exponweib','exponpow','f','fatiguelife','fisk','foldcauchy','foldnorm','frechet_r','frechet_l','genlogistic','genpareto','gennorm','genexpon','genextreme','gausshyper','gamma','gengamma','genhalflogistic','gilbrat','gompertz','gumbel_r','gumbel_l','halfcauchy','halflogistic','halfnorm','halfgennorm','hypsecant','invgamma','invgauss','invweibull','johnsonsb','johnsonsu','kstwobign','laplace','levy','levy_l','logistic','loggamma','loglaplace','lognorm','lomax','maxwell','mielke','nakagami','ncx2','ncf','nct','norm','pareto','pearson3','powerlaw','powerlognorm','powernorm','rdist','reciprocal','rayleigh','rice','recipinvgauss','semicircular','t','triang','truncexpon','truncnorm','tukeylambda','uniform','vonmises','vonmises_line','wald','weibull_min','weibull_max']

size = 10000
x = np.arange(size)
dist_name = "weibull_min"
name = "Fitted Weibull-Min Distribution"

h = plt.hist(speed, bins=50, label="Data")
plt.xlabel("Wind Speed")
plt.ylabel("Freuency")
plt.title(state)
dist = getattr(stats, dist_name)
param = dist.fit(speed)
pdf_fitted = dist.pdf(x, *param[:-2], loc=param[-2], scale=param[-1])
scale_pdf = np.trapz(h[0], h[1][:-1]) / np.trapz(pdf_fitted, x)
pdf_fitted *= scale_pdf

plt.plot(x, pdf_fitted, label=name)
plt.xlim(0,9)
plt.legend()
fname = state
plt.savefig("/home/ashryaagr/Desktop/"+fname+".png")

results = []
for i in list_of_dists:
    dist = getattr(stats, i)
    param = dist.fit(speed)
    a = stats.kstest(speed, i, args=param)
    results.append((i, a[0], a[1]))

results.sort(key=lambda x: float(x[2]), reverse=True)
for j in results:
    print("{}: statistic={}, pvalue={}".format(j[0], j[1], j[2]))