import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from header import set_size

'''
0 for frank and or ridge
1 for real and or lasso
'''
df = 0
reg = 0
data_func = ['frank','real']
reg_method = ['ridge','lasso']

n = 10
phi = np.arange(1,n+1)
if df == 0:
    Nlams = 50 # for frank
    lambdas = np.logspace(-18,1,Nlams) # for frank
elif df == 1:
    Nlams = 20 # for real
    lambdas = np.logspace(-10,1,Nlams) # for real
lmb_ind = 10
titlelmb = lambdas[lmb_ind]
å
# Load chunky matrices
msetrain = np.loadtxt(f'matrices/{data_func[df]}/{reg_method[reg]}/msetrain_Nlmb{Nlams}.txt')#, dtype=float)
msetest = np.loadtxt(f'matrices/{data_func[df]}/{reg_method[reg]}/msetest_Nlmb{Nlams}.txt')#, dtype=float)
bmse = np.loadtxt(f'matrices/{data_func[df]}/{reg_method[reg]}/bmse_Nlmb{Nlams}.txt')#, dtype=float)
cvtest= np.loadtxt(f'matrices/{data_func[df]}/{reg_method[reg]}/cvtest_Nlmb{Nlams}.txt')#, dtype=float)
cvtrain = np.loadtxt(f'matrices/{data_func[df]}/{reg_method[reg]}/cvtrain_Nlmb{Nlams}.txt')#, dtype=float)


# Initial plot of Ridge or Lasso for Frank Figure 10 & 13
'''
plt.style.use("ggplot") 
plt.figure(figsize=set_size(345), dpi=80)
plt.plot(phi, np.log10( msetrain[:,lmb_ind]  ), color='green', label="Train")
plt.plot(phi, np.log10( msetest[:,lmb_ind]  ), '--', color='red', label="Test")
plt.xlabel(r"$\phi$")
plt.ylabel(r"log10(MSE)")
plt.title(f"{reg_method[reg]}: lmb={titlelmb:.3}")    #f' lmb:{lambdas[lmb_ind]}')
plt.legend()
# plt.savefig(f'results/ridgelmb-{titlelmb:.2}.pdf', format='pdf', bbox_inches='tight')
# plt.savefig(f'results/lassolmb-{titlelmb:.2}.pdf', format='pdf', bbox_inches='tight')
plt.show()
'''

# Finding optimal lambda
"""
# Pre-process the phi x lambda matrix with MSE vals
z = cvtest
xlen,ylen = np.shape(cvtest)
x = np.linspace(1,xlen,xlen)  # == phi
y = np.log10(np.logspace(-18,1,ylen))   # == lambdas
xmesh,ymesh = np.meshgrid(x,y,indexing='ij')
z_min, z_max = -np.abs(z).max(), np.abs(z).max()
"""
# Plot heatmap of lambda dependence (crossvalid)
"""
plt.style.use("ggplot") 
fig, ax = plt.subplots(figsize=set_size(345), dpi=80)
c = ax.pcolormesh(xmesh,ymesh,z,cmap=cm.coolwarm)#,vmin=z_min, vmax=z_max)
ax.set_xlabel(r"$\phi$")
ax.set_ylabel(r"log10($\lambda$)")
#ax.axis([x.min(), x.max(), y.min(), y.max()])
fig.colorbar(c)# , ax=ax)
plt.title(f"{reg_method[reg]}")
plt.savefig(f'results/heatmaps/{reg_method[reg]}_matrices/{data_func[df]}.pdf', format='pdf', bbox_inches='tight')
plt.show()
"""
'''
# phi vs MSE
for i in range(10):
    plt.plot(z[i,:],label=f'phi_index={i}')
    plt.legend()
    plt.show()
# lmb vs MSE
for i in range(Nlams): # np.arange(30,40,1):
    plt.plot(z[:,i],label=f'lmb_index={i}')
    plt.legend()
    plt.show()
'''

'''
print("lambda value: ",lambdas[4])
print("MSE: ", z[3,4])
plt.plot(z[:,4])
plt.show()
'''


# Boots vs CV Figure 11 & 14
'''
# l for lasso and r for ridge
l = [30,35,38,45]
r = [8,18,38,42]
for i in range(len(l)):
    # lmb = l[i] # lasso
    # lmb = r[i] # ridge
    plt.style.use("ggplot")
    plt.figure(figsize=set_size(345), dpi=80)
    # Boots plot
    plt.plot(phi, np.log10( bmse[lmb,:] ), label="boots_MSE")
    # CV plot
    plt.plot(phi, np.log10( cvtest[:,lmb]), label="cv_MSE")
    plt.xlabel(r"$\phi$")
    plt.ylabel(r"log10(MSE)")
    # plt.title(f'Lasso_lmb:{lambdas[lmb]:.3}')
    # plt.title(f'Ridge_lmb:{lambdas[lmb]:.3}')
    plt.legend()
    # plt.savefig(f'results/lambdadep/BootsCVLassolmb-{lmb}.pdf', format='pdf', bbox_inches='tight')
    # plt.savefig(f'results/lambdadep/BootsCVRidgelmb-{lmb}.pdf', format='pdf', bbox_inches='tight')
    plt.show()
'''



'''
** K = 20 Folds **

Frank:
    Lasso:
        at 30th lambda = 4.29e-7
        min deg b4 overfitting at degree 3 (index = 2)
        MSE: 0.99688424
    Ridge:
        at 38th lambda = 0.0005428675439323859
        min deg b4 overfitting at degree 6 (index = 5)
        MSE: 0.00846642190362118

Real:
    Lasso:
        at 11th lambda =  0.00023357214690901214
        min deg b4 overfitting at deg 3 (index = 2)
        MSE: 0.5016612801911691

    Ridge:
        at 4th lambda =  2.0691380811147902e-08
        min deg b4 overfitting at deg 4 (index = 3)
        MSE: 0.4979094441821377
'''

