import numpy as np
import matplotlib.pyplot as plt
import math as math
import pyfits
import scipy.optimize as optimize
import scipy.cluster.hierarchy as hclust
import copy
import scipy.ndimage as ndimage
import congrid as congrid
import resource
import pickle
import sys
from collections import Counter
from mpfit import mpfit
from mpl_toolkits.mplot3d import Axes3D
import glob

pi=3.14159265359
parlist = []


class FitParms(object):
    """__init__() functions as the class constructor"""
    def __init__(self, objname=None, trialnum=None, diskparm=None, coneparm=None, fracdisk=None, chi2=None, frontonly=0, backonly=0, cone_oangle2=0.):
        self.objname = objname
        self.trialnum = trialnum
        self.diskparm = diskparm
        self.coneparm = coneparm
        self.fracdisk = fracdisk
        self.chi2 = chi2
        self.frontonly=frontonly
        self.backonly=backonly
        self.cone_oangle2=cone_oangle2


class InitialInput(object):
    def __init__(self, piguess1=None, piguess2=None, objname=None, velfitsname=None, highcut=None, lowcut=None,
                 modelnum=None):
        self.piguess1 = piguess1
        self.piguess2 = piguess2
        self.objname = objname
        self.velfitsname = velfitsname
        self.highcut = highcut
        self.lowcut = lowcut
        self.modelnum = modelnum


def mk_initial_input():

"""
Get initial guesses for each target
piguess1 = ['iangle', 'vc', 'pa', 'xoff', 'yoff', 'veloff'] - disk
piguess2 = ['oangle1','iangle', 'pa', 'k1', 'xoff', 'yoff', 'veloff'] - bicone
"""
    
    inilist = []

# SBS1250+568 - good disk
    piguess1 = [18.,430., 61., 0.5, 1.5, 0.]
    piguess2 = [35.,30., -135., 20., 1, 1, 0.]
    objname = 'SBS1250+568'
    velfitsname = 'SBS1250+568vel_slice1.fits'
    highcut=250
    lowcut=-250
    modelnum = 1
    inilist.append(InitialInput(piguess1,piguess2,objname,velfitsname,highcut,lowcut,modelnum))

# HB1741+279 - good disk
    piguess1 = [17.,300., -47., -0.7, -0.7, -190.]
    piguess2 = [55.,15., 140., 30., 0., 0., -180.]
    objname = 'HB1741+279'
    velfitsname = 'HB1741+279vel_slice1_1.fits'
    highcut=1
    lowcut=-400
    modelnum = 1
    inilist.append(InitialInput(piguess1,piguess2,objname,velfitsname,highcut,lowcut,modelnum))

# 4C+21.26 - well constrained disk
    piguess1 = [10.,376., 15., 2.0, -1.8, -165.]
    piguess2 = [35., 20.,-65., 15., 0, 0, -200.]
    objname = '4C+21.26'
    velfitsname = '4C+21.26vel_slice1_1.fits'
    highcut=-50
    lowcut=-300
    modelnum = 1
    inilist.append(InitialInput(piguess1,piguess2,objname,velfitsname,highcut,lowcut,modelnum))

# HB0812+020 - well constrained disk, but very big y-offset
    piguess1 = [18.,236., -88., 2.8, 0.7, -105.]
    piguess2 = [35., 40.,45., 15., -2, 1, -200.]
    objname = 'HB0812+020'
    velfitsname = 'HB0812+020vel_slice1_0.fits'
    highcut=100
    lowcut=-230
    modelnum = 1
    inilist.append(InitialInput(piguess1,piguess2,objname,velfitsname,highcut,lowcut,modelnum))

# HB0952+097 - well constrained cone, oangle2 = 0
#    piguess1 = [20.,100., 90., -0.5, -1, -100.]
#    piguess2 = [25., 83.,-90., 76., -0.87, -0.4, -117.]
#    objname = 'HB0952+097'
#    velfitsname = 'HB0952+097vel_slice1_0.fits'
#    highcut=100
#    lowcut=-350
#    modelnum = 2
#    inilist.append(InitialInput(piguess1,piguess2,objname,velfitsname,highcut,lowcut,modelnum))

# HB1153+317 - well constrained disk
    piguess1 = [36.,162., -31., 0.94, 0.94, -61.]
    piguess2 = [35., 20.,-65., 15., 0, 0, -200.]
    objname = 'HB1153+317'
    velfitsname = 'HB1153+317vel_slice1_0.fits'
    highcut=100
    lowcut=-200
    modelnum = 1
    inilist.append(InitialInput(piguess1,piguess2,objname,velfitsname,highcut,lowcut,modelnum))

# HB1745+163 - well constrained disk, Vcenter not very well constrained likely due to small inclination angle
    piguess1 = [6.,2300., -163., 0.2, -0.2, -74.]
    piguess2 = [35., 20.,-20., 35., 0, 0, -200.]
    objname = 'HB1745+163'
    velfitsname = 'HB1745+163vel_slice1_1.fits'
    highcut=400
    lowcut=-400
    modelnum = 1
    inilist.append(InitialInput(piguess1,piguess2,objname,velfitsname,highcut,lowcut,modelnum))

# HB2209+080 - ok constrained disk + cone, oangle2 = 20, best disk frac = 0.14 +- 0.28
        #fit parameter median and mean should be calculated with just the best 25 fits
        #Disk inclination should be closer to 30, even though best fit is 48
        #Vcenter not very well constrained 
        #Disk xoffset not very wel,l constrained
        #Difficult - trying to fit combined model to small number of valid spaxels
  #  piguess1 = [30.,550., -180., -1.2, 1.0, 20.]
  #  piguess2 = [50., 5.,50., 180., -2.2, 0.1, -80.]
  #  objname = 'HB2209+080'
  #  velfitsname = 'HB2209+080vel_slice1_1.fits'
  #  highcut=100
  #  lowcut=-300
  #  modelnum = 0
  #  inilist.append(InitialInput(piguess1,piguess2,objname,velfitsname,highcut,lowcut,modelnum))

# HB2251+113 - well constrained disk + cone, Vcenter not very well constrained likely due to small inclination angle
            # oangle2 = 20, best frac disk = 0.76 +- 0.17
#    piguess1 = [10.,1500., 82., 0.0, 0, -151.]
#    piguess2 = [56., 81.,-87., 400., -0.5, -0.4, -253.]
#    objname = 'HB2251+113'
#    velfitsname = 'HB2251+113vel_slice1_4.fits'
#    highcut=200
#    lowcut=-450
#    modelnum = 0
#    inilist.append(InitialInput(piguess1,piguess2,objname,velfitsname,highcut,lowcut,modelnum))

# 4C+10.71 - well constrained disk
    piguess1 = [67.,245., 106., 0., 0.7, -161.]
    piguess2 = [35., 20.,-90., 15., 1, 0, -200.]
    objname = '4C+10.71'
    velfitsname = '4C+10.71vel_slice1_0.fits'
    highcut=100
    lowcut=-400
    modelnum = 1
    inilist.append(InitialInput(piguess1,piguess2,objname,velfitsname,highcut,lowcut,modelnum))

# PG1545+210 - well constrained disk + cone, oangle2 = 20, best disk frac = 0.74 +- 0.17
    piguess1 = [26.,481., 68., -0.8, 0.2, -51.]
    piguess2 = [61., 3.,-170., 305., -0.2, 0.5, 122.]
    objname = 'PG1545+210'
    velfitsname = 'PG1545+210vel_slice1_2.fits'
    highcut=370
    lowcut=-300
    modelnum = 0
    inilist.append(InitialInput(piguess1,piguess2,objname,velfitsname,highcut,lowcut,modelnum))

# 3C299 - well constrained disk
    piguess1 = [23.,563., 158., -0.5, 0.6, -60.]
    piguess2 = [35., 20.,-65., 25., 0, 0, 0.]
    objname = '3C299'
    velfitsname = '3C299vel_slice1_0.fits'
    highcut= 200
    lowcut=-300
    modelnum = 1
    inilist.append(InitialInput(piguess1,piguess2,objname,velfitsname,highcut,lowcut,modelnum))



# 3C459
    piguess1 = [30.,200., -135., -1, -1, -200.]
    piguess2 = [35., 20.,-135., 25., 0, 0, -200.]
    objname = '3C459'
    velfitsname = '3C459vel_slice1_0.fits'
    highcut=100
    lowcut=-400
    inilist.append(InitialInput(piguess1,piguess2,objname,velfitsname,highcut,lowcut))

# 4C+34.42
    piguess1 = [30.,200., 90., 0, 0, -100.]
    piguess2 = [35., 20.,-90., 15., 0, 0, -100.]
    objname = '4C+34.42'
    velfitsname = '4C+34.42vel_slice1_1.fits'
    highcut=1
    lowcut=-300
    inilist.append(InitialInput(piguess1,piguess2,objname,velfitsname,highcut,lowcut))

# FBQSJ122539.5+245836 - can't be described by disk-bicone model, exclude for now
#    piguess1 = [15.,400., 70., 1, -1, 0.]
#    piguess2 = [35., 20.,-45., 30., 1, 1, -200.]
#    objname = 'FBQSJ122539.5+245836'
#    velfitsname = 'FBQSJ122539.5+245836vel_slice1_0.fits'
#    highcut=100
#    lowcut=-400
#    inilist.append(InitialInput(piguess1,piguess2,objname,velfitsname,highcut,lowcut))

# HB0017+257
    piguess1 = [30., 300., 90., 1, 1, -150.]
    piguess2 = [35., 45.,0., 10., 1, -2, -150.]
    objname = 'HB0017+257'
    velfitsname = 'HB0017+257vel_slice1_0.fits'
    highcut=150
    lowcut=-200
    inilist.append(InitialInput(piguess1,piguess2,objname,velfitsname,highcut,lowcut))

    inifile = open('InitialInput', 'wb')
    pickle.dump(inilist, inifile)
    inifile.close()




def repeat_mpfit_all(nrepeat = 10, startnum = 0, modelnum=0):
    
    nloop = nrepeat - startnum

    inifile = open('InitialInput', 'rb')
    inilist = pickle.load(inifile)
    inifile.close()
    
    n_object = len(inilist)

    for k in xrange(n_object):
        objname = inilist[k].objname
        piguess1 = inilist[k].piguess1
        piguess2 = inilist[k].piguess2
        velfitsname = inilist[k].velfitsname
        highcut = inilist[k].highcut
        lowcut = inilist[k].lowcut
        modelnum = inilist[k].modelnum

#Set up to perturb the initial guess using a random factor drawn from Gaussian distribution
    #Disk Component
        npar1 = 6

        pardict1 = {'name':'tmp', 'mean':0., 'stddev':0}
        parnames1 = ['iangle', 'vc', 'pa', 'xoff', 'yoff', 'veloff']
        parlimits1 = []
        for i in range(npar1): 
            parlimits1.append(copy.deepcopy(pardict1))
        for i in range(npar1): 
            parlimits1[i]['name'] = parnames1[i]

        parlimits1[0]['stddev'] = 5.
        parlimits1[1]['stddev'] = 10.
        parlimits1[2]['stddev'] = 10.
        parlimits1[3]['stddev'] = 0.3
        parlimits1[4]['stddev'] = 0.3
        parlimits1[5]['stddev'] = 50.

        for i in range(npar1): 
            parlimits1[i]['mean'] = piguess1[i]

        pmean1 = np.zeros(npar1)
        pstddev1 = np.zeros(npar1)
        for i in range(npar1):
            pmean1[i] = parlimits1[i]['mean'] 
            pstddev1[i] =  (parlimits1[i]['stddev'] + parlimits1[i]['stddev']) / 2.

    #Bicone Component
        npar2 = 7

        pardict2 = {'name':'tmp', 'mean':0., 'stddev':0}
        parnames2 = ['oangle1','iangle', 'pa', 'k1', 'xoff', 'yoff', 'veloff']
        parlimits2 = []
        for i in xrange(npar2): 
            parlimits2.append(copy.deepcopy(pardict2))
        for i in xrange(npar2): 
            parlimits2[i]['name'] = parnames2[i]

        parlimits2[0]['stddev'] = 5.
        parlimits2[1]['stddev'] = 5.
        parlimits2[2]['stddev'] = 10.
        parlimits2[3]['stddev'] = 5.
        parlimits2[4]['stddev'] = 0.3
        parlimits2[5]['stddev'] = 0.3
        parlimits2[6]['stddev'] = 50.

        for i in xrange(npar2): 
            parlimits2[i]['mean'] = piguess2[i]

        pmean2 = np.zeros(npar2)
        pstddev2 = np.zeros(npar2)
        for i in range(npar2):
            pmean2[i] = parlimits2[i]['mean'] 
            pstddev2[i] =  (parlimits2[i]['stddev'] + parlimits2[i]['stddev']) / 2.

        

# Perform mpfit on slightly randomly perturbed intitial guesses
        for i in xrange(nloop):
            randn1 = np.random.normal(size=npar1)
            p1 = (randn1 * pstddev1) + pmean1 
            if (p1[0] <= 0.): p1[0] = 0.
            if (p1[0] > 90.): p1[0] = 90.
            if (p1[1] <= 20.): p1[1] = 20.

            randu2 = np.random.normal(size=npar2)
            p2 = (randu2 * pstddev2) + pmean2
            if (p2[0] <= 20.): p2[0] = 20.
            if (p2[0] > 90.): p2[0] = 90.
            if (p2[1] <= 0.): p2[1] = 0.
            if (p2[1] > 90.): p2[1] = 90.

            loopnum=i+startnum
            

            if (modelnum == 0):
                pngname = objname + '_mpfit_soliddisk_cone' + `loopnum` + '.png'
                mpfit_diskcone(velfitsname, objname=objname, piguess1=p1, piguess2=p2, highcut=highcut,\
                                   lowcut=lowcut, loopnum=loopnum, pngname=pngname, cone_oangle2=20)
                bestparfile = objname + '_mpfit_soliddisk_cone_bestparms'

            if (modelnum == 1):
                pngname = objname + '_mpfit_soliddisk' + `loopnum` + '.png'
                mpfit_diskcone(velfitsname, objname=objname, piguess1=p1, piguess2=p2, fracdisk=1, highcut=highcut,\
                               lowcut=lowcut, loopnum=loopnum, pngname=pngname)
                bestparfile = objname + '_mpfit_soliddisk_bestparms'

            if (modelnum == 2):
                pngname = objname + '_mpfit_cone' + `loopnum` + '.png'
                mpfit_diskcone(velfitsname, objname=objname, piguess1=p1, piguess2=p2, fracdisk=0, highcut=highcut,\
                               lowcut=lowcut, loopnum=loopnum, pngname=pngname, cone_area_frac=1.0, frontonly=0)
                bestparfile = objname + '_mpfit_cone_bestparms'

            if (loopnum % 10 == 0):
                parmfile = open(bestparfile, 'wb')
                pickle.dump(parlist, parmfile)
                parmfile.close()
 
        parmfile = open(bestparfile, 'wb')
        pickle.dump(parlist, parmfile)
        parmfile.close()
        
        list(parlist.pop() for z in xrange(len(parlist)))






def mpfit_diskcone(velfitsname, objname='tmp', piguess1=[0.,100.,0.,0.,0.,0.], piguess2=[45.,0.,0.,10.,0.,0.,0.], fracdisk=-1,
                   highcut=300, lowcut=-300, loopnum=0, pngname='mpfit.png', cone_area_frac=1.0, frontonly=0,
                   backonly=0, cone_oangle2=0):

    fits = pyfits.open(velfitsname)
    fitsdata = fits[0].data
    fitsdim = fitsdata.shape
    zeroarr = np.zeros([fitsdim[0],fitsdim[1]])
    onearr = zeroarr + 1

    mask = mkmask(fitsdata, lowcut=lowcut, highcut=highcut)

    ey = onearr

    fa = {'y':fitsdata, 'err':ey, 'mask':mask, 'cone_area_frac':cone_area_frac, 'frontonly':frontonly,
          'backonly':backonly, 'cone_oangle2':cone_oangle2}
    parbase={'value':0., 'fixed':0, 'limited':[0,0], 'limits':[0.,0.], 'step':0.}
    parinfo1=[]
    parinfo2=[]

    for i in range(len(piguess1)):
        parinfo1.append(copy.deepcopy(parbase))
    for i in range(len(piguess1)): 
        parinfo1[i]['value']=piguess1[i]

    parinfo1[0]['limited'] = [1,1]
    parinfo1[0]['limits'] = [0., 90]
    parinfo1[0]['step'] = 10
    parinfo1[1]['limited'] = [1,0]
    parinfo1[1]['limits'] = [50., 10000.]
    parinfo1[1]['step'] = 20

    parinfo1[2]['step'] = 10
    parinfo1[3]['step'] = 0.2
    parinfo1[4]['step'] = 0.2
    parinfo1[5]['step'] = 20

    p0=[]

    for i in range(len(piguess2)):
        parinfo2.append(copy.deepcopy(parbase))

    for i in range(len(piguess2)): 
        parinfo2[i]['value']=piguess2[i]
    
    #Keep inclination between 0 and 90 degrees
    parinfo2[0]['limited'] = [1,1]
    parinfo2[0]['limits'] = [25, 90]
    parinfo2[0]['step'] = 10
    parinfo2[1]['limited'] = [1,1]
    parinfo2[1]['limits'] = [0, 90]
    parinfo2[1]['step'] = 10
    parinfo2[2]['step'] = 10
    parinfo1[2]['limited'] = [1,1]
    parinfo1[2]['limits'] = [-360., 360]
    parinfo2[3]['limited'] = [1,0]
    parinfo2[3]['limits'] = [0.0, 10000.]
    parinfo2[3]['step'] = 5
    parinfo2[4]['step'] = 0.2
    parinfo2[5]['step'] = 0.2
    parinfo2[6]['step'] = 20

    parinfo_fracdisk = parbase
    if (fracdisk == -1): #if fracdisk is not set, initial value will be picked from a random uniform distribution
            parinfo_fracdisk['value'] = np.random.uniform(low=0.0, high=1.0, size=1)
            parinfo_fracdisk['fixed'] = 0
    if (fracdisk > -0.5): #if fracdisk is set to be fixed
            parinfo_fracdisk['value'] = fracdisk
            parinfo_fracdisk['fixed'] = 1
    parinfo_fracdisk['limited'] = [1,1]
    parinfo_fracdisk['limits'] = [0., 1.]
    parinfo_fracdisk['step'] = 0.3

    #Combine the two parinfo's plus parinfo for fraction of disk
    parinfo=[]
    for i in range(len(piguess1)):
        parinfo.append(parinfo1[i])
        p0.append(piguess1[i])
    for i in range(len(piguess2)):
        parinfo.append(parinfo2[i])
        p0.append(piguess2[i])   
    parinfo.append(parinfo_fracdisk)
    p0.append(parinfo_fracdisk['value'])

    m = mpfit(myfunct_diskcone_mp, p0, parinfo=parinfo,functkw=fa, ftol=1.E-4)
    print velfitsname, loopnum, m.params

    
#Create best fit model for plotting
    binsize = 30 // fitsdata.shape[0]    
#For disk component
    diangle=m.params[0]
    vc = m.params[1]
    pa = m.params[2]
    centeroffset=np.int_([m.params[3] * binsize,m.params[4] * binsize])
    veloffset = m.params[5]
    radius1=7

    fitmodel1 = rotdisk_solid(iangle=diangle, vc=vc, pa=pa, radius=radius1, centeroffset=centeroffset, veloffset=veloffset)
    rfitmodel1 = congrid.congrid(fitmodel1,fitsdim)

#For bicone component, assume 20 degree 'cone thickness'
    radius2=30
    halfcone = 0
    depth = 3
    k2 = 0.
    vmax = 2000.

    center= np.int_([m.params[10] * binsize,m.params[11] * binsize])
    oangle1 = m.params[6]
    ##oangle2 = oangle1 - 20
    oangle2=cone_oangle2
    if (oangle2 < 0.): oangle2 = 0
    iangle2 = m.params[7]
    pa2 = m.params[8]
    k1 = m.params[9] / binsize
    veloffset = m.params[12]

    fitmodel2 = proj_bicone(radius=radius2, oangle1=oangle1, oangle2=oangle2,\
                           pa=pa2, iangle=iangle2, k1=k1, k2=k2, vmax=vmax, \
                           frontonly=frontonly, backonly=backonly, halfcone=halfcone,center=center,\
                           depth=depth, veloffset=veloffset)
    fl = np.int(fitmodel2.shape[0] * cone_area_frac / 2.)
    hl = np.int(fitmodel2.shape[0] / 2.)
    fracmodel2 = fitmodel2[hl-fl:hl+fl,hl-fl:hl+fl]
    
    rfitmodel2 = congrid.congrid(fracmodel2,fitsdim)

    fitmodel = (rfitmodel1 * m.params[13]) +(rfitmodel2 * (1. - m.params[13]))   
    rfitmodel = congrid.congrid(fitmodel,fitsdim)
    
    plt.clf()
    gridbicone = rfitmodel2  * (1. - m.params[13])
    griddisk = rfitmodel1 * m.params[13]
    grid = rfitmodel*mask
    grid2 = fitsdata*mask
    grid3 = grid2 - grid

    fig, axes = plt.subplots(nrows=2, ncols=3,figsize=(7,7))
    axes[0][0].imshow(grid, interpolation='none', origin='lower', vmin=lowcut, vmax=highcut)
    axes[0][1].imshow(gridbicone, interpolation='none', origin='lower', vmin=lowcut, vmax=highcut)
    axes[0][2].imshow(griddisk, interpolation='none', origin='lower', vmin=lowcut, vmax=highcut)
    axes[1][0].imshow(grid2, interpolation='none', origin='lower', vmin=lowcut, vmax=highcut)
    axes[1][1].imshow(grid3, interpolation='none', origin='lower', vmin=-100, vmax=100)


    plt.savefig(pngname)

    bestparm1 = m.params[0:6]
    bestparm2 = m.params[6:13]
    print bestparm2
    bestfracdisk = m.params[13]
    lowestchi2 = sum(sum(grid3**2.))
    parlist.append(FitParms(objname, loopnum, bestparm1, bestparm2, bestfracdisk, lowestchi2,frontonly,backonly,oangle2))







def myfunct_diskcone_mp(p0, fjac=None, y=None, err=None, mask=None, cone_area_frac=1.0, frontonly=0,
                        backonly=0, cone_oangle2=0):

#The fraction of each models
    fracdisk = p0[13]
    fraccone = 1 - fracdisk

#For disk component
    diangle=p0[0]
    vc = p0[1]
    pa = p0[2]
    centeroffset=[p0[3],p0[4]]
    veloffset = p0[5]
    radius1=7

    model1 = rotdisk_solid(iangle=diangle, vc=vc, pa=pa, radius=radius1, centeroffset=centeroffset, veloffset=veloffset)

#For bicone component, assume 20 degree 'cone thickness'
    radius2=30
    halfcone = 0
    depth = 3
    k2 = 0.
    vmax = 2000.
    
    binsize = radius2 // y.shape[0]
    center= np.int_([p0[10] * binsize,p0[11] * binsize])
    oangle1 = p0[6]
    ##oangle2 = oangle1 - 20
    oangle2 = cone_oangle2
    if (oangle2 < 0.): oangle2 = 0
    biangle = p0[7]
    pa = p0[8]
    k1 = p0[9] / binsize
    veloffset = p0[12]

    #If opening angle is too small and inclination is close to 90 the fracmodel shape will be smaller than y.shape, causing problem for the congrid.congrid routine
    model2 = proj_bicone(radius=radius2, oangle1=oangle1, oangle2=oangle2,\
                           pa=pa, iangle=biangle, k1=k1, k2=k2, vmax=vmax, \
                           frontonly=frontonly, backonly=backonly, halfcone=halfcone,center=center,\
                           depth=depth, veloffset=veloffset)

    # Retrieve the fractional area of the bicone model as specified by the cone_area_frac variable
    fl = np.int(model2.shape[0] * cone_area_frac / 2.)
    hl = np.int(model2.shape[0] / 2.)
    fracmodel2 = model2[hl-fl:hl+fl,hl-fl:hl+fl]
    #If opening angle is too small and inclination is close to 90 the fracmodel shape will be smaller than y.shape, causing problem for the congrid.congrid routine

    rmodel1 = congrid.congrid(model1,y.shape)
    rmodel2 = congrid.congrid(fracmodel2,y.shape)
    rmodel = (fracdisk*rmodel1) + (fraccone*rmodel2)

    goodvalidx = np.where(mask > 0)
    vals = (y-rmodel)/err
    listvals = np.ravel(vals[goodvalidx])

    status = 0
    return [status, listvals]





def proj_bicone(radius=30, oangle1=30., oangle2=15., pa=0., iangle=0., k1=1., k2=0., vmax=2000.,
                frontonly=0, backonly=0, halfcone=0, center=[0,0], depth=3, veloffset=0.):

    conevel = bicone(radius=radius, oangle1=oangle1, oangle2=oangle2, iangle=iangle, k1=k1, k2=k2, vmax=vmax, halfcone=halfcone)
    diam = radius * 2

    iangle = degtorad(iangle)

    cosi = math.cos(iangle)
    sini = math.sin(iangle)

    cospa = 1.
    sinpa = 0.
   
    z,y,x = conevel.nonzero()
    if (len(x) == 0):
            print 'x list is empty'
            print oangle1, oangle2, pa, k1, center, veloffset
    xinc = np.int_(np.round((x * cosi) - (z * sini))) 
    zinc = np.int_(np.round((x * sini) + (z * cosi)))
    yrot = np.int_(np.round((y * cospa) - (zinc * sinpa)))
    zrot = np.int_(np.round((y * sinpa) + (zinc * cospa)))
 
 # Uncomment for plot of bicone model
 #   fig = plt.figure()
 #   ax = fig.add_subplot(111, projection='3d')
 #   ax.set_xlabel("x")
 #   ax.set_ylabel("y")
 #   ax.set_zlabel("z")
 #   ax.set_xlim([min(xinc), max(xinc)])
 #   ax.set_ylim([min(yrot), max(yrot)])
 #   ax.set_zlim([min(zrot), max(zrot)])
 #   p=ax.scatter(xinc, yrot, zrot, zdir='z', c=conevel[z,y,x])
 #   plt.colorbar(p)
 #   plt.savefig("cone3d_solid.png")

#For fitting purpose need to make y,z dimension the same, cube will be summed in x dimension so x size just need to be big enough

    halfx = np.round((max(xinc) - min(xinc))/2)
    halfy = np.round((max(yrot) - min(yrot))/2)
    halfz = np.round((max(zrot) - min(zrot))/2)
    xpsize = max(xinc) - min(xinc) + 2
    ypsize = max(yrot) - min(yrot) + 2
    zpsize = max(zrot) - min(zrot) + 2
    squaresize = max([ypsize,zpsize])
    halfxsize = np.round(xpsize/2)
    halfsquare = np.round(squaresize/2) 
    xinc = xinc - min(xinc) - halfx + halfxsize 
    yrot = yrot - min(yrot) - halfy + halfsquare + center[0]
    zrot = zrot - min(zrot) - halfz + halfsquare + center[1]


    valid = np.where(((yrot < squaresize) & (zrot < squaresize) & (yrot >= 0) & (zrot >= 0)))


    projcube = np.zeros([squaresize,squaresize,xpsize])
    projslice = np.zeros([squaresize,squaresize])
    projcube[zrot[valid],yrot[valid],xinc[valid]] = conevel[z[valid],y[valid],x[valid]]
    mprojcube = np.ma.masked_array(projcube, (projcube == 0))

 # If frontonly set to one only take average of the first n=depth non-zero pixels for every projected velocity data point
    if (frontonly==1):
        for i in xrange(squaresize):
            for j in xrange(squaresize):
                tmpline = projcube[i,j,0:]
                nonzero = np.where(tmpline != 0)
                nonzero = nonzero[0]    # Turn tupple into array
                maxnz = min(len(nonzero)-1,depth-1)
                if (maxnz < 0): maxnz = 0
                projslice[i,j] = np.mean(tmpline[nonzero[0:maxnz]])
                if (np.isnan(projslice[i,j])): projslice[i,j] = 0.

    if (backonly==1):
        for i in xrange(squaresize):
            for j in xrange(squaresize):
                tmpline = projcube[i,j,0:]
                nonzero = np.where(tmpline != 0)
                nonzero = nonzero[0]    # Turn tupple into array
                maxnz = min(len(nonzero)-1,depth-1)
                lastidx = len(nonzero)-1
                if (maxnz < 0): maxnz = 0
                if (lastidx < 0): lastidx = 0     
                projslice[i,j] = np.mean(tmpline[nonzero[lastidx-maxnz:lastidx]])
                if (np.isnan(projslice[i,j])): projslice[i,j] = 0.


    if ((frontonly==0) and (backonly==0)):
        projslice = np.mean(mprojcube.data, axis=2)


    rotprojslice = ndimage.interpolation.rotate(projslice, pa)
    rotprojslice = rotprojslice + veloffset
    
    return rotprojslice





def bicone(radius=7, oangle1=30., oangle2=15., iangle=0., k1=1., k2=0., vmax=2000., halfcone=0):

"""
    Produce 3-D bicone structure with velocities projected along the observer's line of sight

    size = number of pixel per axis used for the 3-D array
    oangle1 = outer opening angle of the cone in degrees
    oangle2 = inner opening angle of the cone in degrees
    iangle = inclination angle along observe's line of sight (parallel to x-axis)
    pa = position angle of the cone in directions parallel to the sky
    inc = inclination of the cone (tilt w.r.t. line of sight)
    k1 = va = k1 * r, proportionality constant used to define increase of velocity with radius
    k2 = vd = -k2 * r, proportionality constant used to define decrease of velocity with radius
    frontonly = 0 for the full cone, frontonly=1 to get only the front portion facing the observer
"""

    #k1 = zero causes errors, set k1 to a very small number if input k1 = 0
    if (k1 == 0.):
        k1 = 0.1
    
    

    diam = radius * 2

    oangle1 = degtorad(oangle1)
    oangle2 = degtorad(oangle2)
    iangle = degtorad(iangle)

    cube = np.zeros([diam,diam,diam]) #Create a 3-D array of specified size
    indices = np.where(cube == 0)

    xcenter = radius  #Set the center of the x,y axis at the center of cube
    ycenter = radius
    zcenter = radius

    coord3d = get3dcoord(size=[diam,diam,diam], center=[zcenter,ycenter,xcenter])

    #Create 3 arrays containig x,y,z coordinate of each array element
    xarr = coord3d[0]
    yarr = coord3d[1]
    zarr = coord3d[2]

    r2 = (xarr**2. + yarr**2.)**0.5 #Radii from z-axis
    r3 = (xarr**2. + yarr**2. + zarr**2.)**0.5 #Radii from nucleus
    rmax = np.absolute(zarr[0:,0,0]) * math.tan(oangle1)
    rmin = np.absolute(zarr[0:,0,0]) * math.tan(oangle2)

    conevel = r3 * k1 #Velocity array 

    #If decceleration parameters are defined, calculated deccelerated velocities
    if ((k2 > 0.) and (vmax < 10000.)):
        dec_idx = np.where(conevel > vmax)
        conevel[dec_idx] = vmax - (k2 * r3[dec_ix])
        neg_idx = np.where(conevel < 0)
        conevel[neg_idx] = 0.
    
    for i in xrange(diam):
        tmp_r2_slice = r2[i,0:,0:]
        outpix = (tmp_r2_slice > rmax[i])
        inpix = (tmp_r2_slice < rmin[i])
        outidx = np.where(outpix | inpix)
        conevel[i,outidx[1],outidx[0]] = 0.

    if (halfcone == 1):
        # If halfcone=1 return only half the cone (front half facing the observer when iangle=0)
        backidx = np.where(xarr <= 0)
        conevel[backidx] = 0.


    # Make observer's line of sight parallel to x-axis
    alos = goodarctan(xarr,zarr) # should be xarr instead of yarr, don't know why this works.....
    negz = np.where(zarr <= 0)
    posz = np.where(zarr > 0)
    alos = alos + iangle

    conevel = conevel * np.cos(alos)

    return conevel




def rotdisk_solid(iangle=45.,vc=100., pa=0, radius=5, centeroffset=[0,0], veloffset=0.):
    xcoords=[0]
    ycoords=[0]
    vels=[0]

    pi = np.pi

    for i in range(radius):
        ring = rotring_solid(iangle=iangle, vc=vc, radius=i+1, maxrad=radius)
        xcoords = np.concatenate([xcoords,ring[0]])
        ycoords = np.concatenate([ycoords,ring[1]])
        vels = np.concatenate([vels,ring[2]])
        
    xcoords = xcoords + centeroffset[1]
    ycoords = ycoords + centeroffset[0]

    

    size = radius * 2 + 1
    velslice = np.zeros([size,size])
    xidx = np.int_(np.round(xcoords + radius))
    yidx = np.int_(np.round(ycoords + radius))
    xvalid = (xidx < size) & (xidx >= 0)
    yvalid = (yidx < size) & (yidx >= 0)
    valid = np.where(xvalid & yvalid)
    velslice[xidx[valid],yidx[valid]] = vels[valid]
    rotvelslice = ndimage.interpolation.rotate(velslice, pa)
    rotvelslice = rotvelslice + veloffset # Offset the whole velocity field by a constant

    return rotvelslice
        
        



def rotring_solid(iangle=0., vc=100., radius=5, maxrad=10):
    #Velocity as a function of x-y positions in a ring within a rotating disk
    #Generate an array of x-y coordinates within a ring, sort by x, then sort by y

    pi = np.pi
    iangle = iangle * pi / 180.

    ringcoord = getringcoord(radius=radius)
    ringx = ringcoord[0]
    ringy = ringcoord[1]

    psi = goodarctan(ringx,ringy)
 

    vel = vc * radius / maxrad * 2.
    ringvel = vel * math.sin(iangle) * np.cos(psi)
    ringy = ringy * math.cos(iangle)
    
    return [ringx, ringy, ringvel]





def getringcoord(radius=5):

    coord2d = get2dcoord(size=[radius*2+2,radius*2+2], center=[radius+1,radius+1])
    xarr = coord2d[0]
    yarr = coord2d[1]

    rad = (xarr**2. + yarr**2.)**0.5
    intrad = rad.astype(int)
    
    ringidx = np.where(intrad == radius)
    
    return [xarr[ringidx],yarr[ringidx]]
    



def mkmask(fitsdata, lowcut=-300, highcut=300):
    
    fitsdim = fitsdata.shape
    zeroarr = np.zeros([fitsdim[0],fitsdim[1]])
    onearr = zeroarr + 1

    goodvals = (fitsdata > lowcut) & (fitsdata < highcut)
    goodmask = np.where(goodvals, onearr, zeroarr)
    coord2d = get2dcoord(size=[fitsdim[0],fitsdim[1]], center=[0,0])
    xcoord = coord2d[0]
    ycoord = coord2d[1]

    goodx = xcoord[np.nonzero(goodmask)]
    goody = ycoord[np.nonzero(goodmask)]
    datacoord = [goodx,goody]


#Use hierarchical clustering to find the biggest cluster and eliminate the random pixels that land in between the velocity cuts
    threshold = 1. # The distance threshold for separate clusters, set to 1.5 such that no pixels in cluster is more than 1 pixel apart (diagonals ok)
    clusters = hclust.fclusterdata(np.transpose(datacoord), threshold, criterion='distance')
    counts = Counter(clusters).most_common(1)
    bigclusnum = counts[0][0]
    cluster1 = np.where(clusters == bigclusnum)
    finalmask = zeroarr
    finalmask[goodx[cluster1[0]], goody[cluster1[0]]] = 1.
    finalmask = finalmask.T

    return finalmask




def get2dcoord(size=[20,20], center=[10,10]):
    #Size and center needs to be 3 x 1 vector
    aslice = np.zeros([size[0],size[1]]) #Create a 3-D array of specified size
    indices = np.where(aslice == 0)

    #Create 3 arrays containig x,y,z coordinate of each array element
    xarr = indices[1] - center[1]
    yarr = indices[0] - center[0]
    xarr = xarr.reshape(size[0],size[1])
    yarr = yarr.reshape(size[0],size[1])

    return [xarr,yarr]





def get3dcoord(size=[20,20,20], center=[10,10,10]):
    #Size and center needs to be 3 x 1 vector
    cube = np.zeros([size[0],size[1],size[2]]) #Create a 3-D array of specified size
    indices = np.where(cube == 0)

    #Create 3 arrays containig x,y,z coordinate of each array element
    xarr = indices[2] - center[2]
    yarr = indices[1] - center[1]
    zarr = indices[0] - center[0]
    xarr = xarr.reshape(size[0],size[1],size[2])
    yarr = yarr.reshape(size[0],size[1],size[2])
    zarr = zarr.reshape(size[0],size[1],size[2])

    del cube
    return [xarr,yarr,zarr]






def goodarctan(x,y):
    psi = np.arctan(np.float_(y)/np.float_(x))
    posx = x > 0
    negx = x < 0
    zerox = x == 0
    posy = y > 0
    negy = y <= 0
    q2idx = np.where(posy & negx)
    q3idx = np.where(negy & negx)
    q4idx = np.where(negy & posx)
    psi[q2idx] = psi[q2idx] + pi
    psi[q3idx] = psi[q3idx] + pi
    psi[q4idx] = psi[q4idx] + (2. * pi)

    poszerox = np.where(zerox & posy)
    negzerox = np.where(zerox & negy)
    psi[poszerox] = pi/2.
    psi[negzerox] = 3.*pi/2.

    return psi


def radtodeg(x):
    pi = np.pi
    y = x * 180. / pi
    return y

def degtorad(x):
    pi = np.pi
    y = x * pi / 180.
    return y


def rebin(a, shape):
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)



def repeat_mcfit_all(nrepeat = 50,startnum = 0):
    
    nloop = nrepeat - startnum

    inifile = open('InitialInput', 'rb')
    inilist = pickle.load(inifile)
    inifile.close()

    for k in range(1,2):
        objname = inilist[k].objname
        piguess1 = inilist[k].piguess1
        piguess2 = inilist[k].piguess2
        velfitsname = inilist[k].velfitsname
        highcut = inilist[k].highcut
        lowcut = inilist[k].lowcut
        
        for i in xrange(nloop):
            mcfit_all(velfitsname, objname=objname, piguess1=piguess1, piguess2=piguess2, highcut=highcut, lowcut=lowcut, loopnum=i+startnum)

        bestparfile = objname + '_bestparms'
        parmfile = open(bestparfile, 'wb')
        pickle.dump(parlist, parmfile)
        parmfile.close()
        list(parlist.pop() for z in xrange(len(parlist)))


def mcfit_all(velfitsname, objname='tmp', piguess1=[0.,100.,0.,0.,0.,0.], piguess2=[45.,0.,0.,10.,0.,0.,0.], highcut=300, lowcut=-300, loopnum=0):

    nsample = 1000

#Disk Component
    npar1 = 6

    pardict1 = {'name':'tmp', 'mean':0., 'stddev':0}
    parnames1 = ['iangle', 'vc', 'pa', 'xoff', 'yoff', 'veloff']
    parlimits1 = []
    for i in range(npar1): 
        parlimits1.append(copy.deepcopy(pardict1))
    for i in range(npar1): 
        parlimits1[i]['name'] = parnames1[i]

    parlimits1[0]['stddev'] = 20.
    parlimits1[1]['stddev'] = 100.
    parlimits1[2]['stddev'] = 20.
    parlimits1[3]['stddev'] = 1.
    parlimits1[4]['stddev'] = 1.
    parlimits1[5]['stddev'] = 50.


#Bicone Component
    npar2 = 7

    pardict2 = {'name':'tmp', 'mean':0., 'stddev':0}
    parnames2 = ['oangle1','iangle', 'pa', 'k1', 'xoff', 'yoff', 'veloff']
    parlimits2 = []
    for i in xrange(npar2): 
        parlimits2.append(copy.deepcopy(pardict2))
    for i in xrange(npar2): 
        parlimits2[i]['name'] = parnames2[i]

    parlimits2[0]['stddev'] = 15.
    parlimits2[1]['stddev'] = 20.
    parlimits2[2]['stddev'] = 20.
    parlimits2[3]['stddev'] = 20.
    parlimits2[4]['stddev'] = 2.
    parlimits2[5]['stddev'] = 2.
    parlimits2[6]['stddev'] = 60.

    for i in range(npar1): 
        parlimits1[i]['mean'] = piguess1[i]
    for i in xrange(npar2): 
        parlimits2[i]['mean'] = piguess2[i]
    pngname = objname + '_mcfit_disk' + `loopnum` + '.png'
    mcfit(velfitsname, parlimits1, npar1, parlimits2, npar2, nsample, highcut=highcut, lowcut=lowcut, pngname=pngname, fracdisk=1, objname=objname, trialnum=loopnum)




def mcfit(fitsname, parlimits1, npar1, parlimits2, npar2, nsample, highcut=250, lowcut=-250, fracdisk=-1, objname='tmp', trialnum=0):

# parlimits = a dictionary which sets the lower and upper limits of the parameter space to explore

    fits = pyfits.open(fitsname)
    fitsdata = fits[0].data
    y = fitsdata
    
    fitsdim = fitsdata.shape
    zeroarr = np.zeros([fitsdim[0],fitsdim[1]])
    onearr = zeroarr + 1

    mask = mkmask(fitsdata, lowcut=lowcut, highcut=highcut)
    
    
    np.random.seed()
    chi2 = np.zeros(nsample)



# For a random sampling of parameter space with Guassian distribution

    # Disk component
    pmean1 = np.zeros(npar1)
    pstddev1 = np.zeros(npar1)
    for i in range(npar1):
        pmean1[i] = parlimits1[i]['mean'] 
        pstddev1[i] =  (parlimits1[i]['stddev'] + parlimits1[i]['stddev']) / 2.

    # Bicone component
    pmean2 = np.zeros(npar2)
    pstddev2 = np.zeros(npar2)
    for i in range(npar2):
        pmean2[i] = parlimits2[i]['mean'] 
        pstddev2[i] =  (parlimits2[i]['stddev'] + parlimits2[i]['stddev']) / 2.

    for j in range(nsample):
        randn1 = np.random.normal(size=npar1)
        p1 = (randn1 * pstddev1) + pmean1

        randu2 = np.random.normal(size=npar2)
        p2 = (randu2 * pstddev2) + pmean2
        if (p2[0] <= 20.): p2[0] = 20.
        if (p2[0] > 90.): p2[0] = 90.
        if (p2[1] <= 0.): p2[1] = 0.
        if (p2[1] > 90.): p2[1] = 90.

        if (fracdisk == -1): #if fracdisk is not set
            fracdisk = np.random.uniform(low=0.0, high=1.0, size=1)

        fraccone = 1. - fracdisk

        
        chivals = myfunct_diskcone(p1, p2, fraccone, fracdisk, y=fitsdata, err=onearr, mask=mask)
        currentchi2 = sum(chivals[1] ** 2.)
        if (j == 0):
            bestparm1 = p1
            bestparm2 = p2
            bestfracdisk = fracdisk
            lowestchi2 = currentchi2
        else:
            if (currentchi2 < lowestchi2):
                bestparm1 = p1
                bestparm2 = p2
                bestfracdisk = fracdisk
                lowestchi2 = currentchi2

    parlist.append(FitParms(objname, trialnum, bestparm1, bestparm2, bestfracdisk, lowestchi2))

    print pngname
    print 'Disk parm = ', bestparm1
    print 'Cone parm = ', bestparm2
    print 'Disk fraction = ', bestfracdisk 
    print 'Lowest Chi Square = ', lowestchi2

    fitmodel1 = rotdisk_solid(iangle=bestparm1[0], vc=bestparm1[1], pa=bestparm1[2], radius=7, centeroffset=[bestparm1[3],bestparm1[4]], veloffset=bestparm1[5])

    binsize = 30 // fitsdata.shape[0]
    center = np.int_([bestparm2[4] * binsize,bestparm2[5] * binsize])
    fitmodel2 = proj_bicone(radius=30, oangle1=bestparm2[0], oangle2=5.,\
                        iangle=bestparm2[1], pa=bestparm2[2], k1=bestparm2[3],\
                        center=center, veloffset=bestparm2[6])
    rfitmodel2 = congrid.congrid(fitmodel2,fitsdim)
    fitmodel = (fitmodel1*fracdisk) + (rfitmodel2*fraccone)
    plt.clf()
    grid = fitmodel*mask
    grid2 = fitsdata*mask
    grid3 = grid2 - grid
    fig, axes = plt.subplots(nrows=3, ncols=1,figsize=(7,7))
    axes[0].imshow(grid, interpolation='none', origin='lower', vmin=lowcut, vmax=highcut)
    axes[1].imshow(grid2, interpolation='none', origin='lower', vmin=lowcut, vmax=highcut)
    axes[2].imshow(grid3, interpolation='none', origin='lower', vmin=-200, vmax=200)

    pngname = objname + '_mcfit' + `loopnum` + '.png'
    plt.savefig(pngname)


    return [bestparm1, bestparm2, fitmodel]



def myfunct_diskcone_mc(p1, p2, fracdisk, fraccone, fjac=None, y=None, err=None, mask=None):

#For disk component
    iangle=p1[0]
    vc = p1[1]
    pa = p1[2]
    centeroffset=[p1[3],p1[4]]
    veloffset = p1[5]
    radius1=7

    model1 = rotdisk(iangle=iangle, vc=vc, pa=pa, radius=radius1, centeroffset=centeroffset, veloffset=veloffset)

#For bicone component, assume 20 degree 'cone thickness'
    radius2=30
    frontonly = 0
    halfcone = 0
    depth = 3
    k2 = 0.
    vmax = 2000.
    
    binsize = radius2 // y.shape[0]
    center= np.int_([p2[4] * binsize,p2[5] * binsize])
    oangle1 = p2[0]
    oangle2 = oangle1 - 20
    if (oangle2 < 0.): oangle2 = 0
    iangle = p2[1]
    pa = p2[2]
    k1 = p2[3] / binsize
    veloffset = p2[6]

    #If opening angle is too small and inclination is close to 90 the fracmodel shape will be smaller than y.shape, causing problem for the congrid.congrid routine
    model2 = proj_bicone(radius=radius2, oangle1=oangle1, oangle2=oangle2,\
                           pa=pa, iangle=iangle, k1=k1, k2=k2, vmax=vmax, \
                           frontonly=frontonly, halfcone=halfcone,center=center,\
                           depth=depth, veloffset=veloffset)

    rmodel1 = congrid.congrid(model1,y.shape)
    rmodel2 = congrid.congrid(model2,y.shape)
    rmodel = (fracdisk*rmodel1) + (fraccone*rmodel2)

    vals = (y-rmodel)/err*mask
    listvals = np.ravel(vals)

    status = 0
    return [status, listvals]





