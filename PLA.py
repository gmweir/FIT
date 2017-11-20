# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 15:21:45 2016

@author: gawe
"""
# ======================================================================== #
# ======================================================================== #

from __future__ import absolute_import, with_statement, absolute_import, \
                       division, print_function, unicode_literals

# ======================================================================== #

import numpy as _np
import scipy.signal as _dsp
#import interp1D
#from scipy.interpolate import interp1d
import matplotlib.pyplot as _plt
from pybaseutils import interp

# function [uavg,uvar,tavg] = phase_lock_average(r_t,u_t,dt,fref,tdelay,plotit,varname)
#
#
# ======================================================================== #

def getLPF(tau, Fs, fref=None, verbose=True):
    #Create the digital low pass filter that is necessary for the phase lock amplifier
    if tau == 0 and fref is not None:
        #Filtering time-constant
        tau = 5.0/fref
    #end if
    if verbose: print('Using a filtering time-constant of %f s'%(tau,))
    lowpass_n, lowpass_d = _dsp.butter(2, 2.0/(Fs*tau), btype='low')
    # lowpass_n, lowpass_d =  _dsp.bessel(2, 2.0/(Fs*tau), 'lowpass')
    return lowpass_n, lowpass_d 

def applyLPF(lowpass_n, lowpass_d, u_t, verbose=True):
    nch = u_t.shape[1]
    for ii in range(nch):
        if verbose: print('Non-causal filtering channel %i of %i'%(ii,nch))
        u_t[:, ii] = _dsp.filtfilt( lowpass_n, lowpass_d, u_t[:, ii]) #(Non-Causal) LPF
    #endif        
    return u_t
    
def lockin_heterodyne(r_t, u_t, Fs, tau=1.0/10.0, fref=None, verbose=True):

    u_dc = _np.zeros_like(u_t)
    nch = u_t.shape[1]
    for ii in range(nch):
        u_dc[:,ii] = r_t*u_t[:,ii]
    # end for

    # get a DC filter
    lowpass_n, lowpass_d = getLPF(tau, Fs, fref, verbose)        
    
    # apply the DC low pass filter to the mixed signal
    u_dc = applyLPF(lowpass_n, lowpass_d, u_dc, verbose)
    return u_dc

# =================================== #
    
class SimPLL(object):
    def __init__(self, lf_bandwidth):
        self.phase_out = 0.0
        self.freq_out = 0.0
        self.vco = _np.exp(1j*self.phase_out)
        self.phase_difference = 0.0
        self.bw = lf_bandwidth
        self.beta = _np.sqrt(lf_bandwidth)

    def update_phase_estimate(self):
        self.vco = _np.exp(1j*self.phase_out)

    def update_phase_difference(self, in_sig):
        self.phase_difference = _np.angle(in_sig*_np.conj(self.vco))

    def step(self, in_sig):
        # Takes an instantaneous sample of a signal and updates the PLL's inner state
        self.update_phase_difference(in_sig)
        self.freq_out += self.bw * self.phase_difference
        self.phase_out += self.beta * self.phase_difference + self.freq_out
        self.update_phase_estimate()


def generate_RefSig(in_sig, num_samples=500):
    import matplotlib.pyplot as plt

    lf_bandwidth=0.01         # pll bandwidth
    zeta = 1.0/_np.sqrt(2)    # pll damping factor    
    K    = 1e3                # pll loop gain
    phase_offset = 3.0        # carrier phase offset
    frequency_offset = -0.2   # carrier frequency offset

    # loop filter parameters (active PI "proportional plus integration" filter design)
    tau1 = K/(lf_bandwidth*lf_bandwidth)
    tau2 = 2*zeta/lf_bandwidth

    # feed forward coefficients (numerator)
    b0 = (4.0*K/tau1)*(1.0+tau2/2.0)
    b1 = (8.0*K/tau1)
    b2 = (4.0*K/tau1)*(1.0-tau2/2.0)
    
    # feed back coefficients (denominator)
    a0 = 1.0  # this is implied in the code
    a1 = -2.0
    a2 = 1.0

    print('b=[b0:%12.8f, b1:%12.8f, b2:%12.8f]\n'%(b0, b1, b2))    
    print('a=[a0:%12.8f, a1:%12.8f, a2:%12.8f]\n'%(a0, a1, a2))

    # filter buffer
    v0 = 0.0
    v1 = 0.0
    v2 = 0.0

    # initialize the system
    phi = phase_offset.copy()      # input signal's initial phase
    phi_hat = 0.0                  # initial phase of PLL    

    for ii in range(num_samples):

        # Compute input sinusoide and update phase
        x = _np.cos(phi) + 1j*_np.sin(phi)
        phi += frequency_offset
        
        # Compute PLL output from phase estimate
        y = _np.cos(phi_hat) + 1j*_np.sin(phi_hat)
        
        # Compute the error estimate
        delta_phi = _np.angle( x*_np.conj(y) )
        
        # print results
        print(" %6u %12.8f %12.8f %12.8f %12.8f %12.8f \n"%
            (ii, _np.real(x),_np.imag(x), _np.real(y), _np.imag(y), delta_phi) )
    
        # push result through loop filter, updating phase estimate
        #
        # advance the buffer
        v2 = v1.copy()   # shift center register to upper register
        v1 = v0.copy()   # shift lower register to center register
        # compute new lower register
        v0 = delta_phi - v1*a1 - v2*a2
        
        # compute new ouptut phase
        phi_hat = v0*b0 + v1*b1 + v2*b2
    # end for
        
    pll = SimPLL(lf_bandwidth)


    ref = []
    out = []
    diff = []
    for ii in range(0, num_samples - 1):
        in_sig = _np.exp(1j*phase_offset)
        phase_offset += frequency_offset
        pll.step(in_sig)
        ref.append(in_sig)
        out.append(pll.vco)
        diff.append(pll.phase_difference)
    #plt.plot(ref)
    plt.plot(ref, 'b')
    plt.plot(out, 'g')
    plt.plot(diff, 'r')
    plt.show()
    
# =================================== #
    
def LPF(r_t, u_t, dt=1.0, fref=0, tau=0, tdelay=0, plotit=1, varname='u_{avg}', verbose=True, refch=0):
    
    na   = _np.size( u_t, axis=0)
    tt   = dt*_np.arange(0,na,1)
    
    if fref == 0:
        if verbose: print('No reference frequency specified, calculating from peak in CPSD')
        import fft_pwelch

        #Calculate the phase difference between the two signals using one long
        #time-window to calculate the power spectra.
        tbounds = [tt[0],tt[-1]]
        Navr = 1
        windowoverlap = 0
        windowfunction = []
        useMATLAB = 1
        verbose = 1
        
        [freq,Pxy,Pxx,Pyy,Cxy,phi_xy,info] = \
            fft_pwelch( tt, r_t, u_t[:,refch], \
                       tbounds, Navr, windowoverlap, windowfunction, \
                       useMATLAB, plotit, verbose )
        
        
        ifref = index_max(Pxx)
        fref = freq( ifref )
    else:
        freq = Fs*_np.arange(0.0,1.0,1.0/na)
        if (na%2):
            #freq = Fs*(0:1/(nfft+1):1)
            freq = Fs*_np.arange(0.0,1.0,1.0/(na+1))
        #end if nfft is odd
        freq = freq[0:_np.floor(na/2)] #[Hz]
        
        ifref = _np.floor( 1+(fref-freq[0])/( freq[1]-freq[0] ) )
    #endif
    if verbose: print('Using a reference frequency of %f Hz'%(fref,))

    # ================================================================== #

    #Create the digital low pass filter that is necessary for the phase lock amplifier
    lowpass_n, lowpass_d = getLPF(tau=tau, Fs=Fs, fref=fref, verbose=verbose)
    
    # ================================================================== #

    debug = True
    if debug:

        #Calculate the frequency response of the lowpass filter,
        w, h = _dsp.freqz(lowpass_n, lowpass_d, worN=12000) #

        #Convert to frequency vector from rad/sample
        w = (Fs/(2.0*_np.pi))*w

        _plt.figure(num=3951)
        # _plt.clf(3951)
        _plt.subplot(3, 1, 1)
        _plt.plot( tt, u_t, 'k')
        _plt.ylabel('Signal', color='k')
        _plt.xlabel('t [s]')

        _plt.subplot(3, 1, 2)
        _plt.plot(w, 20 * _np.log10(abs(h)), 'b')
        _plt.xscale('log')
        _plt.grid(which='both', axis='both')
        _plt.plot(1.0/tau, 0.5*_np.sqrt(2), 'ko')
        _plt.axvline(1.0/tau, color='k')
        _plt.ylabel('|LPF| [dB]', color='b')
        _plt.xlabel('Frequency [Hz]')
        _plt.title('Digital LPF frequency response (Stage 1)')
        _plt.grid()
        _plt.axis('tight')
    #endif plotit

    u_t = applyLPF(lowpass_n, lowpass_d, u_t, verbose=verbose)
    
    if debug:
        _plt.subplot(3, 1, 3)
        _plt.plot(tt,u_t, 'k')
        _plt.ylabel('Filt. Signal', color='k')
        _plt.xlabel('t [s]')
    #EndDebug

    return u_t, tt, fref, tdelay, varname

def index_average(u_t, tt, fref, tdelay=0, plotit=1, varname='u_{avg}', verbose=True):
    
    try:
        nch = _np.size(u_t, axis=1)
    except:
        u_t = _np.atleast_2d(u_t).transpose()
        nch = 1 
    # end try
    nt = _np.size(u_t, axis=0)
    dt = tt[2]-tt[1]
    
    # ================================================================== #

    Tlock = 1.0/fref              #Period at which to lock
    nlock  = int( _np.floor(1+Tlock/dt) ) #Number of samples in period
    ndelay = int( _np.floor(1+tdelay/dt) ) #Number of samples each channel is delayed by
    # ndelay(tdelay<0) = -ndelay(tdelay<0);

    # Truncate to an integral number of periods
    ncut = nt%nlock
    if ncut>0:        
        u_t = u_t[:-ncut,:]
        tt = tt[:-ncut]
        nt = _np.size(u_t, axis=0)

    if _np.min(ndelay)<0:
        ndelay = ndelay - _np.min(ndelay);
    #endif

    # ================================================================== #

    uavg = _np.zeros( (nlock, nch), float)
    uvar = _np.zeros_like( uavg )

    if verbose: print('locking on a period of %i us over %i channels'%(int(1e6*Tlock),nch))
    for gg in range(nlock): #gg = 0:(nlock-1)
        if verbose: print('working on index %i of %i'%(gg,nlock))
        ttmp = _np.arange( dt*gg, tt[-1], Tlock )
        nseg = len(ttmp)

        uinterp = interp(xi=tt, yi=u_t, ei=None, xo=ttmp)
        uinterp = uinterp.reshape(nseg, nch)
        uavg[gg,:] = _np.mean( uinterp, axis=0 )
        uvar[gg,:] = _np.var(  uinterp, axis=0 )

#        for hh in range(nch): #hh = 1:nch;
#            # Using scipy's interp1d function
##            uinterp = interp1d( tt, u_t[hh,:], 'cubic' )
##            uavg[hh,gg] = _np.mean( uinterp( ttmp ) )
##            uvar[hh,gg] = _np.var(  uinterp( ttmp ) )           
#
#            uinterp = _np.interp( ttmp, tt, u_t[hh,:] )
#            uavg[hh,gg] = _np.mean( uinterp )
#            uvar[hh,gg] = _np.var(  uinterp )
#            
#            #Using the home-maded interp1D function (with error propagation)
##            [ uinterp, vinterp ] = interp1D( tt, u_t[hh,:], ttmp,0, 3 )
##
##            uavg[hh,gg] = _np.mean( uinterp )
##            uvar[hh,gg] = _np.var(  uinterp ) + _np.sum( vinterp )/( _np.shape(vinterp, axis=0) )**2
#        #endif
    #endif
    tavg = dt*_np.arange(0,nlock,1)  #dt*( 0:(nlock-1) )

    if plotit:
        _plt.figure()
        clrs = ['r', 'b', 'g', 'm', 'c', 'y', 'k']
        jj = 0
        for ii in range(nch):
           if jj > len(clrs):
              jj -= len(clrs)                
           _plt.plot(tavg, uavg[:,ii], clrs[jj]+'-')
           jj += 1
        # end for
        _plt.xlabel('t [s]')
        _plt.ylabel(varname)
        _plt.title('Phase Averaged Signal')
        _plt.show()
    #endif plotit

    return uavg,uvar,tavg

def phase_lock_average(r_t, u_t, dt=1.0, fref=0, tau=0, tdelay=0, plotit=1, varname='u_{avg}', verbose=True):

    if verbose: print('entering low pass filter section')
    u_t, tt, fref, tdelay, varname = \
        LPF(r_t=r_t, u_t=u_t, dt=dt, fref=fref, tau=tau, tdelay=tdelay, plotit=plotit, varname=varname, verbose=verbose)

    if verbose: print('entering index averaging section')
    uavg,uvar,tavg = \
        index_average(u_t=u_t, tt=tt, fref=fref, tdelay=tdelay, plotit=plotit, varname=varname, verbose=verbose)

    return uavg,uvar,tavg
    
# ========================================================================== #

def index_max(signal_in):
    return _np.argmax(signal_in)

# ========================================================================== #
# ========================================================================== #

if __name__ == "__main__":
    #Suggested test-case for index locking 
    Fs = 1.0e6       #[Hz]
    fref = 473.0   #[Hz]
#    tau = 5.0/fref  #Filtering time-constant,
    tau = 1.0/(3*fref)  #Filtering time-constant,
    dt = 1.0/Fs
    nch = 3
    tt = _np.arange(0, 1e3/fref,dt) 

#    #Suggested test-case
#    Fs = 1.0e3       #[Hz]
#    fref = 47.0   #[Hz]
#    tau = 5.0/fref  #Filtering time-constant,
#    dt = 1.0/Fs
#    nch = 3
#    tt = _np.arange(0, 2e5/fref,dt) #0:dt:(2e5/fref)
    
    #Noisy phase-shifted sine-wave
    u_t = _np.zeros((tt.shape[0],3), dtype=_np.float64)
    u_t[:,0] = 0.001*_np.sin(2.0*_np.pi*fref*tt-_np.pi/8.0) + 1.0*_np.random.standard_normal( (tt.shape[0], 1) ).flatten()
    u_t[:,1] = 0.01*_np.sin(2.0*_np.pi*fref*tt-3*_np.pi/8.0) + 1.0*_np.random.standard_normal( (tt.shape[0], 1) ).flatten()
    u_t[:,2] = 0.1*_np.sin(2.0*_np.pi*fref*tt-6*_np.pi/8.0) + 1.0*_np.random.standard_normal( (tt.shape[0], 1) ).flatten()
    u_t += 1.0
#    u_t = ( 5.0 + 0.05*_np.sin(2.0*_np.pi*fref*tt-_np.pi/8.0).reshape((tt.shape[0],1)))*_np.ones( (1,nch),float) \
#                        + 1.0*_np.random.standard_normal( (tt.shape[0], nch) )
#    u_t = _np.sin(2.0*_np.pi*fref*tt-_np.pi/8.0).reshape(tt.shape[0], nch) \
#        + 0.1*_np.random.standard_normal( ( tt.shape[0], nch) )
                        
    u_t = u_t.reshape((tt.shape[0], nch))                        

    import time
    now = time.time()
    if True:
        #Sine-wave reference signal
        r_t = 10.0+_np.sin(2*_np.pi*fref*tt)

        # #Nyquist filtering means no filtering by default.  This should really be lower
        # tau = (2*dt)
        
        #Square-wave
        print('Filtering, then index locking')    
        r_t = -1.0+_dsp.square(2*_np.pi*fref*tt,0.5)
                 
        uavg,uvar,tavg = phase_lock_average(r_t=r_t,u_t=u_t,dt=dt,fref=fref,tau=tau)
    else:
        print('Index locking')
        uavg,uvar,tavg = \
            index_average(u_t=u_t, tt=tt, fref=fref, tdelay=0, plotit=1, varname='index_avg')
    # endif
    print(time.time()-now)
#endif

# ========================================================================== #
# ========================================================================== #
