import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy.interpolate import interp1d


'''
Objects for interfacing with Michael Porter's models. 
Designed in analog with the types of objects called by read_shd, write_fieldflp, write_env, etc.
'''


class Source:
    def __init__(self, depth):
        self.depth = depth
        self.x = None
        self.y = None

class Dom:
    def __init__(self, ran, depth, offsets=None):
        self.range = ran # in km
        self.depth = depth
        if offsets is not None:
            self.offsets = offsets

class Pos:
    def __init__(self, Source, Dom):
        self.s = Source
        self.r = Dom

class Ice:
    def __init__(self, BumDen, eta, xi):
        self.BumDen=  BumDen
        self.eta = eta
        self.xi = xi

class SSPraw:
    def __init__(self,z, alphaR, betaR, rho, alphaI, betaI):
        """
        Input:
        z - numpy 1d array
            Depths the ssp is taken at
        alphaR - numpy 1d array
            ssp vals
        betaR - numpy 1d array
            shear speeds
        rho - numpy 1d array
            density vals at the points
        alphaI - numpy 1d array
            attenuation (p-wave)
        betaI - numpy 1d array
            shear attenuation
        """
        self.z = z
        self.alphaR = alphaR # sound speed in array
        self.betaR = betaR # shear wave speed
        self.rho = rho 
        self.alphaI = alphaI # atten. in array (alpha (z))
        self.betaI = betaI # shear attenuation

    def make_sspf(self):
        self.sspf = interp1d(self.z, self.alphaR)

    def interp_all(self):
        self.betaI_f = interp1d(self.z, self.betaI)
        self.betaR_f = interp1d(self.z, self.betaR)
        self.rho_f = interp1d(self.z, self.rho)
        self.alphaR_f = interp1d(self.z, self.alphaR)
        self.alphaI_f = interp1d(self.z, self.alphaI)
        return self.alphaR_f, self.betaR_f, self.rho_f, self.alphaI_f, self.betaI_f

class SSP:
    def __init__(self, raw, depth, NMedia, Opt=None, N=None, sigma=None, ranges=0):
        self.NMedia	= NMedia # number of media layers
        self.Opt = Opt # option: not sure what this is for
        self.N			=	N	 # array with num points in each layer, don't include one for halfpace
        if sigma is None:
            self.sigma = [0] * (NMedia + 1)
        else:
            self.sigma		=	sigma	 # ''interfacial roughness'', one for each layer? see Kuperman and ingenito 1977 for definition, include a point for the halfspace
        self.depth		=	depth # depth array for layers
        self.ranges = ranges
        self.raw = raw # list of raw ssp profile
        self.sspf = None

    def make_sspf(self):
        for raw in self.raw:
            raw.make_sspf()
        if self.NMedia > 1:
            self.ssp_vals = np.zeros((len(self.depth), 1))
            if self.NMedia == 2:
                layer_depth = self.raw[0].z[-1]
                self.sspf = lambda z: np.piecewise(z, [z<layer_depth, z>=layer_depth], [lambda z: self.raw[0].sspf(z), lambda z: self.raw[1].sspf(z)])
            elif self.NMedia == 3:
                layer_depth_one = self.raw[0].z[-1]
                layer_depth_two = self.raw[1].z[-1]
                func1 = lambda z: np.piecewise(z, [z<layer_depth_one, z>= layer_depth_one], [lambda z: self.raw[0].sspf(z), lambda z: self.raw[1].sspf(z)])
                self.sspf = lambda z: np.piecewise(z, [z<layer_depth_two, z>= layer_depth_two], [lambda z: func1(z), lambda z: self.raw[2].sspf(z)])
            else:
                raise ValueError("too many layers")
        else:
            self.ssp_vals = self.raw[0].alphaR
            self.sspf = self.raw[0].sspf

class HS:
    def __init__(self, alphaR=np.array([]), betaR=np.array([]), rho=np.array([]), alphaI=np.array([]), betaI=np.array([])):
        self.alphaR = np.array(alphaR)
        self.betaR = np.array(betaR)
        self.rho = np.array(rho)
        self.alphaI = np.array(alphaI)
        self.betaI = np.array(betaI)
        
class BotBndry:
    def __init__(self, Opt, Hs, depth=[], betaI=[0], ):
        self.Opt = Opt # 'A' for analytic or 'CVW' for interpolated ssp
        self.hs = Hs

class TopBndry:
    def __init__(self, Opt, depth=[]):
        self.Opt = Opt
        self.cp = None
        self.cs = None
        self.rho = None 

class Bndry:
    def __init__(self, top, bot):
        self.Top = top
        self.Bot = bot

class Box:
    def __init__(self, z, r):
        self.r = r
        self.z = z

class Beam:
    def __init__(self, RunType=None, Type=None,Nbeams=None, Ibeam=None, Nrays=None, alpha=None, deltas=None, box=None, epmult=None, rloop=None, Ibwin=None, Nimage = None):
        self.RunType = RunType
        self.Type = Type
        self.Nbeams =  Nbeams
        self.Ibeam  =  Ibeam 
        self.Nrays  =  Nrays
        self.alpha  =  alpha
        self.deltas =  deltas
        self.box    =  box
        self.epmult =  epmult
        self.rloop  =  rloop
        self.Ibwin = Ibwin
        self.Nimage = None

class cInt:
    def __init__(self, low, high):
        self.Low = low
        self.High = high

class Field:
    def __init__(self, **kwargs):
        self.pos = kwargs['pos']
        self.greens = kwargs['pressure']
        self.freq = kwargs['freq'] # a list of frequencies

    def contour(self):
        shape = np.shape(self.greens)
        if len(shape) > 2:
            raise ValueError("Field object is too high dimensional, try multi_contour")
        pos = self.pos 
        ax = plt.contourf(pos.r.range, -pos.r.depth, np.log10(abs(self.greens.real)))
        return ax

    def multi_contour(self):
        print('not implemented, maybe one day...')

class Modes:
    def __init__(self, **kwargs):
        self.M = kwargs['M']
        self.k = kwargs['modes_k']
        self.z = np.array(kwargs['z'])
        self.phi = kwargs['modes_phi']
        self.top = kwargs['top']
        self.bot = kwargs['bot']
        self.N = kwargs['N'] 
        self.Nfreq = kwargs['Nfreq']
        self.Nmedia = kwargs['Nmedia']
        self.depth = kwargs['depth']
        self.rho = kwargs['rho']
        self.freqvec = kwargs['freqVec']
        self.init_dict = kwargs
        self.num_modes = self.M # easier to remember

    def get_excited_modes(self, sd, threshold):
        '''
        return an array of modes that are excited by a source at sd meters depth
        threshold 
        
        also populate some structures in moded caled excited_phi and excited_k

        '''
        if sd not in self.z:
            raise ValueError("sd not in the depth array, are you sure it's the right depth you're passing?")
        depth_ind = [i for i in range(len(self.z)) if self.z[i] == sd][0]
        vals = self.phi[depth_ind,:]
        const = np.max(abs(vals)) 
        filtered_inds = [i for i in range(len(self.k)) if abs(self.phi[depth_ind, i]) / const > threshold]
        self.excited_phi = self.phi[:, filtered_inds]
        self.excited_k = self.k[filtered_inds]
        return self.excited_phi, self.excited_k

    def get_source_depth_ind(self, sd):
        """
        sd is an int 
        """
        tol = 1e-2
        sind = [i for i in range(len(self.z)) if abs(self.z[i]-sd) < tol]
        if len(sind) == 0 :
            raise ValueError("sd not in the depth array, are you sure it's the right depth you're passing?")
        else:
            self.sind = sind[0]
        return  self.sind

    def remove_source_pos(self, sd):
        """
        Take the source at sd from the mode matrix
        Initiate a new attribute to hold the source
        modal value
        """
        sind = self.get_source_depth_ind(sd)
        new_pos_len = len(self.z) - 1
        new_phi = np.zeros((new_pos_len, self.num_modes), dtype=self.phi.dtype)
        new_phi[:sind, :] = self.phi[:sind, :]
        new_phi[sind:,:] = self.phi[sind+1:,:]
        self.phi = new_phi
        new_z = np.zeros((new_pos_len), dtype=self.z.dtype)
        new_z[:sind] = self.z[:sind]
        new_z[sind:] = self.z[sind+1:]
        self.z = new_z
        self.source_strength = self.phi[sind,:]
        return 
            
    def get_source_strength(self, sd):
        """
        Get the value of each mode at the source depth sd (meters)
        Initialize new attribute for the source strength
        """
        sind = self.get_source_depth_ind(sd)
        vals = self.phi[sind,:]
        self.source_strength = vals
        return  vals

    def get_receiver_modes(self, zr):
        """
        zr is array like 
        """
        tol = 1e-3
        r_inds = [i for i in range(len(self.z)) if np.min(abs(self.z[i]-zr)) < tol]
        receiver_modes = self.phi[r_inds, :]
        self.receiver_modes = receiver_modes
        return receiver_modes

    def get_source_excitation(self, zs):
        """
        For case where there is a track, there may be multiple repeats in zs
        """
        tol = 1e-3
        r_inds = [np.argmin(abs(zs[i] - self.z)) for i in range(len(zs))]
        strength_modes = self.phi[r_inds, :]
        self.strength_modes = strength_modes
        return strength_modes
        

    def plot(self):
        figs = []
        if self.M > 5:
            for i in range(self.M):
                fig = plt.figure(i)
                plt.plot(self.phi[:,i], -self.z)
                figs.append(fig)
        return figs

    def __repr__(self):
        return 'Modes object with ' + str(self.M) + ' distinct modes'
      
class Arrival:
    def __init__(self, info_list):
        self.amp = info_list[0]
        self.delay = info_list[1] 
        self.src_ang = info_list[2] 
        self.rec_ang = info_list[3] 
        self.num_top_bnc = info_list[4] 
        self.num_bot_bnc = info_list[5] 
        
class Arrivals:
    def __init__(self, arrival_list):
        self.arrivals = arrival_list
    
    def plot_cir(self, vals=None,ax=None):
        """
        Plot the channel impulse response
        Vals provides the option to pass in a set of arrivals to use fo calibrating the y axis
        If vals=None, I use the arrivals in the object to calibrate the y-axis
        I don't create a figure
        I return ymin, ymax, vals so that the values from this arrival set can be used in a more global
        scope to set the y-axis limits (useful for creating videos comprised of snapshots of a bunch
        of CIR's). For example, I want to plot a bunch of Arrivals on the same plot, or combine a bunch of
        CIRs into a video. Then I need to have the same axis limits for all the plots. 
        I can use default vals (None) for the first plot, then use the ymin, ymax, vals return values
        in a larger scope to standardize the axes. Then for the second, third, ... plots, I pass in vals
        to make sure they all have the same time axis.
        Output - 
            ymin - float
            ymax -float 
                plus/minus 1.5 times max amplitude value
            vals - list
                vals[0] is the times
                vals[1] is the amplitude vals
        """
        arrival_list = self.arrivals
        amps = np.array([abs(x.amp) for x in arrival_list])
        phase = np.array([np.angle(x.amp) for x in arrival_list])
        times = np.array([x.delay for x in arrival_list])
        """
        Create the time axis
        """
        if type(vals) != type(None):
            t = np.linspacea(.9*np.min(vals[0]), 1.1*np.max(vals[0]), 100)
        else:
            t = np.linspace(.9*np.min(times), 1.1*np.max(times), 100)
        zeros = np.zeros(t.size)
        if type(ax) == type(None):
            fig, ax = plt.subplots(1,1) 
        ax.scatter(t.real, zeros, s=6)
        ax.stem(times.real, amps.real, markerfmt=' ', basefmt=' ',use_line_collection=True)
        ax.scatter(times.real, amps,s=16,c='k')
        scale = 1.5*np.max(abs(amps)) # set yaxis scale
        ymin = - scale
        ymax = scale
        vals = [times.real, amps]
        return ymin, ymax, vals

    def plot_ellipse(self, vals=None, ref_amp =None, src_ang=False, ax=None):
        """
        Produce the travel time ellipse with colorbar for amplitude
        Input - 
        self - contains all the arrival info
        vals - optional
            allows me to use a reference arrival set to calibrate the plot axes
            vals[0] = times
            vals[0] = amps
        """
        arrival_list = self.arrivals
        amps = np.array([abs(x.amp) for x in arrival_list])
        times = np.array([x.delay.real for x in arrival_list])
        rec_angles = np.array([x.rec_ang for x in arrival_list])
        src_angles = np.array([x.rec_ang for x in arrival_list])
        """
        Create the time axis
        """
        if type(vals) != type(None):
            tmin, tmax = np.min(vals[0]), np.max(vals[0])
        else:
            tmin, tmax = np.min(times), np.max(times)
        if type(ref_amp) == type(None):
            amps = amps / np.max(abs(amps))
        else:
            amps /= abs(ref_amp)
        cvals = 10*np.log10(abs(amps))
        #cvals = np.linspace(-5, 0, len(amps))
        """ Make a nice cmap """ 
        cmap = cm.get_cmap('Spectral')
        """ Configure the x axis """

        if type(ax) == type(None):
            fig, ax = plt.subplots() 
        ax.scatter([tmin, tmax], [0,0], s=0)
        if src_ang == False:
            angles = rec_angles
        else:
            angles = src_angles
        max_db_down = 10
        cvals_normed = abs(cvals) / max_db_down
        strong_inds = cvals > -max_db_down
        ax.scatter(times[strong_inds], angles[strong_inds], s=25, c=cmap(cvals_normed[strong_inds]))
        vals = [times, amps]
        ymin, ymax = -1.5*np.max(abs(angles)), 1.5*np.max(abs(angles))
        return ymin, ymax, vals

class KernInput:
    def __init__(self, Field_r, Field_s, env):
        self.gr = Field_r.greens_mat
        self.num_rcvrs = len(self.gr)
        self.gs = np.flip(Field_s.greens_mat, axis=1) # move source off axis
        self.env = env
        self.Pos = self.Field_r.Pos

class Eigenray:
    def __init__(self, launch_ang, num_top_bnc, num_bot_bnc, xy_arr):
        """
        launch_ang
            launch angl in degrees
        num_top_bnc
            integer
        num_bot_bnc
            integer
        xy_arr - np array
            2 by num steps, first is x position, second is y
            (so plt.plot(xy_arr[0,:], xy_arr[1,:]) will plot the ray)
        """
        self.src_ang = launch_ang
        self.num_top_bnc = num_top_bnc
        self.num_bot_bnc = num_bot_bnc
        self.xy = xy_arr

def plot_ssp(ssp):
    """
    Plot SSP object
    Make a plot for comprresive speed, shear speed, density, compressional atten.
    and shear atten.
    """
    N = ssp.NMedia
    fig, axes = plt.subplots(2,3, sharey=True)
    layer_ssps = ssp.raw # list of raw ssp profile
    for layer_ind in range(N):
        layer_ssp_raw = layer_ssps[layer_ind]
        _ = layer_ssp_raw # just to abbreviate the syntax
        y_list = [_.alphaR, _.betaR, _.rho, _.alphaI, _.betaI]
        z = layer_ssp_raw.z
        for k in range(5):
            ax = axes.ravel()[k]
            ax.plot(y_list[k], z)
    labels = ['c', 'cs', '$\\rho$', '$\\alpha$', '$\\beta$']
    for k in range(5):
        ax = axes.ravel()[k]
        ax.invert_yaxis()
        ax.set_xlabel(labels[k])
        ax.set_ylabel('Depth (m)')

    fig.subplots_adjust(wspace=.2, hspace=.25)
    return fig, axes
        

