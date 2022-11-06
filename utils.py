import numpy as np

import lenstronomy.Util.util as util
import lenstronomy.Util.constants as const
from astropy.cosmology import default_cosmology
from lenstronomy.Cosmo.background import Background 
from lenstronomy.LensModel.Profiles.spp import SPP
from lenstronomy.LensModel.Profiles.sie import SIE
from lenstronomy.LensModel.Profiles.coreBurkert import CoreBurkert
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
import lenstronomy.Util.param_util as param_util
from lenstronomy.LensModel.lens_model import LensModel

from lenstronomy.SimulationAPI.ObservationConfig.HST import HST
from lenstronomy.SimulationAPI.sim_api import SimAPI


def rho_crit(cosmo, z):
    """
    critical density of cosmo at redshift z 
    :return: value in M_sol/Mpc^3
    """
    h = cosmo.H(z).value / 100.
    return 3 * h ** 2 / (8 * np.pi * const.G) * 10 ** 10 * const.Mpc / const.M_sun


def epl_m2thetae(m200, gamma, rho_c, s_crit): 
    '''
    Args: 
        m200 (float, np.array): m200's 
        gamma (float, np.array): density slopes 
        rho_c (float): critical density of universe 
        s_crit (float): sigma critical 
    
    Returns: 
        theta_E in arcsec corresponding to m200
    '''
    r = np.power(3*m200/(4*np.pi)/(200*rho_c), 1/3)
    rho0 = (3-gamma)/(4*np.pi)*m200/s_crit*r**(gamma-3)
    
    return SPP.rho2theta(rho0, gamma)


def get_sigmac(lensCosmo): 
    '''
    Args: 
        LensCosmo object in lenstronomy 
        
    Returns: 
        s_crit (float): sigma critical 
    '''
    # in Mpc 
    da_lens = lensCosmo.dd
    da_source = lensCosmo.ds
    da_ls = lensCosmo.dds

    Si_crit = const.c**2*da_source/(4*np.pi*const.G*da_lens*da_ls*const.Mpc)  # in kg/m^2 
    Si_crit = Si_crit/const.M_sun*const.Mpc**2  # in M_sun/Mpc^2 
    
    return Si_crit



def m_unif(x, m0=10**8, m1=10**11, beta=-1.9):
    '''
    taken from mathematica inverse of probability distribution of 0809.0898v1
    (x*m_max^(1+beta) + (1-x)*m_min^(1+beta))^(1/(1+beta))
    
    Args: 
        x (np.array): values pulled from U(0, 1) 
        m0 (float): min mass 
        m1 (float): max mass 
        beta (float): slope of SHMF 
        
    Returns: 
        array of masses pulled from SHMF 
    '''
    m0_b = np.power(m0, 1+beta)
    m1_b = np.power(m1, 1+beta)
    m = np.power(x * m1_b + (1-x)*m0_b, 1 / (1+beta))
    return m

def mass_to_concentration(m):
    '''
    mass-concentration for nfw from https://academic.oup.com/mnras/article/441/4/3359/1209689 at z=0.5
    
    Args: 
        m (float, np.array): mass 
        
    Returns: 
        concentration(s) corresponding to m 
    '''
    return 10**(0.814 - 0.086*np.log10(m/(10**12/0.7)))



# ******************************************************************************
# Image Generation
# ******************************************************************************


def make_image(z_lens=0.2,
               z_source=0.6,
               numPix=100, 
               deltapix=0.08,
               main_lens_type='SIE',
               subhalo_type='EPL', 
               concentration_factor=1,  # concentration factor if subhalo_type=NFW, fix c=20 if set to 0
               max_sources=5,  # max number of sersics if sourceargs are not given 
               nsub=None,
               msubs=None,
               minmass=10**7,
               maxmass=10**10,
               beta=-1.9,
               gamma=None,    # optional pre-defined gammas for subhalos if nsub is given 
               pix_scale=0.5,   # threshold of brightness to determine positions to put subhalos 
               noise=True,    # whether to add 10 orbits of noise 
               nms=True,      # whether to add negative mass sheet 
               shear=0,       # external shear will be U(-shear, shear)
               sourceargs=None,   # includes source_model_list, kwargs_source matching lenstronomy 
               lensargs=None):  # kwargs matching lenstronomy; lenargs need to match main_lens_type 
    
    cosmo = default_cosmology.get()
    side_length = numPix * deltapix   # side length in arcsec 
    # we set up a grid in coordinates and evaluate basic lensing quantities on it
    x_grid, y_grid = util.make_grid(numPix=numPix, deltapix=deltapix)

    lensCosmo = LensCosmo(z_lens=z_lens, z_source=z_source, cosmo=cosmo)
    mpc_per_arcsec_lens = lensCosmo.arcsec2phys_lens(1)

    # in M_sun and arcsec in lens plane 
    Si_crit = get_sigmac(lensCosmo)*mpc_per_arcsec_lens**2
    rho_c = rho_crit(cosmo, z_lens)*mpc_per_arcsec_lens**3

    if (subhalo_type == 'EPL'): 
        keys = ['theta_E', 'gamma', 'e1', 'e2', 'center_x', 'center_y']
    elif (subhalo_type == 'NFW'): 
        keys = ['Rs', 'alpha_Rs', 'center_x', 'center_y'] 
    elif (subhalo_type == 'coreBURKERT'): 
        keys = ['Rs', 'alpha_Rs', 'r_core', 'center_x', 'center_y']
    
    kwargs_source = []
    source_model_list = []
    lens_model_list = []
    kwargs_lens_list = []
    
    # make sersic source if sourceargs aren't given 
    if (sourceargs is None):
        N_Sources = np.random.randint(1 if max_sources==1 else 3, max_sources+1)

        for i in range(N_Sources):

            # the first source 
            if i == 0:
                phi_G, q = np.random.uniform(-np.pi, np.pi), np.random.uniform(1, 3)
                e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
                source_x, source_y = np.random.multivariate_normal(mean=[0, 0], cov=[[0.001, 0], [0, 0.001]])
                amp_power = 2.5 + np.random.random()**8  #np.random.uniform(2.5, 3.5)
                max_amp = np.power(10, amp_power)
                max_R = np.random.uniform(0.1,0.8)
                R = max_R
                amp = max_amp
            else:
                phi_G1, q1 = np.random.uniform(-np.pi, np.pi), np.random.uniform(1, 5)
                e1, e2 = param_util.phi_q2_ellipticity(phi_G1, q1)
                source_x, source_y = np.random.multivariate_normal(mean=[0, 0], cov=[[0.001, 0], [0, max_R**2]])
                source_x, source_y = np.cos(phi_G) * source_x + np.sin(phi_G) * source_y, np.cos(phi_G) * source_y - np.sin(phi_G) * source_x
                amp = np.random.uniform(0.1, max_amp)
                R = np.random.uniform(0.1, max_R*0.75)

                kwargs_sersic_source = {'amp': amp * R,
                                'R_sersic': R,
                                'n_sersic': np.random.uniform(0.9, 1.5),
                                'e1': e1,
                                'e2': e2,
                                'center_x': source_x,
                                'center_y': source_y
                               }

                source_model_list.append('SERSIC_ELLIPSE')
                kwargs_source.append(kwargs_sersic_source)
    
    else: 
        source_model_list, kwargs_source = sourceargs
    
    if (lensargs is None): 
        if main_lens_type == 'SIE':
            xlens, ylens = np.random.uniform(-0.25, 0.25, size=2)
            theta_E = np.random.uniform(2.8, 3.2)
            kwargs_lens_main = {'theta_E': theta_E, 
                                'e1': np.random.uniform(-0.2, 0.2),
                                'e2': np.random.uniform(-0.2, 0.2),
                                'center_x': xlens,
                                'center_y': ylens}
        elif main_lens_type == 'EPL':
            xlens, ylens = np.random.uniform(-0.25, 0.25, size=2)
            theta_E = np.random.uniform(2.8, 3.2)
            kwargs_lens_main = {'theta_E': theta_E, 
                                'gamma': np.random.uniform(1.8, 2.2),
                                'e1': np.random.uniform(-0.2, 0.2),
                                'e2': np.random.uniform(-0.2, 0.2),
                                'center_x': xlens,
                                'center_y': ylens}
    else: 
        kwargs_lens_main = lensargs
        
    lens_model_list.append(main_lens_type)
    kwargs_lens_list.append(kwargs_lens_main)
    
    # add shear 
    if shear:
        kwargs_shear = {'gamma1': np.random.uniform(-shear, shear),
                        'gamma2': np.random.uniform(-shear, shear)}
        lens_model_list.append('SHEAR')
        kwargs_lens_list.append(kwargs_shear)
        
    kwargs_model_mainlens = {'lens_model_list': lens_model_list,  # list of lens models to be used
                    'source_light_model_list': source_model_list,  # list of extended source models to be used
                    'z_lens': z_lens,
                    'z_source': z_source,
                    'cosmo': cosmo
                    }
    
    # making this image to put the subhalos inside the ring 
    hst_ml = HST(band='WFC3_F160W', psf_type='PIXEL')
    Hub_ml = SimAPI(numpix=numPix,
                 kwargs_single_band=hst_ml.kwargs_single_band(),
                 kwargs_model=kwargs_model_mainlens
                )

    hb_ml = Hub_ml.image_model_class(kwargs_numerics = {'point_source_supersampling_factor': 1})
    im_ml = hb_ml.image(kwargs_lens=kwargs_lens_list, kwargs_source=kwargs_source)
    
    # determine valid pixels for subhalo positioning 
    pix_max = np.max(im_ml)
    x_options, y_options = x_grid[im_ml.flatten() > pix_scale*pix_max], y_grid[im_ml.flatten() > pix_scale*pix_max]
    inds_choice = np.random.choice(range(len(x_options)), nsub)
    
    x = x_options[inds_choice] + np.random.uniform(-deltapix/2, deltapix/2, nsub)
    y = y_options[inds_choice] + np.random.uniform(-deltapix/2, deltapix/2, nsub)
    
    if (subhalo_type == 'EPL'): 
        # get other subhalo parameters 
        if (msubs is None): 
            msubs = m_unif(np.random.random(nsub), minmass, maxmass, beta=beta)
                      
        thetaes = epl_m2thetae(msubs, gamma, rho_c, Si_crit)
        e1s, e2s = np.random.uniform(-0.2, 0.2, (2, nsub)) 

        # subhalo parameter list
        vals = np.array([thetaes, gamma, e1s, e2s, x, y]).T
        
    elif (subhalo_type == 'NFW'): 
        if (msubs is None): 
            mass_nfw = m_unif(np.random.random(nsub), minmass, maxmass, beta=beta)
        else: 
            mass_nfw = msubs
        
        if (concentration_factor == 0):
            cs = 20*np.ones(len(mass_nfw)) 
        else: 
            cs = mass_to_concentration(mass_nfw)*concentration_factor
            
        Rs_angle, alpha_Rs = lensCosmo.nfw_physical2angle(M=mass_nfw, c=cs)
        
        # subhalo parameter list
        vals = np.array([Rs_angle, alpha_Rs, x, y]).T
        
    elif (subhalo_type == 'coreBURKERT'): 
        if (msubs is None): 
            mass_burk = m_unif(np.random.random(nsub), minmass, maxmass, beta=beta)
        else: 
            mass_burk = msubs
            
        Rs_angle,_ = lensCosmo.nfw_physical2angle(M=mass_burk, c=mass_to_concentration(mass_burk))
        alpha_Rs = coreBurkert_mtoalpha(mass_burk, Rs_angle, Rs_angle, Si_crit, rho_c)
        
        vals = np.array([Rs_angle, alpha_Rs, Rs_angle, x, y]).T

    kwargs_subhalo_lens_list = []
    for val in vals: 
        kwargs_subhalo_lens_list.append(dict(zip(keys, val)))
            

    lensModel = LensModel(lens_model_list + [subhalo_type]*nsub)
    kappa = lensModel.kappa(x_grid, y_grid, kwargs_lens_list + kwargs_subhalo_lens_list)

    subhalo_lens_list = [subhalo_type]*nsub 
    
    if (nms):
        # negative mass sheet 
        lensModel_sh = LensModel([subhalo_type]*nsub)
        kappa_subhalos = lensModel_sh.kappa(x_grid, y_grid, kwargs_subhalo_lens_list)
        
        mass_sheet = np.mean(kappa_subhalos)
        subhalo_lens_list = subhalo_lens_list + ['CONVERGENCE']
        
        kwargs_subhalo_lens_list = kwargs_subhalo_lens_list + [{'kappa': -mass_sheet}]  
        
    
    kwargs_model = {'lens_model_list': lens_model_list + subhalo_lens_list,  # list of lens models to be used
                    'source_light_model_list': source_model_list,  # list of extended source models to be used
                    'z_lens': z_lens,
                    'z_source': z_source,
                    'cosmo': cosmo
                    }
    
    # make image with noise 
    hst = HST(band='WFC3_F160W', psf_type='PIXEL')
    norbits = hst.kwargs_single_band()
    norbits['pixel_scale'] = deltapix
    norbits['seeing'] = deltapix
    
    if (noise): 
        norbits['exposure_time'] = 5400 * 10  # each orbit is 5400 secs 
    
    Hub = SimAPI(numpix=numPix,
                 kwargs_single_band=norbits,
                 kwargs_model=kwargs_model
                )

    hb_im = Hub.image_model_class(kwargs_numerics = {'point_source_supersampling_factor': 1})

    im = hb_im.image(kwargs_lens=kwargs_lens_list + kwargs_subhalo_lens_list, kwargs_source=kwargs_source)
    
    # add noise 
    if (noise): 
        hubnoise = Hub.noise_for_model(im)
        im = im + hubnoise
    
    out_dict = {'kwargs_model': kwargs_model,
                'kwargs_source': kwargs_source,
                'kwargs_lens': kwargs_lens_list + kwargs_subhalo_lens_list, 
                'Image': im, 
                'Image_ml': im_ml
               }
    
    return out_dict 