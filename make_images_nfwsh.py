#############################################################
# This file produces images using lenstronomy and paltas COSMOS
# sources with NFW subhalos.
#############################################################


import numpy as np
import time, sys, os
import argparse

from scipy.stats import truncnorm

from paltas.Sources.cosmos import COSMOSExcludeCatalog, COSMOSIncludeCatalog
import pandas as pd

from utils import *

parser = argparse.ArgumentParser('Generate Strong Lensing Images')
parser.add_argument("--n_start", type=int, default=0, help='Start index of images.')
parser.add_argument("--n_image", type=int, default=100000, help='Number of images.')
parser.add_argument("--minlogmass", type=float, default=8, help='Lowerbound on subhalo mass function.')
parser.add_argument("--maxlogmass", type=float, default=10, help='Upperbound on subhalo mass function.')
parser.add_argument("--beta", type=float, default=-1.9, help='Slope of subhalo mass function.')
parser.add_argument("--minnsub", type=int, default=30, help='Lowerbound on number of subhalos.')
parser.add_argument("--maxnsub", type=int, default=50, help='Upperbound on number of subhalos.')
parser.add_argument("--deltapix", type=float, default=0.08, help='Pixel resolution.')
parser.add_argument("--numpix", type=int, default=100, help='Number of pixels per side of image.')
parser.add_argument("--pixscale", type=float, default=0.5, help='pixscale*pix_max in an image determines the pixels to put subhalos.')
parser.add_argument("--c_factor", type=float, default=1, help='Factor to be multiplied to concentrations of CDM mass-conc relation.')
parser.add_argument("--z_lens", type=float, default=0.2, help='Lens redshift') 
parser.add_argument("--z_source", type=float, default=0.6, help='Source redshift') 
parser.add_argument('--subhalo_type', type=str, default='NFW')
parser.add_argument('--ml_type', type=str, default='SIE')
parser.add_argument('--resume', action='store_true', help='Whether prudction is resumed from halfway; if true, load_dir needs to be given.')
parser.add_argument('--data_type', default=None, help='Options include val, train, test.')
parser.add_argument('--path', type=str, default='/n/holyscratch01/dvorkin_lab/gzhang/Storage/SL_images/', help='Path to save data.')
parser.add_argument('--load_dir', type=str, default=None, default=None, help='Directory to load model from.')
parser.add_argument('--noise', action='store_true', help='Whether to add noise to images.')
parser.add_argument('--nms', action='store_true', help='Whether to add negative mass sheet to images.') 
parser.add_argument('--shear', type=float, default=0, help='Bounds on adding shear.')

args = parser.parse_args()

PATH = args.path

# specify keys corresponding to profiles in lenstronomy 
if (args.ml_type == 'SIE'): 
    keys_ml = ['theta_E', 'e1', 'e2', 'center_x', 'center_y']
elif (args.ml_type == 'EPL'): 
    keys_ml = ['theta_E', 'gamma', 'e1', 'e2', 'center_x', 'center_y']
    
n_total = args.n_image 

if (args.noise): print('Add noise', flush=True) 

# make directories and initialize gamma 
if (args.resume): 
    print('Resuming', flush=True) 
    PATH_save = args.load_dir
    PATH_saveim = PATH_save + 'images/'
    PATH_lensargs = PATH_save + 'lensargs/'
    PATH_modelargs = PATH_save + 'modelargs/'
    PATH_sourceargs = PATH_save + 'sourceargs/'
    
else:
    if (args.nms and args.noise): 
        PATH_save = PATH + 'deltapix{}_numpix{}_{}sh_{}ml_logm{}to{}_beta{}_nsub{}to{}_{}maxpix_zl{}zs{}_nms_noise10orb_varycosmos'.format(args.deltapix, args.numpix, args.subhalo_type, args.ml_type, args.minlogmass, args.maxlogmass, args.beta, args.minnsub, args.maxnsub, args.pixscale, args.z_lens, args.z_source)
    elif (args.nms):
        PATH_save = PATH + 'deltapix{}_numpix{}_{}sh_{}ml_logm{}to{}_beta{}_nsub{}to{}_{}maxpix_zl{}zs{}_nms_varycosmos'.format(args.deltapix, args.numpix, args.subhalo_type, args.ml_type, args.minlogmass, args.maxlogmass, args.beta, args.minnsub, args.maxnsub, args.pixscale, args.z_lens, args.z_source)
    elif (args.noise):
        PATH_save = PATH + 'deltapix{}_numpix{}_{}sh_{}ml_logm{}to{}_beta{}_nsub{}to{}_{}maxpix_zl{}zs{}_noise10orb_varycosmos'.format(args.deltapix, args.numpix, args.subhalo_type, args.ml_type, args.minlogmass, args.maxlogmass, args.beta, args.minnsub, args.maxnsub, args.pixscale, args.z_lens, args.z_source)
        
    if (args.shear):
        PATH_save = PATH_save + '_shear{}/'.format(args.shear)
    else: 
        PATH_save = PATH_save + '/'
        
    PATH_saveim = PATH_save + 'images/'
    PATH_lensargs = PATH_save + 'lensargs/'
    PATH_modelargs = PATH_save + 'modelargs/'
    PATH_sourceargs = PATH_save + 'sourceargs/'
    os.makedirs(PATH_save, exist_ok=True)
    os.makedirs(PATH_saveim, exist_ok=True)
    os.makedirs(PATH_lensargs, exist_ok=True)
    os.makedirs(PATH_modelargs, exist_ok=True)
    os.makedirs(PATH_sourceargs, exist_ok=True)

    
print(PATH_save, flush=True) 


# make main lens args 
thetas_ml = np.random.uniform(2.7, 3, size=n_total)
e1s = np.random.uniform(-0.2, 0.2, size=n_total)
e2s = np.random.uniform(-0.2, 0.2, size=n_total)
center_xs = np.random.uniform(-0.2, 0.2, size=n_total)
center_ys = np.random.uniform(-0.2, 0.2, size=n_total)

if (args.ml_type == 'SIE'): 
    vals = np.array([thetas_ml, e1s, e2s, center_xs, center_ys]).T
elif (args.ml_type == 'EPL'): 
    gammas_ml = np.random.uniform(1.8, 2.2, size=n_total)
    vals = np.array([thetas_ml, gammas_ml, e1s, e2s, center_xs, center_ys]).T


# args for fixed source 
cosmos_folder = '/n/holyscratch01/dvorkin_lab/gzhang/Storage/COSMOS_23.5_training_sample'
output_ab_zeropoint = 25.127 #25.9463 # (25.127 was also used in PALTAS code. The HST zeropoint is 25.96.)
z_lens, z_source = args.z_lens, args.z_source 

# check which type of dataset we are making 
if (args.data_type == 'val'): 
    print('Making validation set', flush=True) 
    source_parameters = {
        'z_source':z_source,
        'cosmos_folder':cosmos_folder,
        'max_z':1.0,'minimum_size_in_pixels':64,'faintest_apparent_mag':20,
        'smoothing_sigma':0.00,'random_rotation':True,
        'output_ab_zeropoint':output_ab_zeropoint,
        'min_flux_radius':10.0,
        'center_x':0,
        'center_y':0, 
        'source_inclusion_list': pd.read_csv('/n/holyscratch01/dvorkin_lab/gzhang/Storage/val_galaxies.csv',
                        names=['catalog_i'])['catalog_i'].to_numpy()[:70]}

    cc = COSMOSIncludeCatalog('planck18', source_parameters)
elif (args.data_type == 'train'): 
    print('Making training set', flush=True) 
    source_parameters = {
        'z_source':z_source,
        'cosmos_folder':cosmos_folder,
        'max_z':1.0,'minimum_size_in_pixels':64,'faintest_apparent_mag':20,
        'smoothing_sigma':0.00,'random_rotation':True,
        'output_ab_zeropoint':output_ab_zeropoint,
        'min_flux_radius':10.0,
        'center_x':0,
        'center_y':0, 
        'source_exclusion_list':np.append(
            pd.read_csv('/n/holyscratch01/dvorkin_lab/gzhang/Storage/bad_galaxies.csv',
                        names=['catalog_i'])['catalog_i'].to_numpy(), 
            pd.read_csv('/n/holyscratch01/dvorkin_lab/gzhang/Storage/val_galaxies.csv',
                        names=['catalog_i'])['catalog_i'].to_numpy())}

    cc = COSMOSExcludeCatalog('planck18', source_parameters)
elif (args.data_type == 'test'): 
    print('Making test set', flush=True) 
    source_parameters = {
        'z_source':z_source,
        'cosmos_folder':cosmos_folder,
        'max_z':1.0,'minimum_size_in_pixels':64,'faintest_apparent_mag':20,
        'smoothing_sigma':0.00,'random_rotation':True,
        'output_ab_zeropoint':output_ab_zeropoint,
        'min_flux_radius':10.0,
        'center_x':0,
        'center_y':0, 
        'source_inclusion_list': pd.read_csv('/n/holyscratch01/dvorkin_lab/gzhang/Storage/val_galaxies.csv',
                        names=['catalog_i'])['catalog_i'].to_numpy()[70:]}

    cc = COSMOSIncludeCatalog('planck18', source_parameters)
    

for i in range(n_total): 
    # vary source location 
    source_parameters['center_x'] = np.random.uniform(-0.1, 0.1)
    source_parameters['center_y'] = np.random.uniform(-0.1, 0.1)
    
    if (args.data_type == 'train'): 
        cc = COSMOSExcludeCatalog('planck18', source_parameters)
    else: 
        cc = COSMOSIncludeCatalog('planck18', source_parameters)
        
    source_model_list, kwargs_source = cc.draw_source()
    
    # determine some parameters of subhalos first 
    nsub = np.random.randint(args.minnsub, args.maxnsub)
    
    # make image 
    dic = make_image(z_lens=z_lens, z_source=z_source, numPix=args.numpix, deltapix=args.deltapix, minmass=10**args.minlogmass, maxmass=10**args.maxlogmass, nsub=nsub, pix_scale=args.pixscale, subhalo_type=args.subhalo_type, lensargs=dict(zip(keys_ml, vals[i])), sourceargs=(source_model_list, kwargs_source), noise=args.noise, concentration_factor=args.c_factor, nms=args.nms, beta=args.beta, shear=args.shear, main_lens_type=args.ml_type)
    
    np.save(PATH_saveim + 'SLimage_{}'.format(args.n_start + i + 1), dic['Image'])
    np.save(PATH_lensargs + 'lensarg_{}'.format(args.n_start + i + 1), dic['kwargs_lens'])
    np.save(PATH_modelargs + 'modelarg_{}'.format(args.n_start + i + 1), dic['kwargs_model'])
    np.save(PATH_sourceargs + 'sourcearg_{}'.format(args.n_start + i + 1), dic['kwargs_source'])
    
    if ((i+1) % 5000 == 0): 
        print('Image {} saved'.format(args.n_start + i + 1), flush=True)