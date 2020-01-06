import numpy as np
import xml.dom.minidom
import subprocess
from ast import literal_eval
from skimage import io
from skimage.feature import register_translation
from skimage import transform as tf
from skimage.exposure import is_low_contrast,rescale_intensity
import os
import re
from natsort import natsorted
from skimage.filters import gaussian

import pylab as pl

from csbdeep.utils import Path, download_and_extract_zip_file, plot_some
from csbdeep.utils.tf import limit_gpu_memory
from csbdeep.io import save_tiff_imagej_compatible
from csbdeep.models import CARE
from time import sleep

limit_gpu_memory(fraction=1/4)

os.chdir("C:\\AutoTracker")

attributes = ['spatial-calibration-x', 'spatial-calibration-y',
             'stage-position-x','stage-position-y'
             ,'z-position','stage-label','_IllumSetting_']

def get_metadata(fin,frame,attributes):
    a = subprocess.Popen(["tiffinfo", "-" + str(frame), fin], stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    output, err = a.communicate()
    tiff_string = output.decode('unicode_escape')#.split('\\n')#readlines()
    #tiff_string = "".join(tiff_string)
    xml_str = tiff_string[(tiff_string.find("<MetaData>")):(10 + tiff_string.rfind("MetaData>"))]
    dom = xml.dom.minidom.parseString(xml_str)
    props = dom.getElementsByTagName("prop")
    attr_dict={}
    for p in props:
        attr = p.getAttribute("id")
        if attr in attributes:
            try:
                attr_dict[attr]=literal_eval(p.getAttribute('value'))
            except:
                attr_dict[attr]=p.getAttribute('value')
    return attr_dict

def get_scales(filename):
    attr_dict = get_metadata(filename,0,attributes)
    #Check if the images are taken with the 488 (green) laser
    #if not, print a warning
    if '488' not in attr_dict['_IllumSetting_']:
        print('Warning! Loaded images are not in the green channel.')
        print('Illumination setting is:', attr_dict['_IllumSetting_'])
    dx = attr_dict['spatial-calibration-x']
    dy = attr_dict['spatial-calibration-y']
    x0 = attr_dict['stage-position-x']
    y0 = attr_dict['stage-position-y']
    z_beg = attr_dict['z-position']
    label = attr_dict['stage-label']
    attr_dict = get_metadata(filename,sz-1,['z-position'])
    z_end = attr_dict['z-position']
    attr_dict = get_metadata(filename,1,['z-position'])
    z_step = attr_dict['z-position']
    dz = np.abs(z_step-z_beg)
    z0 = (z_end+z_beg)/2
    return dx,dy,dz,x0,y0,z0,label

def get_center(img):
    if is_low_contrast(img,fraction_threshold=1e-4,upper_percentile=99):
        print('Low contrast image!')
        return sx/2,sy/2,sz/2
    else:
        # Blur the image and find the center of mass
        img_y_max_proj = np.max(img,axis=1)#+np.max(img,axis=2)
        img_y_max_proj = rescale_intensity(img_y_max_proj, in_range='image', out_range=(0,1.))
        img_y_max_proj = gaussian(img_y_max_proj,sigma=4.)
        img_y_max_proj = np.max(img_y_max_proj,axis=1)#+np.max(img,axis=2)
        cmz = np.argmax(img_y_max_proj)

        img_z_max_proj = np.max(img,axis=0) #np.copy(img[cmz,:,:])
        img_z_max_proj = rescale_intensity(img_z_max_proj, in_range='image', out_range=(0,1.))

        mxz=np.max(img_z_max_proj)
        img_z_max_proj = gaussian(img_z_max_proj,sigma=50.)
        #print 'Maximum: ', np.unravel_index(img_z_max_proj.argmax(), img_z_max_proj.shape)
        cmy,cmx = np.unravel_index(img_z_max_proj.argmax(), img_z_max_proj.shape)

        return int(cmx),int(cmy),int(cmz)

def write_stg_position(f,label,x,y,z):
    f.write('['+label+']\n')
    f.write('Stage.X = {:.2f}\n'.format(x))
    f.write('Stage.Y = {:.2f}\n'.format(y))
    f.write('Stage.Z = {:.2f}\n'.format(z))
    f.write('\n')

def write_position_log(f,x0,y0,z0,dlx,dly,dlz):
    if os.path.getsize(f.name)==0:
        f.write('['+label+']\n')
        f.write('\n')
        f.write('{:.2f}\t{:.2f}\t{:.2f}\n'.format(x0,y0,z0))
    f.write('{:.2f}\t{:.2f}\t{:.2f}\n'.format(x0+dlx,y0+dly,z0+dlz))

def load_position_log(log_fname):
    try:
        xyz_array = np.loadtxt(log_fname,skiprows=3,delimiter='\t')
        print(xyz_array.shape)
    except:
        print('Unable to open ',log_fname)
        xyz_array = np.zeros((0,3))
    if xyz_array.shape==(3,):
        xyz_array = np.zeros((0,3))
    return xyz_array

model = CARE(config=None, name='bactn-gfp', basedir='CARE_Models')
axes = 'ZYX'

# fig, ax = pl.subplots()
# pl.ion()
# pl.show()
print('Ready to run')
while 1!=0:
    log_list=[f for f in os.listdir('.') if ('.LOG' in f)]
    #print(log_list)
    for logfile in log_list:
        try:
            dirlog=open(logfile,'r')
        except:
            break
        success = True
        while success==True:
            try:
                path=dirlog.readline().split('"')[1]
                basename=dirlog.readline().split('"')[1]
                stg_pos=dirlog.readline().strip('\n')
                dirlog.close()
                success = True
            except:
                success = False

        INI_out= 'AutoTracker_Stg'+stg_pos+'.INI'

        try:
            os.remove(INI_out)
        except:
            pass

        log_out = os.path.join(path,'AutoTrackerLog', 'AutoTracker_Stg'+stg_pos+'.log')
        rec_path = os.path.join(path,'CARE_Reconstructed')


        image_list=[f for f in os.listdir(path) if (('.tif' in f.lower())
                                                and ('thumb' not in f.lower())
                                                and (basename in f))]
        #img=io.imread(os.path.join(path,image_list[-1]))
        #nz,nx,ny=img.shape

        w_re=re.compile('_w1')
        if w_re.search(image_list[0]) != None:
            multiwavelength=1
            print('Multiple Wavelengths Detected')
        else:
            multiwavelength=0

        stg_file=open(INI_out,'w')
        log_file=open(log_out,'a')

        s_re =re.compile('_s'+stg_pos+'_')
        stage_img_list=natsorted([f for f in image_list if s_re.search(f) !=None])
        #If there are multiple wavelengths find the first one.
        if multiwavelength == 1:
            stage_img_list=natsorted([f for f in stage_img_list if w_re.search(f) !=None])
        n_time_points=len(stage_img_list)
        print('Finding center and shifts for image: ')
        name=os.path.join(path,stage_img_list[-1])
        print(name)
        img=io.imread(name)

        sz,sy,sx=img.shape
        dx,dy,dz,x0,y0,z0,label = get_scales(name)

        recon_img = model.predict(img, axes, n_tiles=(1,8,4))

        xc,yc,zc = get_center(recon_img)
        # xc,yc,zc = get_center(img)

        re_out = os.path.join(rec_path,stage_img_list[-1])
        io.imsave(re_out,recon_img.astype(img.dtype))

        xshift = xc-(sx/2)
        yshift = yc-(sy/2)
        zshift = zc-(sz/2)

        xyz_array = load_position_log(log_out)
        print(xyz_array.shape)
        z_array = xyz_array[:,2]

        if len(z_array > 3):
            zshift = (zshift+np.sum(z_array[-3:]))/4.

        print("Pixel size (um) (x, y, z): {:.4f} {:.4f} {:.4f}".format(dx,dy,dz))
        print("Stage position (x, y, z): {:.2f} {:.2f} {:.2f}".format(x0,y0,z0))
        print("Detected pixel offset (x, y, z): {} {} {}".format(xshift,yshift,zshift))
        print("New stage position (x, y, z): {:.2f} {:.2f} {:.2f}".format(x0+dx*xshift,y0-dy*yshift,z0+dz*zshift))
        #write_stg_position(stg_file,label,x0+dx*xshift,y0-dy*yshift,z0+dz*zshift)
        write_stg_position(stg_file,label,dx*xshift,-dy*yshift,dz*zshift)
        write_position_log(log_file,x0,y0,z0,dx*xshift,-dy*yshift,dz*zshift)
        stg_file.close()
        log_file.close()
        try:
            os.remove('current_directory.bkp')
        except:
            pass
        try:
            os.rename(logfile,'current_directory.bkp')
        except:
            pass
        # ax.imshow(np.max(recon_img,axis=0))
        # pl.draw()
        # pl.pause(0.1)
    sleep(1)
