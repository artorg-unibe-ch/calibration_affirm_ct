"""
************************************
         Benjamin Voumard
  benjamin.voumard@istb.unibe.ch
  Created on %(date)s
************************************
"""
import numpy as np
import matplotlib.pylab as plt
import SimpleITK as sitk
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import mpl_toolkits.mplot3d as m3d
# from glob import glob
from numba import jit
import time
import os
import struct
import re
import socket
from sys import stdout  # argv, exit, stdout
from six import string_types
import cv2
from multiprocessing import Pool
import pysnooper
from skimage import measure
from stl import mesh
from mayavi import mlab
import pandas as pd
# import visvis as vv
import mayavi.core.api
import mayavi.sources.vtk_file_reader
import mayavi.modules.surface
import pyface.api
import trimesh
import vtk
import glob

"""
*************************************************************************
*************************************************************************
"""


def help():
    print("""

Help:
Swissknife.help()

Read images:
# READ: ISQ, AIM, MHD, DICOM
img_sitk, additData, is_it_downscaled, file_type = Swissknife.read(filename)

Swissknife.read(filename, forced)
Swissknife.read_DICOM(data_directory, forced)
Swissknife.read_AIM_header(fileINname)
Swissknife.read_AIM_ints(f)
Swissknife.read_AIM(inFileName, printLog)
Swissknife.read_ISQ(inFileName, BMD, echo, info)

Write Image:
Swissknife.write_MHD(img, filename)
Swissknife.write_DICOM(img, filename)
Swissknife.write_STL(img, filename, factor)
Swissknife.write_Imesh(curVoxelModel, resolution, displacement, folder)

Image rendering:
Swissknife.cut_3d(img, title)
Swissknife.cut_3d_2_images(img1, img2, color, title)
Swissknife.plot_stack(image, title, play_fps, big_scroll, key_func, view)
Swissknife.plot_special(image, title, play_fps, big_scroll, key_func, view)
Swissknife.mayavi_plot_mesh(image, factor, opacity, fig, color, title)
Swissknife.mayavi_plot_vtk(vtkFile)
Swissknife.visvis_plot_mesh(image, factor, opacity, fig, color)
Swissknife.plot_2d(image, title)
Swissknife.mayavi_add_sphere(fig, center, radius, color)
Swissknife.mayavi_add_sphere_list(fig, coord_list, radius, color)
Swissknife.mayavi_arrow_R_t(fig, R, t, arrow_len)
Swissknife.same_length_3D(ax)
Swissknife.plot_voxel(img_sitk, title, factor, same_length)
Swissknife.plt_scatter(coordinates, factor)
Swissknife.plot_arrow_R_t(ax, R, t, arrow_len)
Swissknife.plot_origin(ax, arrow_len)
Swissknife.plot_arrow_3d(ax, v, t, arrow_len, color)
Swissknife.plot_ax()
Swissknife.histo(img)

Image processing:
img_sitk = Swissknife.downscale_kernel(size4func, factor, img_np_cropped, downscaled_np, type_img)
img_sitk = Swissknife.downscale_numba(img, factor, type_img)
img_sitk = Swissknife.downscale_sitk(img_target, resolution_target)
img_sitk = Swissknife.rescale_isotropic(img_sitk, resolution, type_interpolation)
img_sitk = Swissknife.rotate(img, angles, ang_type, center_rotation_x, center_rotation_y, center_rotation_z)
img_sitk = Swissknife.rotate_R(img, R, t, center_rotation_x, center_rotation_y, center_rotation_z, empty_layers)
img_sitk = Swissknife.kernel_rotate_R_personal(img_bigger_coor, new_coord, R, center_rotation_x, img_bigger_np_shape)
img_sitk = Swissknife.rotate_R_personal(img, R, t, center_rotation_x, center_rotation_y, center_rotation_z, empty_layers)
img_np   = Swissknife.center_of_gravity(img, resolution)
img_sitk = Swissknife.add_layers(img, add_layer)
img_sitk = Swissknife.add_layers_nb(img, add_layer)
img_sitk = Swissknife.add_layers_nb_np(img_np, add_layer)
img_sitk = Swissknife.transform_R_t(img_sitk, R, t, center, add_layer, interpo, bbcut)
img_sitk = Swissknife.gaussian(img_sitk, sigma)
img_sitk = Swissknife.segmentation_otsu(img)
img_sitk = Swissknife.segmentation_overview(img, lower_threshold, upper_threshold)
img_sitk = Swissknife.remove_small_object_old(img)
img_sitk = Swissknife.remove_small_object(img)
img_sitk = Swissknife.cv2_loop(inputs)
img_sitk = Swissknife.masking_parallel(seg_np, radius, nb_cpu)
img_sitk = Swissknife.cv2_dilate_loop(inputs)
img_sitk = Swissknife.dilate_cv2(seg_np, radius, nb_cpu)
img_sitk = Swissknife.get_mask_from_seg(seg_np, radius)
img_sitk = img_sitk = Swissknife.smooth_binary(img, sigma, threshold)
img_sitk = Swissknife.bounding_box(img)
img_sitk = Swissknife.bounding_box_start_end(img)
img_sitk = Swissknife.sphere_array_old(dimensions, radius, center, resolution)
img_np   = Swissknife.sphere_array(shape, radius, position)
img_sitk = Swissknife.sphere_itk(shape, radius, position, resolution)
img_sitk = Swissknife.hollow_sphere_array(shape, outer_radius, inner_radius, center, resolution)
img_sitk = Swissknife.hollow_sphere_coordinate(outer_radius, inner_radius, center, resolution)
img_sitk = Swissknife.hollow_sphere_array_old(dimensions, outer_radius, inner_radius, center, resolution)
img_sitk = Swissknife.same_value_in_array(k, k2)
img_sitk = Swissknife.coordinate_binary_image_old(img)
img_sitk = Swissknife.coordinate_binary_image(img)
img_sitk = Swissknife.img_same_dim(img_fixed, img_small)
img_sitk = Swissknife.img_same_properties(img_origin, img, force2notflip)
img_sitk = img_sitk = Swissknife.cylinder_mask(resolution, dimensions, outer_diameter, inner_diameter, length)
img_sitk = Swissknife.img_registration_sitk(img_fixed, img_moving)
img_sitk = Swissknife.registration_sitk_2(img_fixed, img_moving)
img_sitk = img_sitk = Swissknife.mask_image(size, r_factor, img, resolution, center_x, center_y)
img_sitk = Swissknife.bvtv_mask(size, r_factor, img, start_z, end_z, resolution, center_x, center_y)
img_sitk = Swissknife.dilate(img, radius)
img_sitk = Swissknife.erode(img, radius)
img_sitk = Swissknife.dilate_erode(img, radius)
img_sitk = Swissknife.closing(img, radius)
img_sitk = Swissknife.dilate_erode_ndimage(image, iterations)
img_sitk = Swissknife.erode_dilate(img, radius)
img_sitk = Swissknife.center_gravity(image, resolution)
img_sitk = Swissknife.array_3d_to_coordinate(array_3d, value)
img_sitk = Swissknife.kernel_coordinate_to_array_3d(coordinate, array_3d, value)
img_sitk = Swissknife.coordinate_to_array_3d_old_old(coordinate, dim, value)
img_sitk = Swissknife.coordinate_to_array_3d(coordinate, dim, value)
img_sitk = Swissknife.coordinate_to_array_3d_new(coordinates, reso)
img_sitk = Swissknife.write_image_from_coordinates_offset(coordinate, path)
img_sitk = Swissknife.img_16_bit(img)
norm     = Swissknife.normi(vector)
vector   = Swissknife.linear_regression_3d(data)
angle    = Swissknife.angle_between_vectors(v1, v2)
coord    = Swissknife.closet_point_between_vector(v1, v2, start_v1, start_v2)
coord    = Swissknife.find_new_point_perpendicular_to_vector_end_start_and_point_new_point(point, start, end)
text     = Swissknife.img_description(img)
value    = Swissknife.img_min_max(img)
Swissknife.sizeof_fmt(num, suffix)

Mesh:
Swissknife.coordinate_to_vertice(coordinates, reso, downscale_factor)
Swissknife.vertice_to_vtk(vertices, faces, filename, comment)
Swissknife.vertice_to_stl(vertices, faces, normals, filename)
Swissknife.coordinate_to_vtk(coordinates, filename, data, comment, downscale_factor, reso)
Swissknife.mhd_to_vtk_smooth_decimate(path, output_name, write_stl)
Swissknife.icp(currSource_filename, currTarget_filename)
Swissknife.stl_binary_to_ascii(input_path, output_path)
Swissknife.vtk_read_stl(fname)
Swissknife.vtk_decimate(mesh_vtk, target_reduction)
Swissknife.vtk_write_vtk(mesh_vtk, path)
Swissknife.sitk_to_vtk(img, path)

Miscallenous:
Swissknife._datacheck_peakdetect(x_axis, y_axis)
Swissknife.peakdetect(y_axis, x_axis, lookahead, delta, plot)
Swissknife.plot_peaks(y, min_peaks, max_peaks)
    """)


def read(filename, forced='No'):
    folder_func = filename[:-len(filename.split('/')[-1])]
    #    filename_func = glob.glob('.'.join(filename.split('.')[:-1])+'.*')[0]
    filename_func = filename
    if 'AIM' in filename_func.split('.')[-1]:
        if ';' in filename_func.split('.')[-1]:
            os.rename(filename_func, filename_func.split(';')[0])
            filename_func = filename_func.split(';')[0]
        try:
            files_downscaled = [file for file in os.listdir(folder_func) if all(
                [filename.split('/')[-1].split('.')[0] in file, 'downscal' in file, 'mhd' in file])][0]
            img = sitk.ReadImage(folder_func + files_downscaled)
            is_it_downscaled = 'yes'
            resolution = img.GetSpacing()
            print('\n Downscaled file found: \n', folder_func + files_downscaled)
            additData = {}
            additData['ElementSpacing'] = img.GetSpacing()
            additData['DimSize'] = img.GetSize()
            file_type = '.mhd'
        except:
            voxelModel, additData = read_AIM(filename)
            is_it_downscaled = 'no'
            resolution = additData['ElementSpacing']
            file_type = '.AIM'
            img = sitk.GetImageFromArray(voxelModel)
            img.SetSpacing(resolution)
            print('\nOriginal file found: \n', folder_func)
    elif filename.split('.')[-1] == 'mhd':
        img = sitk.ReadImage(filename)
        additData = {}
        additData['ElementSpacing'] = img.GetSpacing()
        additData['DimSize'] = img.GetSize()
        is_it_downscaled = 'no'
        file_type = '.mhd'
    elif filename.split('.')[-1] == 'nii':
        img = sitk.ReadImage(filename)
        additData = {}
        additData['ElementSpacing'] = img.GetSpacing()
        additData['DimSize'] = img.GetSize()
        is_it_downscaled = 'no'
        file_type = '.nii'
    elif filename.split('.')[-1] == 'ISQ':
        try:
            files_downscaled = [file for file in os.listdir(folder_func) if all(
                [filename.split('/')[-1].split('.')[0] in file, 'downscal' in file, 'mhd' in file])][0]
            img = sitk.ReadImage(folder_func + files_downscaled)
            is_it_downscaled = 'yes'
            resolution = img.GetSpacing()
            print('\n Downscaled file found: \n', folder_func + files_downscaled)
            additData = {}
            additData['ElementSpacing'] = img.GetSpacing()
            additData['DimSize'] = img.GetSize()
            file_type = '.mhd'
        except:
            img, additData = read_ISQ(filename)
            is_it_downscaled = 'no'
            file_type = '.ISQ'
    elif os.path.isdir(filename):
        print('    DICOM files identified')
        img = read_DICOM(filename, forced=forced)
        print('    DICOM files successfully read')
        is_it_downscaled = 'no'
        file_type = '.dcm'
        additData = 0
    else:
        print('Error: file format unknown...')
    if img:
        return img, additData, is_it_downscaled, file_type
    else:
        print('image path uncorrect...')


def read_DICOM(data_directory, forced='Yes'):
    if forced.upper() != 'YES':
        if os.path.isfile(data_directory.split(data_directory.split('/')[-1].split('.')[0])[0] +
                          data_directory.split('/')[-1].split('.')[0] + '.mhd'):
            print('    MHD file found instead of DICOM...')
            return sitk.ReadImage(data_directory.split(data_directory.split('/')[-1].split('.')[0])[0] +
                                  data_directory.split('/')[-1].split('.')[0] + '.mhd')
        elif len(glob.glob(data_directory + "/*.dcm")) == 1:
            print('    read 3D DICOM')
            filename = glob.glob(data_directory + "/*.dcm")[0]
            reader = sitk.ImageFileReader()
            reader.SetFileName(os.path.join(data_directory, filename))
            img = reader.Execute()
        else:
            print('    read DICOMs')
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(data_directory)
            reader.SetFileNames(dicom_names)
            img = reader.Execute()
            # dimension = img.GetSize()
            # if dimension[2] > dimension[0] and dimension[2] > dimension[1]:
            #     return img
            # elif dimension[1] > dimension[0] and dimension[1] > dimension[2]:
            #     return sitk.PermuteAxes(img, [2, 0, 1])
            # elif dimension[0] > dimension[1] and dimension[0] > dimension[2]:
            #     return sitk.PermuteAxes(img, [2, 1, 0])
            # else:
            #     print('problem with dimensions in function "read_DICOM"')
            return img

    elif len(glob.glob(data_directory + "/*.dcm")) == 1:
            print('    read 3D DICOM')
            filename = glob.glob(data_directory + "/*.dcm")[0]
            reader = sitk.ImageFileReader()
            reader.SetFileName(os.path.join(data_directory, filename))
            img = reader.Execute()

    else:
        print('    read DICOMs')
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(data_directory)
        reader.SetFileNames(dicom_names)
        img = reader.Execute()
        # dimension = img.GetSize()
        # if dimension[2] > dimension[0] and dimension[2] > dimension[1]:
        #     return img
        # elif dimension[1] > dimension[0] and dimension[1] > dimension[2]:
        #     return sitk.PermuteAxes(img, [2, 0, 1])
        # elif dimension[0] > dimension[1] and dimension[0] > dimension[2]:
        #     return sitk.PermuteAxes(img, [2, 1, 0])
        # else:
        #     print('problem with dimensions in function "read_DICOM"  ')
    return img

#    return sitk.PermuteAxes(img, [2,0,1])


def read_AIM_header(fileINname):
    """reads an AIM file and provides the corresponding vtk image with spacing, calibration data and header"""
    # read header
    print("     " + fileINname)
    f = open(fileINname, 'rb')
    AIM_ints = read_AIM_ints(f)
    # check AIM version
    if int(AIM_ints[5]) == 16:
        print("     -> version 020")
        if int(AIM_ints[10]) == 131074:
            aimType = 'short'
            print("     -> aimType " + aimType)
        elif int(AIM_ints[10]) == 65537:
            aimType = 'char'
            print("     -> aimType " + aimType)
        elif int(AIM_ints[10]) == 1376257:
            aimType = 'bin compressed'
            print("     -> aimType " + aimType + " not supported! Exiting!")
            exit(1)
        else:
            aimType = 'unknown'
            print("     -> aimType " + aimType + "! Exiting!")
            exit(1)
        aimType_nb = int(AIM_ints[10])
        header = f.read(AIM_ints[2])
    else:
        print("     -> version 030")
        if int(AIM_ints[17]) == 131074:
            aimType = 'short'
            print("     -> aimType " + aimType)
        elif int(AIM_ints[17]) == 65537:
            aimType = 'char'
            print("     -> aimType " + aimType)
        elif int(AIM_ints[17]) == 1376257:
            aimType = 'bin compressed'
            print("     -> aimType " + aimType + " not supported! Exiting!")
            exit(1)
        else:
            aimType = 'unknown'
            print("     -> aimType " + aimType + "! Exiting!")
            exit(1)
        aimType_nb = int(AIM_ints[17])
        header = f.read(AIM_ints[8])
    f.close()
    # collect data from header if existing
    header = re.sub("(?i) +", " ", header)
    header = header.split("\n")
    header.pop(0)
    header.pop(0)
    header.pop(0)
    header.pop(0)
    return header, aimType, aimType_nb


def read_AIM_ints(f):
    """Function by Glen L. Niebur, University of Notre Dame (2010)
    reads the integer data of an AIM file to find its header length"""
    nheaderints = 32
    #   nheaderfloats = 8
    f.seek(0)
    binints = f.read(nheaderints * 4)
    header_int = struct.unpack('=32i', binints)
    return (header_int)


def read_AIM(inFileName, printLog=False):
    print(' ... read AIM file')
    data = open(inFileName).read()
    start = 0
    stop = struct.calcsize('5i')
    sizePreHeader, sizeImStruct, sizeProcLog, sizeImData, sizeImAssocData = struct.unpack('5i', data[start:stop])
    start = stop
    stop = start + struct.calcsize('i')
    start = stop
    stop = start + struct.calcsize('4i')
    start = stop
    stop = start + struct.calcsize('i')
    #    aimTypeNum = struct.unpack('i', data[start:stop])[0]
    #    if aimTypeNum == 131074:
    #        aimType = 'short'
    #    elif aimTypeNum == 65537:
    #        aimType = 'char'
    #    elif aimTypeNum == 1376257:
    #        aimType = 'bin compressed'
    #    else:
    #        aimType = 'unknown'
    header, aimType, aimType_nb = read_AIM_header(inFileName)
    print('     -> ', aimType)
    start = stop
    stop = start + struct.calcsize('3i')
    px, py, pz = struct.unpack('3i', data[start:stop])
    start = stop
    stop = start + struct.calcsize('3i')
    nx, ny, nz = struct.unpack('3i', data[start:stop])
    start = stop
    stop = start + struct.calcsize('3i')
    ox, oy, oz = struct.unpack('3i', data[start:stop])
    start = stop
    stop = start + struct.calcsize('12i')
    start = stop
    stop1 = start + struct.calcsize('B')
    stop2 = start + struct.calcsize('2B')
    stop3 = start + struct.calcsize('3B')
    stop4 = start + struct.calcsize('4B')
    stop = stop4
    bin_swap = struct.pack('cccc', data[stop2:stop3], data[stop3:stop4], data[start:stop1], data[stop1:stop2])
    lx = struct.unpack('f', bin_swap)[0] / 4.0
    start = stop
    stop1 = start + struct.calcsize('B')
    stop2 = start + struct.calcsize('2B')
    stop3 = start + struct.calcsize('3B')
    stop4 = start + struct.calcsize('4B')
    stop = stop4
    bin_swap = struct.pack('cccc', data[stop2:stop3], data[stop3:stop4], data[start:stop1], data[stop1:stop2])
    ly = struct.unpack('f', bin_swap)[0] / 4.0
    start = stop
    stop1 = start + struct.calcsize('B')
    stop2 = start + struct.calcsize('2B')
    stop3 = start + struct.calcsize('3B')
    stop4 = start + struct.calcsize('4B')
    stop = stop4
    bin_swap = struct.pack('cccc', data[stop2:stop3], data[stop3:stop4], data[start:stop1], data[stop1:stop2])
    lz = struct.unpack('f', bin_swap)[0] / 4.0
    start = stop
    stop = start + struct.calcsize('4i')
    start = sizePreHeader + sizeImStruct
    stop = start + sizeProcLog
    time1 = time.clock()
    stdout.write(' ... read & process image data\n')
    stdout.flush()
    start = sizePreHeader + sizeImStruct + sizeProcLog
    if aimType == 'char':
        form = 'i1'
    elif aimType == 'short':
        form = 'i2'
    else:
        stdout.write(' **ERROR** readAimFile() AIM format unknown!\n\n')
        stdout.flush()
    del data
    print('inFileName: ', inFileName)
    #    print('form: ', form
    print('     -> nx: %s ny: %s nz: %s' % (nx, ny, nz))

    #    print('int(nx * ny * nz): ', int(nx * ny * nz)
    #    print('int(start): ', int(start)
    voxelModel = np.core.records.fromfile(inFileName, formats=form, shape=int(nx * ny * nz), offset=int(start))['f0']
    time2 = time.clock()
    stdout.write('     -> read finished in    :   %8.1f sec\n' % (time2 - time1))
    stdout.flush()
    voxelModel = voxelModel.reshape((nz, ny, nx))
    time3 = time.clock()
    stdout.write('     -> reshape finished in :   %8.1f sec\n' % (time3 - time2))
    stdout.flush()
    ndim = [int(nx), int(ny), int(nz)]
    ldim = [float(lx), float(ly), float(lz)]
    additData = {}
    additData['-ldim'] = ldim
    additData['-ndim'] = ndim
    additData['ElementSpacing'] = ldim
    additData['DimSize'] = ndim
    return voxelModel, additData


def read_ISQ(inFileName, BMD=True, echo=True, info=False):
    """
    Read Isq File (Scanco)
    https://github.com/mdoube/BoneJ/blob/master/src/org/bonej/io/ISQReader.java
    """
    if echo:
        print(' ... read isq file \n')
    time1 = time.clock()
    try:
        f = open(inFileName, 'rb')
    except IOError:
        print("\n **ERROR**: read_ISQ: intput file ' % s' not found!\n\n" % inFileName)
        print('\n E N D E D  with ERRORS \n\n')
    for index in np.arange(0, 200, 4):
        f.seek(index)
        print('index %s :          %s' % (index, struct.unpack('i', f.read(4))[0]))
        f.seek(index)
        try:
            print('index %s :          %s' % (index, struct.unpack('c', f.read(4))[0]))
        except:
            print('')
    f.seek(32)
    ct_ID = struct.unpack('i', f.read(4))[0]
    print('scanner ID:                 ', ct_ID)
    if ct_ID != 6020:
        print('!!! unknown muCT -> no slope and intercept known !!!')
    f.seek(28)
    #    sample_nb = struct.unpack('i', f.read(4))[0]
    f.seek(108)
    scanning_time = struct.unpack('i', f.read(4))[0] / 1000
    print('scaning time in ms:         ', scanning_time)
    f.seek(168)
    energy = struct.unpack('i', f.read(4))[0] / 1000.
    print('energy in keV:              ', energy)
    f.seek(172)
    current = struct.unpack('i', f.read(4))[0]
    print('current in muA:             ', current)
    f.seek(44)
    X_pixel = struct.unpack('i', f.read(4))[0]
    print('nb X pixel:                 ', X_pixel)
    f.seek(48)
    Y_pixel = struct.unpack('i', f.read(4))[0]
    print('nb Y pixel:                 ', Y_pixel)
    f.seek(52)
    Z_pixel = struct.unpack('i', f.read(4))[0]
    print('nb Z pixel:                 ', Z_pixel)
    f.seek(56)
    res_general_X = struct.unpack('i', f.read(4))[0]
    print('resolution general X in mu: ', res_general_X)
    f.seek(60)
    res_general_Y = struct.unpack('i', f.read(4))[0]
    print('resolution general Y in mu: ', res_general_Y)
    f.seek(64)
    res_general_Z = struct.unpack('i', f.read(4))[0]
    print('resolution general Z in mu: ', res_general_Z)
    res_X = res_general_X / float(X_pixel)
    print('pixel resolution X in mu:    %.2f' % res_X)
    res_Y = res_general_Y / float(Y_pixel)
    print('pixel resolution Y in mu:    %.2f' % res_Y)
    res_Z = res_general_Z / float(Z_pixel)
    print('pixel resolution Z in mu:    %.2f' % res_Z)
    header_txt = []
    header_txt.append('scanner ID:                 %s' % ct_ID)
    header_txt.append('scaning time in ms:         %s' % scanning_time)
    header_txt.append('scaning time in ms:         %s' % scanning_time)
    header_txt.append('energy in keV:              %s' % energy)
    header_txt.append('current in muA:             %s' % current)
    header_txt.append('nb X pixel:                 %s' % X_pixel)
    header_txt.append('nb Y pixel:                 %s' % Y_pixel)
    header_txt.append('nb Z pixel:                 %s' % Z_pixel)
    header_txt.append('resolution general X in mu: %s' % res_general_X)
    header_txt.append('resolution general Y in mu: %s' % res_general_Y)
    header_txt.append('resolution general Z in mu: %s' % res_general_Z)
    header_txt.append('pixel resolution X in mu:   %.2f' % res_X)
    header_txt.append('pixel resolution Y in mu:   %.2f' % res_Y)
    header_txt.append('pixel resolution Z in mu:   %.2f' % res_Z)
    #    np.savetxt(inFileName.split('.')[0]+'.txt', header_txt)
    write_file = open(inFileName.split('.')[0] + '_info.txt', 'w')
    for item in header_txt:
        write_file.write("%s\n" % item)
    write_file.close()
    f.seek(44)
    header = np.zeros(6)
    for i in range(0, 6):
        header[i] = struct.unpack('i', f.read(4))[0]
    print(header)
    elsp = [header[3] / header[0] / 1000, header[4] / header[1] / 1000, header[5] / header[2] / 1000]
    f.seek(508)
    headersize = 512 * (1 + struct.unpack('i', f.read(4))[0])
    f.seek(headersize)
    if info is True:
        voxelModel = None
    else:
        voxelModel = np.fromfile(f, dtype='i2')
    f.close()
    del f
    ndim = [int(header[0]), int(header[1]), int(header[2])]
    ldim = [float(elsp[0]), float(elsp[1]), float(elsp[2])]
    additData = {}
    additData['-ldim'] = ldim
    additData['-ndim'] = ndim
    additData['ElementSpacing'] = ldim
    additData['DimSize'] = ndim
    additData['HeaderSize'] = headersize
    additData['TransformMatrix'] = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    additData['CenterOfRotation'] = [0.0, 0.0, 0.0]
    additData['Offset'] = [0.0, 0.0, 0.0]
    additData['AnatomicalOrientation'] = 'LPS'
    additData['ElementType'] = 'int16'  # 'float32'
    additData['ElementDataFile'] = inFileName
    time2 = time.clock()
    print('                          t1')
    if echo:
        print('     -> read    data in     : %8.2f sec\n' % (time2 - time1))
    time3 = time.clock()
    if info is False:
        voxelModel = voxelModel.reshape((ndim[2], ndim[1], ndim[0]))
        if echo:
            print('     -> reshape data in     : %8.2f sec\n' % (time3 - time2))

    if ct_ID == 6020 and BMD is True:
        # BE CAREFULL, THIS IS FOR BMD CONVERSION:
        print('muCT 100 of ISTB detected, IS IT CORRECT?')
        slope = 369.154  # ! ATTENTION, dependent on voltage, current and time!!!
        intercept = -191.56
        # slope = 1.27 # ! ATTENTION, dependent on voltage, current and time!!!
        # intercept = -72
        slope = 1  # ! ATTENTION, dependent on voltage, current and time!!!
        intercept = 0
    # Density: slope                         3.69153992e+02
    # Density: intercept                    -1.91559998e+02
    #         slope = 1.2703
    #         intercept = -84.7
    #                       (double(data)./Scaling)*slope + densityIntercept
    #         voxelModel = voxelModel * slope + intercept
    # return voxelModel, additData
    print('before sitk get image')
    voxelModel_sitk = sitk.GetImageFromArray(voxelModel)  # .transpose(2, 1, 0))
    del voxelModel
    print('\n ldim:', ldim)
    voxelModel_sitk.SetSpacing(ldim)
    return voxelModel_sitk, additData


def write_MHD(img, filename, origin=(0, 0, 0), spacing=(1, 1, 1)):
    print('    ... write_MHD')
    if type(img) != np.ndarray:
        pass
    else:
        img = sitk.GetImageFromArray(img.transpose(2, 1, 0))
    img.SetSpacing(spacing)
    img.SetOrigin(origin)
    sitk.WriteImage(sitk.Cast(img, sitk.sitkInt16),
                    filename.split(filename.split('/')[-1].split('.')[0])[0] + filename.split('/')[-1].split('.')[
                        0] + '.mhd')

def write_NII(img, filename, origin=(0, 0, 0), spacing=(1, 1, 1)):
    print('    ... write_NII')
    if type(img) != np.ndarray:
        pass
    else:
        img = sitk.GetImageFromArray(img.transpose(2, 1, 0))
    img.SetSpacing(spacing)
    img.SetOrigin(origin)
    sitk.WriteImage(sitk.Cast(img, sitk.sitkInt16),
                    filename.split(filename.split('/')[-1].split('.')[0])[0] + filename.split('/')[-1].split('.')[
                        0] + '.nii')


def write_DICOM(img, filename):
    print('    ... write_DICOM')
    if type(img) != np.ndarray:
        img_sitk = img
    else:
        img_sitk = sitk.GetImageFromArray(img.transpose(2, 1, 0))
    del img
    # create directory for the dicom:
    if filename.split('.')[-1] == 'ISQ' or filename.split('.')[-1] == 'AIM' or filename.split('.')[-1] == 'mhd':
        directory_name = filename.split('.')[-2]
    else:
        directory_name = filename
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    # Write the 3D image as a series
    # IMPORTANT: There are many DICOM tags that need to be updated when you modify an original image. This is a delicate opration and requires knowlege of
    #            the DICOM standard. This example only modifies some. For a more complete list of tags that need to be modified see:
    #                           http://gdcm.sourceforge.net/wiki/index.php/Writing_DICOM
    #            If it is critical for your work to generate valid DICOM files, It is recommended to use David Clunie's Dicom3tools to validate the files
    #                           (http://www.dclunie.com/dicom3tools.html).

    writer = sitk.ImageFileWriter()
    # Use the study/series/frame of reference information given in the meta-data
    # dictionary and not the automatically generated information from the file IO
    # writer.KeepOriginalImageUIDOn()        #not working, commented by Benjamin Voumard 17.07.2018
    modification_time = time.strftime("%H%M%S")
    modification_date = time.strftime("%Y%m%d")
    # Copy some of the tags and add the relevant tags indicating the change. For the series instance UID (0020|000e), each of the components is a number, cannot start
    # with zero, and separated by a '.' We create a unique series ID using the date and time. tags of interest:
    direction = img_sitk.GetDirection()
    series_tag_values = [("0008|0031", modification_time),  # Series Time
                         ("0008|0021", modification_date),  # Series Date
                         ("0008|0008", "DERIVED\\SECONDARY"),  # Image Type
                         ("0020|000e", "1.2.826.0.1.3680043.2.1125." + modification_date + ".1" + modification_time),
                         # Series Instance UID
                         ("0020|0037",
                          '\\'.join(map(str, (direction[0], direction[3], direction[6],  # Image Orientation (Patient)
                                              direction[1], direction[4], direction[7])))),
                         (
                         "0008|103e", "Created by benjamin.voumard@istb.unibe.ch with SimpleITK")]  # Series Description
    for i in range(img_sitk.GetDepth()):
        image_slice = img_sitk[:, :, i]
        # Tags shared by the series.
        for tag, value in series_tag_values:
            image_slice.SetMetaData(tag, value)
        # Slice specific tags.
        image_slice.SetMetaData("0008|0012", time.strftime("%Y%m%d"))  # Instance Creation Date
        image_slice.SetMetaData("0008|0013", time.strftime("%H%M%S"))  # Instance Creation Time
        # Setting the type to CT preserves the slice location.
        image_slice.SetMetaData("0008|0060", "CT")  # set the type to CT so the thickness is carried over
        # (0020, 0032) image position patient determines the 3D spacing between slices.
        image_slice.SetMetaData("0020|0032", '\\'.join(
            map(str, img_sitk.TransformIndexToPhysicalPoint((0, 0, i)))))  # Image Position (Patient)
        #        image_slice.SetMetaData("0020,0013", str(i)) # Instance Number
        image_slice.SetMetaData("0020,0013", "%04d" % (i))  # Instance Number
        # Write to the output directory and add the extension dcm, to force writing in DICOM format.
        #        writer.SetFileName( directory_name + '/' + str(i) +'.dcm')
        writer.SetFileName(directory_name + '/' + "%04d" % (i) + '.dcm')
        writer.Execute(image_slice)


def write_STL(img, filename, factor=5):
    print('    ... write_STL')
    from skimage import measure
    from stl import mesh
    if type(img) != np.ndarray:
        img_np = sitk.GetArrayFromImage(img).transpose(2, 1, 0)
    verts, faces, normals, values = measure.marching_cubes_lewiner(img_np[::factor, ::factor, ::factor], 0)
    my_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            my_mesh.vectors[i][j] = verts[f[j], :]
    my_mesh.save(
        filename.split(filename.split('/')[-1].split('.')[0])[0] + filename.split('/')[-1].split('.')[0] + '.stl')


def write_Imesh(curVoxelModel, resolution, displacement, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    print('    ... write IMESH')
    curVoxelModel = curVoxelModel.astype(np.int32)
    nx, ny, nz = np.flip(curVoxelModel.shape, -1)
    maxVox = np.max(curVoxelModel)
    maxVox = int(maxVox + 0.5)
    noid = 0
    activeNodes = {}
    nodeSets = {}
    nodeSets['ALL_NODE_S'] = []
    nodeSets['ALL_NODE_N'] = []
    nodeSets['ALL_NODE_E'] = []
    nodeSets['ALL_NODE_W'] = []
    nodeSets['ALL_NODE_T'] = []
    nodeSets['ALL_NODE_B'] = []
    elsetNodes = {}
    ranges = {}
    thresList = []
    matTemplateFilename = 'nothing'
    rangeMin = 1  # was before....
    rangeMax = 255
    ranges[matTemplateFilename] = (rangeMin, rangeMax)
    for crange in ranges:
        for matId in range(ranges[crange][0], ranges[crange][1] + 1):
            elsetNodes[repr(matId)] = []
            thresList.append(matId)
    elid = 0
    nx1 = nx + 1
    nxy1 = (ny + 1) * (nx + 1)

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                grayValue = curVoxelModel[k, j, i]
                if elsetNodes.has_key(repr(grayValue)):
                    elid = elid + 1
                    elnds = [nxy1 * k + nx1 * j + (i + 1),
                             nxy1 * k + nx1 * j + (i + 2),
                             nxy1 * k + nx1 * (j + 1) + (i + 2),
                             nxy1 * k + nx1 * (j + 1) + (i + 1),
                             nxy1 * (k + 1) + nx1 * j + (i + 1),
                             nxy1 * (k + 1) + nx1 * j + (i + 2),
                             nxy1 * (k + 1) + nx1 * (j + 1) + (i + 2),
                             nxy1 * (k + 1) + nx1 * (j + 1) + (i + 1)]
                    elsetNodes[repr(grayValue)].append((elid, elnds))
    for matid in thresList:
        if len(elsetNodes[repr(matid)]) > 0:
            for elnds in elsetNodes[repr(matid)]:
                elid = elnds[0]
                for elnd in elnds[1]:
                    activeNodes[elnd] = 1
    # write COORdinates: *******************************************************************************************
    outfile = open(folder + 'temp2.txt', 'w')
    outfile.write("COORdinates\n")
    nodeCoord = {}
    corresp_list = np.zeros(10000000)
    count = 1
    noid = 0
    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                noid = noid + 1
                if activeNodes.has_key(noid):
                    nodeCoord[noid] = (float(i * resolution[0]), float(j * resolution[1]), float(k * resolution[2]))
                    outfile.write(' % 4s 0  %.5e  %.5e  %.5e\n' % (
                    count, float(i * resolution[0]), float(j * resolution[1]), float(k * resolution[2])))
                    corresp_list[noid] = count
                    count += 1
    outfile.write('\n')
    outfile.close()
    # write ELEMents: **********************************************************************************************
    print('     -> Nodes writing')
    outfile = open(folder + 'temp3.txt', 'w')
    outfile.write('ELEMents \n')
    for matid in thresList:
        if len(elsetNodes[repr(matid)]) > 0:
            for elnds in elsetNodes[repr(matid)]:
                elid = elnds[0]
                #                outfile.write(' % 4s 0 1 %4s %4s %4s %4s %4s %4s %4s %4s\n' % (elid, elnds[1][0], elnds[1][1], elnds[1][2], elnds[1][3], elnds[1][4], elnds[1][5], elnds[1][6], elnds[1][7]))
                #                outfile.write(' % 4s 0 1 %4s %4s %4s %4s %4s %4s %4s %4s\n' %(elid, int(corresp_list[elnds[1][0]]), int(corresp_list[elnds[1][1]]), int(corresp_list[elnds[1][2]]), int(corresp_list[elnds[1][3]]), int(corresp_list[elnds[1][4]]), int(corresp_list[elnds[1][5]]), int(corresp_list[elnds[1][6]]), int(corresp_list[elnds[1][7]])))
                #                A E F B D H G C:                                                                  A                              B                               C                                D                                E                             F                                 G                              H
                outfile.write(' % 4s 0 1 %4s %4s %4s %4s %4s %4s %4s %4s\n' % (
                elid, int(corresp_list[elnds[1][0]]), int(corresp_list[elnds[1][4]]), int(corresp_list[elnds[1][5]]),
                int(corresp_list[elnds[1][1]]), int(corresp_list[elnds[1][3]]), int(corresp_list[elnds[1][7]]),
                int(corresp_list[elnds[1][6]]), int(corresp_list[elnds[1][2]])))
                for elnd in elnds[1]:
                    activeNodes[elnd] = 1
    outfile.close()
    print('     -> Nodes Top, Bottom, writing')
    nb_nodes = count - 1  # noid
    nb_element = elid
    noid = 0
    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                noid = noid + 1
                if activeNodes.has_key(noid):
                    if k == 0:
                        nodeSets['ALL_NODE_B'].append(corresp_list[noid])
                    if k == nz:
                        nodeSets['ALL_NODE_T'].append(corresp_list[noid])
                    if j == 0:
                        nodeSets['ALL_NODE_S'].append(noid)
                    if j == ny:
                        nodeSets['ALL_NODE_N'].append(noid)
                    if i == 0:
                        nodeSets['ALL_NODE_W'].append(noid)
                    if i == nx:
                        nodeSets['ALL_NODE_E'].append(noid)
    stdout.flush()
    # write BOUNdary: ***********************************************************************************************
    outfile = open(folder + 'temp4.txt', 'w')
    outfile.write('BOUNdary restraints\n')
    #    for i in np.arange(nb_nodes)+1:
    for i in nodeSets['ALL_NODE_B']:  # fix bottom nodes
        outfile.write(' % 4s 0 1 1 1\n' % int(i))
    for i in nodeSets['ALL_NODE_T']:  # fix bottom nodes
        outfile.write(' % 4s 0 1 1 1\n' % int(i))
    outfile.write('\n')
    # write DISPlacement: *******************************************************************************************
    outfile.write('DISPlacement loading\n')
    for i in nodeSets['ALL_NODE_T']:
        outfile.write(' % s 0  %.5e  %.5e -%.5e\n' % (int(i), 0, 0, displacement))
    outfile.write('\n')
    outfile.write('INCLude, mate \n')
    outfile.write('\n')
    outfile.write('END mesh \n')
    outfile.write('\n')
    outfile.write('INCLude part.mesh \n')
    outfile.write('\n')
    outfile.write('STOP\n')
    outfile.write('\n')
    outfile.write('\n')
    outfile.close()
    # write header: ************************************************************************************************
    outfile = open(folder + 'temp1.txt', 'w')
    outfile.write("NOCOunt\n")
    outfile.write("FEAP Benjamin Voumard\n")
    outfile.write("        %s        %s           1  3  3  8\n" % (nb_nodes, nb_element))
    outfile.write("\n")
    outfile.write("NOPArse\n")
    outfile.write("NOPRint\n")
    outfile.write("\n")
    outfile.write("1 \n")
    outfile.close()
    filenames = [folder + 'temp1.txt', folder + 'temp2.txt', folder + 'temp3.txt', folder + 'temp4.txt']
    with open(folder + 'Imesh', 'w') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                outfile.write(infile.read())
    for filename in filenames:
        os.remove(filename)


def cut_3d(img, title=''):
    if type(img) != np.ndarray:
        img_sitk = img
    else:
        img_sitk = sitk.GetImageFromArray(img.transpose(2, 1, 0))
    size = img_sitk.GetSize()
    reso = img_sitk.GetSpacing()
    imgX_sitk = img_sitk[size[0] / 2, :, :]
    imgY_sitk = img_sitk[:, size[1] / 2, :]
    imgZ_sitk = img_sitk[:, :, size[2] / 2]
    fig = plt.figure()
    #    plt.set_cmap("gray")
    plt.gray()
    i = 1
    for imges in [imgX_sitk, imgY_sitk, imgZ_sitk]:
        fig.add_subplot(1, 3, i)
        image = sitk.GetArrayFromImage(imges)
        extent = (0, image.shape[1] * reso[1], image.shape[0] * reso[0], 0)
        plt.imshow(image, extent=extent, interpolation=None)
        plt.title(title)
        i += 1
    plt.show()


def cut_3d_2_images(img1, img2, color='b', title=''):
    import matplotlib as mpl
    if type(img1) != np.ndarray:
        img1_sitk = img1
    else:
        img1_sitk = sitk.GetImageFromArray(img1.transpose(2, 1, 0))
    if type(img2) != np.ndarray:
        img2_sitk = img2
        img2_np = sitk.GetArrayFromImage(img2)
    else:
        img2_sitk = sitk.GetImageFromArray(img2.transpose(2, 1, 0))
        img2_np = img2
    size1 = img1_sitk.GetSize()
    size2 = img2_sitk.GetSize()
    reso1 = img1_sitk.GetSpacing()
    reso2 = img2_sitk.GetSpacing()
    img2_np_inverted = (img2_np + 1 + -2 * img2_np)
    img2_sitk_inverted = sitk.GetImageFromArray(img2_np_inverted)
    plt.set_cmap("gray")
    if color == 'b':
        cmap2 = mpl.colors.LinearSegmentedColormap.from_list('my_cmap2', ['green', 'blue'], 256)
    if color == 'r':
        cmap2 = mpl.colors.LinearSegmentedColormap.from_list('my_cmap2', ['green', 'red'], 256)
    cmap2._init()  # create the _lut array, with rgba values
    alphas = np.linspace(0, .5, cmap2.N + 3)
    cmap2._lut[:, -1] = alphas
    img1X = sitk.GetArrayFromImage(img1_sitk[size1[0] / 2, :, :])
    img1Y = sitk.GetArrayFromImage(img1_sitk[:, size1[1] / 2, :])
    img1Z = sitk.GetArrayFromImage(img1_sitk[:, :, size1[2] / 2])
    img2X = sitk.GetArrayFromImage(img2_sitk_inverted[size2[0] / 2, :, :])
    img2Y = sitk.GetArrayFromImage(img2_sitk_inverted[:, size2[1] / 2, :])
    img2Z = sitk.GetArrayFromImage(img2_sitk_inverted[:, :, size2[2] / 2])
    fig = plt.figure()
    ax1 = fig.add_subplot(131)
    extent1 = (0, img1X.shape[1] * reso1[0], img1X.shape[0] * reso1[0], 0)
    extent2 = (0, img2X.shape[1] * reso2[0], img2X.shape[0] * reso2[0], 0)
    ax1.imshow(img1X, extent=extent1, interpolation=None)
    ax1.imshow(img2X, extent=extent2, interpolation=None, cmap=cmap2)
    ax2 = fig.add_subplot(132)
    extent3 = (0, img1Y.shape[1] * reso1[1], img1Y.shape[0] * reso1[1], 0)
    extent4 = (0, img2Y.shape[1] * reso2[1], img2Y.shape[0] * reso2[1], 0)
    ax2.imshow(img1Y, extent=extent3, interpolation=None)
    ax2.imshow(img2Y, extent=extent4, interpolation=None, cmap=cmap2)
    ax3 = fig.add_subplot(133)
    extent5 = (0, img1Z.shape[1] * reso1[2], img1Z.shape[0] * reso1[2], 0)
    extent6 = (0, img2Z.shape[1] * reso2[2], img2Z.shape[0] * reso2[2], 0)
    ax3.imshow(img1Z, extent=extent5, interpolation=None)
    ax3.imshow(img2Z, extent=extent6, interpolation=None, cmap=cmap2)
    plt.show()


def plot_stack(image, title='', play_fps=100, big_scroll=30, key_func=None, view='XY', *args, block=False):
    from matplotlib.widgets import Slider
    from matplotlib.animation import FuncAnimation
    """https://github.com/bilylee/videofig/blob/master/videofig.py
    Figure with horizontal scrollbar and play capabilities + mouse wheel scroll by Benjamin Voumard
    This script is mainly inspired by the elegant work of João Filipe Henriques
      https://www.mathworks.com/matlabcentral/fileexchange/29544-figure-to-play-and-analyze-videos-with-custom-plots-on-top?focused=5172704&tab=function
    :param num_frames: an integer, number of frames in a sequence
    :param redraw_func: callable with signature redraw_func(f, axes) used to draw a new frame at position f using axes, which is a instance of Axes class in matplotlib
    :param play_fps: an integer, number of frames per second, used to control the play speed
    :param big_scroll: an integer, big scroll number used when pressed page down or page up keys.
    :param key_func: optional callable which signature key_func(key), used to provide custom key shortcuts.
    :param args: other optional arguments
    :return: None
    """
    if type(image) != np.ndarray:
        reso = image.GetSpacing()
        image = sitk.GetArrayFromImage(image).transpose(1, 2, 0)
    else:
        image = image.transpose(1, 0, 2)
        reso = (1., 1., 1.)

    def check_int_scalar(a, name):
        assert isinstance(a, int), '{} must be a int scalar, instead of {}'.format(name, type(name))

    def check_callback(a, name):
        # Check http://stackoverflow.com/questions/624926/how-to-detect-whether-a-python-variable-is-a-function
        assert callable(a), '{} must be callable, instead of {}'.format(name, type(name))

    # Check arguments
    check_int_scalar(play_fps, 'play_fps')
    check_int_scalar(big_scroll, 'big_scroll')
    if key_func:
        check_callback(key_func, 'key_func')
    vmin = np.min(image)
    vmax = np.max(image)
    # Initialize figure
    plt.ioff()
    fig_handle = plt.figure()
    plt.set_cmap("gray")  # !to be commented to run on the server (skull)!
    # main drawing axes for video display
    axes_handle = fig_handle.add_axes([0.05, 0.1, .95, 0.9])
    #    axes_handle.set_axis_off()
    # Build scrollbar
    scroll_axes_handle = fig_handle.add_axes([0.08, 0, .84, 0.05])  # , facecolor='orange')
    #    extent = (0, image.shape[1]*reso[1], image.shape[0]*reso[0], 0)
    if view == 'XY':
        # extent = horizontal_min,horizontal_max,vertical_min,vertical_max
        extent = (0, image.shape[1] * reso[1], image.shape[0] * reso[0], 0)
        #        extent = (0, image.shape[0] * reso[0], 0, image.shape[1] * reso[1])
        num_frames = image.shape[2]
    if view == 'XZ':
        extent = (0, image.shape[0] * reso[0], image.shape[2] * reso[2], 0)
        # extent = (0, image.shape[0] * reso[0], 0, image.shape[2] * reso[2])
        num_frames = image.shape[1]
    if view == 'YZ':
        extent = (0, image.shape[1] * reso[1], image.shape[2] * reso[2], 0)
        # extent = (0, image.shape[1] * reso[1], 0, image.shape[2] * reso[2])
        num_frames = image.shape[0]
    scroll_handle = Slider(scroll_axes_handle, '', 0.0, num_frames - 1, valinit=np.rint(num_frames / 2).astype('int'))
    if np.max(image[:, :,
              0]) == 0:  # if the first slice does not contain values > 0 (-> blank slice), the stack is completly black -> solution: add a pixel at 0, 0, 0 with the value 1
        image[0, 0, 0] = 1

    def redraw_func(f, axes, image):  # <---  function put inside by Benjamin Voumard 08.06.2018
        if view == 'XY':
            slice2plot = image[:, :, f]
        if view == 'XZ':
            slice2plot = image[:, f, :].T
        if view == 'YZ':
            slice2plot = image[f, :, :].T
        if not redraw_func.initialized:
            redraw_func.im = axes.imshow(slice2plot, extent=extent, interpolation=None, vmin=vmin, vmax=vmax)
            redraw_func.initialized = True
        else:
            redraw_func.im.set_array(slice2plot)

    redraw_func.initialized = False
    check_callback(redraw_func, 'redraw_func')

    def draw_new(_):
        # Set to the right axes and call the custom redraw function
        plt.sca(axes_handle)
        redraw_func(int(scroll_handle.val), axes_handle, image)
        fig_handle.canvas.draw_idle()

    def scroll(new_f):
        new_f = int(min(max(new_f, 0), num_frames - 1))  # clip in the range of [0, num_frames - 1]
        cur_f = scroll_handle.val
        # Stop player at the end of the sequence
        if new_f == (num_frames - 1):
            play.running = False
        if cur_f != new_f:
            # move scroll bar to new position
            scroll_handle.set_val(new_f)
        return axes_handle

    def play(period):
        play.running ^= True  # Toggle state
        if play.running:
            frame_idxs = range(int(scroll_handle.val), num_frames)
            play.anim = FuncAnimation(fig_handle, scroll, frame_idxs, interval=1000 * period, repeat=False)
            plt.draw()
        else:
            play.anim.event_source.stop()

    # Set initial player state
    play.running = False

    def key_press(event):
        key = event.key
        f = scroll_handle.val
        if key == 'left':
            scroll(f - 1)
        elif key == 'right':
            scroll(f + 1)
        elif key == 'pageup':
            scroll(f - big_scroll)
        elif key == 'pagedown':
            scroll(f + big_scroll)
        elif key == 'scroll':
            scroll(f + 1)
        elif key == 'home':
            scroll(0)
        elif key == 'end':
            scroll(num_frames - 1)
        elif key == 'enter' or key == ' ':  # <---  added by Benjamin Voumard 08.06.2018
            play(1 / play_fps)
        elif key == 'backspace':
            play(5 / play_fps)
        # elif event.button == 3:             # <---  added by Benjamin Voumard 08.06.2018
        #     play(1 / play_fps)
        try:
            if event.button == 'up':  # <---  added by Benjamin Voumard 08.06.2018
                scroll(f - 1)
            if event.button == 'down':  # <---  added by Benjamin Voumard 08.06.2018
                scroll(f + 1)
        except:
            pass
        else:
            if key_func:
                key_func(key)

    # Register events
    scroll_handle.on_changed(draw_new)
    fig_handle.canvas.mpl_connect('key_press_event', key_press)
    fig_handle.canvas.mpl_connect('scroll_event', key_press)  # <---  added by Benjamin Voumard 08.06.2018
    fig_handle.canvas.mpl_connect('button_press_event', key_press)  # <---  added by Benjamin Voumard 08.06.2018
    #    plt.gcf().text(0.94, 0.07, '/%s' % image.shape[2], fontsize=10)
    if view.upper() == 'XY':
        plt.gcf().text(0.92, 0.07, '/%s' % (image.shape[2]), fontsize=10)
    if view.upper() == 'XZ':
        plt.gcf().text(0.92, 0.07, '/%s' % (image.shape[1]), fontsize=10)
    if view.upper() == 'YZ':
        plt.gcf().text(0.92, 0.07, '/%s' % (image.shape[0]), fontsize=10)
    # Draw initial frame
    redraw_func(np.rint(num_frames / 2).astype('int'), axes_handle, image)
    #    redraw_func(np.rint(image.shape[2]/2*reso[2]).astype('int'), axes_handle, image)
    # Start playing
    #    play(1 / play_fps)
    plt.title(title)
    #    plt.tight_layout()     # !to be commented to run on the server (skull)!
    if socket.gethostname() == 'ben-PC':
        fig_handle.tight_layout()
    # plt.show()
    plt.show(block=block)


def plot_special(image, title='', play_fps=100, big_scroll=30, key_func=None, view='XY', *args):
    from matplotlib.widgets import Slider
    from matplotlib.animation import FuncAnimation
    if type(image) != np.ndarray:
        reso = image.GetSpacing()
        image = sitk.GetArrayFromImage(image).transpose(1, 2, 0)
    else:
        image = image.transpose(1, 0, 2)
        reso = (1., 1., 1.)

    def check_int_scalar(a, name):
        assert isinstance(a, int), '{} must be a int scalar, instead of {}'.format(name, type(name))

    def check_callback(a, name):
        # Check http://stackoverflow.com/questions/624926/how-to-detect-whether-a-python-variable-is-a-function
        assert callable(a), '{} must be callable, instead of {}'.format(name, type(name))

    # Check arguments
    check_int_scalar(play_fps, 'play_fps')
    check_int_scalar(big_scroll, 'big_scroll')
    if key_func:
        check_callback(key_func, 'key_func')
    vmin = np.min(image)
    vmax = np.max(image)
    fig_handle, ax = plt.subplots(2, 2, figsize=(19.2 * .6, 10.8 * .6))  # , sharex='col', sharey='row')
    plt.set_cmap("gray")  # !to be commented to run on the server (skull)!
    ax1 = ax[0, 0]
    ax2 = ax[0, 1]
    ax3 = ax[1, 1]
    # ax4 = ax[1,0]
    print('image.shape', image.shape)
    print('image reso', reso)
    print('image mm', np.array(reso) * np.array(image.shape))
    # Build scrollbar                          x1,   y1,   length,  height
    scroll_ax1 = fig_handle.add_axes([0.12, 0.97, 0.35, 0.03])
    scroll_ax2 = fig_handle.add_axes([0.55, 0.97, 0.35, 0.03])
    scroll_ax3 = fig_handle.add_axes([0.55, 0.00, 0.35, 0.03])
    extent1 = (0, image.shape[1] * reso[1], image.shape[0] * reso[0], 0)
    num_frames1 = image.shape[2]
    extent2 = (0, image.shape[0] * reso[0], image.shape[2] * reso[2], 0)
    num_frames2 = image.shape[1]
    extent3 = (0, image.shape[1] * reso[1], image.shape[2] * reso[2], 0)
    num_frames3 = image.shape[0]
    scroll_handle1 = Slider(scroll_ax1, '', 0.0, num_frames1 - 1, valinit=np.rint(num_frames1 / 2).astype('int'))
    scroll_handle2 = Slider(scroll_ax2, '', 0.0, num_frames2 - 1, valinit=np.rint(num_frames2 / 2).astype('int'))
    scroll_handle3 = Slider(scroll_ax3, '', 0.0, num_frames3 - 1, valinit=np.rint(num_frames3 / 2).astype('int'))
    if np.max(image[:, :,
              0]) == 0:  # if the first slice does not contain values > 0 (-> blank slice), the stack is completly black -> solution: add a pixel at 0, 0, 0 with the value 1
        image[0, 0, 0] = 1

    def redraw_func1(f1, axes, image):  # <---  function put inside by Benjamin Voumard 08.06.2018
        slice2plot1 = image[:, :, f1]
        if not redraw_func1.initialized:
            redraw_func1.im = axes.imshow(slice2plot1, extent=extent1, interpolation=None, vmin=vmin, vmax=vmax)
            redraw_func1.initialized = True
        else:
            redraw_func1.im.set_array(slice2plot1)

    def redraw_func2(f2, axes, image):  # <---  function put inside by Benjamin Voumard 08.06.2018
        slice2plot2 = image[:, f2, :].T
        if not redraw_func2.initialized:
            redraw_func2.im = axes.imshow(slice2plot2, extent=extent2, interpolation=None, vmin=vmin, vmax=vmax)
            redraw_func2.initialized = True
        else:
            redraw_func2.im.set_array(slice2plot2)

    def redraw_func3(f3, axes, image):  # <---  function put inside by Benjamin Voumard 08.06.2018
        slice2plot3 = image[f3, :, :].T
        if not redraw_func3.initialized:
            redraw_func3.im = axes.imshow(slice2plot3, extent=extent3, interpolation=None, vmin=vmin, vmax=vmax)
            redraw_func3.initialized = True
        else:
            redraw_func3.im.set_array(slice2plot3)

    redraw_func1.initialized = False
    check_callback(redraw_func1, 'redraw_func')
    redraw_func2.initialized = False
    check_callback(redraw_func2, 'redraw_func')
    redraw_func3.initialized = False
    check_callback(redraw_func3, 'redraw_func')

    def draw_new(_):
        # Set to the right axes and call the custom redraw function
        redraw_func1(int(scroll_handle1.val), ax1, image)
        redraw_func2(int(scroll_handle2.val), ax2, image)
        redraw_func3(int(scroll_handle3.val), ax3, image)
        fig_handle.canvas.draw_idle()

    def scroll1(new_f1):
        new_f1 = int(min(max(new_f1, 0), num_frames1 - 1))
        cur_f1 = scroll_handle1.val
        if new_f1 == (num_frames1 - 1):
            play.running = False
        if cur_f1 != new_f1:
            scroll_handle1.set_val(new_f1)
        return ax1

    def scroll2(new_f2):
        new_f2 = int(min(max(new_f2, 0), num_frames2 - 1))
        cur_f2 = scroll_handle2.val
        if new_f2 == (num_frames2 - 1):
            play.running = False
        if cur_f2 != new_f2:
            scroll_handle2.set_val(new_f2)
        return ax2

    def scroll3(new_f3):
        new_f3 = int(min(max(new_f3, 0), num_frames3 - 1))
        cur_f3 = scroll_handle3.val
        if new_f3 == (num_frames3 - 1):
            play.running = False
        if cur_f3 != new_f3:
            scroll_handle3.set_val(new_f3)
        return ax3

    def play(period):
        play.running ^= True  # Toggle state
        if play.running:
            frame_idxs1 = range(int(scroll_handle1.val), num_frames1)
            frame_idxs2 = range(int(scroll_handle2.val), num_frames2)
            frame_idxs3 = range(int(scroll_handle3.val), num_frames3)
            play.anim = FuncAnimation(fig_handle, scroll1, frame_idxs1, interval=1000 * period, repeat=False)
            play.anim = FuncAnimation(fig_handle, scroll2, frame_idxs2, interval=1000 * period, repeat=False)
            play.anim = FuncAnimation(fig_handle, scroll3, frame_idxs3, interval=1000 * period, repeat=False)
            plt.draw()
        else:
            play.anim.event_source.stop()

    play.running = False

    def key_press(event):
        key = event.key
        f1 = scroll_handle1.val
        f2 = scroll_handle2.val
        f3 = scroll_handle3.val
        if key == 'left':
            scroll1(f1 - 1)
            scroll2(f2 - 1)
            scroll3(f3 - 1)
        elif key == 'right':
            scroll1(f1 + 1)
            scroll2(f2 + 1)
            scroll3(f3 + 1)
        elif key == 'pageup':
            scroll1(f1 - big_scroll)
            scroll2(f2 - big_scroll)
            scroll3(f3 - big_scroll)
        elif key == 'pagedown':
            scroll1(f1 + big_scroll)
            scroll2(f2 + big_scroll)
            scroll3(f3 + big_scroll)
        elif key == 'home':
            scroll1(0)
            scroll2(0)
            scroll3(0)
        elif key == 'end':
            scroll1(num_frames1 - 1)
            scroll2(num_frames2 - 1)
            scroll3(num_frames3 - 1)
        elif key == 'enter' or key == ' ':  # <---  added by Benjamin Voumard 08.06.2018
            play(1 / play_fps)
        elif key == 'backspace':
            play(5 / play_fps)
        elif event.button == 3:  # <---  added by Benjamin Voumard 08.06.2018
            play(1 / play_fps)
        else:
            if key_func:
                key_func(key)

    scroll_handle1.on_changed(draw_new)
    scroll_handle2.on_changed(draw_new)
    scroll_handle3.on_changed(draw_new)
    fig_handle.canvas.mpl_connect('key_press_event', key_press)
    fig_handle.canvas.mpl_connect('scroll_event', key_press)  # <---  added by Benjamin Voumard 08.06.2018
    fig_handle.canvas.mpl_connect('button_press_event', key_press)  # <---  added by Benjamin Voumard 08.06.2018

    def onclick(event):
        for i, ax in enumerate([ax1, ax2, ax3]):
            if ax == event.inaxes:
                if i == 0:
                    redraw_func2(np.rint(mp.actual_coord[0] / reso[0]).astype('int'), ax2, image)
                    redraw_func3(np.rint(mp.actual_coord[1] / reso[2]).astype('int'), ax3, image)
                    scroll2(np.rint(mp.actual_coord[0] / reso[1]).astype('int') - 1)
                    scroll3(np.rint(mp.actual_coord[1] / reso[2]).astype('int') - 1)
                if i == 1:
                    redraw_func1(np.rint(mp.actual_coord[1] / reso[0]).astype('int'), ax1, image)
                    redraw_func3(np.rint(mp.actual_coord[0] / reso[2]).astype('int'), ax3, image)
                    scroll1(np.rint(mp.actual_coord[1] / reso[0]).astype('int') - 1)
                    scroll3(np.rint(mp.actual_coord[0] / reso[2]).astype('int') - 1)
                if i == 2:
                    redraw_func1(np.rint(mp.actual_coord[1] / reso[0]).astype('int'), ax1, image)
                    redraw_func2(np.rint(mp.actual_coord[0] / reso[1]).astype('int'), ax2, image)
                    scroll1(np.rint(mp.actual_coord[1] / reso[0]).astype('int') - 1)
                    scroll2(np.rint(mp.actual_coord[0] / reso[1]).astype('int') - 1)

    def scrollme(event):
        f1 = scroll_handle1.val
        f2 = scroll_handle2.val
        f3 = scroll_handle3.val
        for i, ax in enumerate([ax1, ax2, ax3]):
            if ax == event.inaxes:
                print('i', i)
                if i == 0:
                    if event.button == 'up':
                        scroll1(f1 - 1)
                    if event.button == 'down':
                        scroll1(f1 + 1)
                if i == 1:
                    if event.button == 'up':
                        scroll2(f2 - 1)
                    if event.button == 'down':
                        scroll2(f2 + 1)
                if i == 2:
                    if event.button == 'up':
                        scroll3(f3 - 1)
                    if event.button == 'down':
                        scroll3(f3 + 1)

    redraw_func2(np.rint(num_frames2 / 2).astype('int'), ax2, image)
    redraw_func3(np.rint(num_frames3 / 2).astype('int'), ax3, image)
    redraw_func1(np.rint(num_frames1 / 2).astype('int'), ax1, image)
    plt.title(title)
    #    ax4.set_title(title)
    fig_handle.delaxes(ax[1, 0])
    # plt.show()
    plt.show(block=False)

    mp = GetMousePosition(fig_handle)
    mp.connect()
    fig_handle.canvas.mpl_connect('motion_notify_event', onclick)
    fig_handle.canvas.mpl_connect('scroll_event', scrollme)
    #plt.tight_layout()
    return mp


class GetMousePosition:
    lock = None

    def __init__(self, figure):
        self.figure = figure
        self.press = None
        self.background = None
        self.actual_coord = [0, 0]

    def connect(self):
        'connect to all the events we need'
        self.cidpress = self.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = self.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = self.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

    #        return actual_coord

    def on_press(self, event):
        'on button press we will see if the mouse is over us and store some data'
        if GetMousePosition.lock is not None:
            return
        self.press = event.xdata, event.ydata
        GetMousePosition.lock = self

    def on_motion(self, event):
        'on motion we will move the rect if the mouse is over us'
        if GetMousePosition.lock is not self:
            return
        self.actual_coord = [event.xdata, event.ydata]

    def on_release(self, event):
        'on release we reset the press data'
        if GetMousePosition.lock is not self:
            return
        self.press = None
        GetMousePosition.lock = None
        self.background = None
        self.figure.canvas.draw()


def mayavi_plot_mesh(image, factor=1, opacity=1, fig=None, color=(0.69163319, 0.58612982, 0.42201347), title=''):
    if type(image) != np.ndarray:
        image = sitk.GetArrayFromImage(image).transpose(2, 1, 0)
    else:
        try:
            if len(image.shape) == 2:
                if image.shape[1] == 3:
                    # coordinate detected...
                    image = coordinate_to_array_3d(image)
        except:
            if len(np.array(image).shape) == 2:
                if np.array(image).shape[1] == 3:
                    # coordinate detected...
                    image = coordinate_to_array_3d(np.array(image))
    if fig is None:
        fig = mlab.figure()
    else:
        pass
    verts, faces, normals, values = measure.marching_cubes_lewiner(image[::factor, ::factor, ::factor], 0)
    # my_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    # for i, f in enumerate(faces):
    #     for j in range(3):
    #         my_mesh.vectors[i][j] = verts[f[j], :]
    mlab.triangular_mesh(verts[:, 0], verts[:, 1], verts[:, 2], faces, opacity=opacity, color=color)
    mlab.title(title)
    return fig


def mayavi_plot_vtk(vtkFile):
    """
    plots a given vtk file in 3d
    :param vtkFile: path to the vtk file or a vtkpolydata/tvtkpolydata instance
    :return:
    """
    import tvtk

    # Create the MayaVi engine and start it.
    engine = mayavi.core.api.Engine()
    engine.start()
    scene = engine.new_scene()

    if type(vtkFile) is str:
        # Read in VTK file and add as source
        reader = mayavi.sources.vtk_file_reader.VTKFileReader()
        reader.initialize(vtkFile)
        engine.add_source(reader)

    elif type(vtkFile) is vtk.vtkPolyData:
        tvtkFile = tvtk.api.tvtk.to_tvtk(vtkFile)
        src = mayavi.sources.vtk_data_source.VTKDataSource(data=tvtkFile)
        engine.add_source(src)
    elif type(vtkFile) is tvtk.tvtk_classes.poly_data.PolyData:
        src = mayavi.sources.vtk_data_source.VTKDataSource(data=vtkFile)
        engine.add_source(src)
    # elif type(vtkFile) is vtk.vtkAlgorithmOutput:

    else:
        print('unsupported datatype')


    # Add Surface Module
    surface = mayavi.modules.surface.Surface()
    engine.add_module(surface)

    # Move the camera
    scene.scene.camera.elevation(-70)

    # Create a GUI instance and start the event loop.
    # This stops the window from closing

    gui = pyface.api.GUI()
    gui.start_event_loop()


def visvis_plot_mesh(image, factor=1, opacity=1, fig=None, color=(0.69163319, 0.58612982, 0.42201347)):
    if type(image) != np.ndarray:
        image = sitk.GetArrayFromImage(image).transpose(2, 1, 0)
    else:
        try:
            if len(image.shape) == 2:
                if image.shape[1] == 3:
                    # coordinate detected...
                    image = coordinate_to_array_3d(image)
        except:
            if len(np.array(image).shape) == 2:
                if np.array(image).shape[1] == 3:
                    # coordinate detected...
                    image = coordinate_to_array_3d(np.array(image))
    if fig is None:
        fig = mlab.figure()
    else:
        pass
    verts, faces, normals, values = measure.marching_cubes_lewiner(image[::factor, ::factor, ::factor], 0)
    vv.mesh(np.fliplr(verts), faces, normals, values)
    vv.use().Run()
    return fig


def plot_2d(image, title=''):
    """plot_2d plots a 2d numpy array with imshow (with XY transposed)"""
    if type(image) != np.ndarray:  # check type of img (numpy or simpleitk)
        img_np = sitk.GetArrayFromImage(image).transpose(1, 0)
    else:
        img_np = image
    plt.figure()
    plt.imshow(img_np.transpose(1, 0))
    plt.title(title)
    plt.show()


def mayavi_add_sphere(fig, center, radius=3, color=(1, 0, 0)):
    from tvtk.api import tvtk
    from tvtk.common import configure_input_data
    sphere = tvtk.SphereSource(center=(center[0], center[1], center[2]), radius=radius)
    sphere_mapper = tvtk.PolyDataMapper()
    configure_input_data(sphere_mapper, sphere.output)
    sphere.update()
    p = tvtk.Property(opacity=0.99, color=color)
    sphere_actor = tvtk.Actor(mapper=sphere_mapper, property=p)
    fig.scene.add_actor(sphere_actor)


def mayavi_add_sphere_list(fig, coord_list, radius=3, color=(1, 0, 0)):
    for i in np.arange(len(coord_list)):
        mayavi_add_sphere(fig, coord_list[i, :], radius=radius, color=color)


def mayavi_arrow_R_t(fig, R=np.eye(3), t=np.zeros(3), arrow_len=1.5):
    from tvtk.tools import visual
    print('    mayavi_arrow_R_t')
    x1 = t
    x2 = t + R[:, 0]
    y1 = t
    y2 = t + R[:, 1]
    z1 = t
    z2 = t + R[:, 2]

    arrow_x = visual.arrow(x=t[0] + R[0, 0], y=t[0] + R[0, 1], z=t[0] + R[0, 2])
    arrow_x.length_cone = 0.4
    arrow_y = visual.arrow(x=t[1] + R[1, 0], y=t[1] + R[1, 1], z=t[1] + R[1, 2])
    arrow_y.length_cone = 0.4
    arrow_z = visual.arrow(x=t[2] + R[2, 0], y=t[2] + R[2, 1], z=t[2] + R[2, 2])
    arrow_z.length_cone = 0.4

    arrow_length = arrow_len
    arrow_x.actor.scale = [arrow_length, arrow_length, arrow_length]
    arrow_x.pos = arrow_x.pos / arrow_length
    arrow_x.axis = [x2 - x1, y2 - y1, z2 - z1]
    fig.scene.add_actor(arrow_x)

    # x = Arrow3D([t[0], t[0] + R[0,0] * arrow_len], [t[1],t[1] + R[1,0] * arrow_len], [t[2],t[2] + R[2,0] * arrow_len], mutation_scale=10, lw=3, arrowstyle="-|>", color="r", alpha=0.7)
    # ax.add_artist(x)
    # y = Arrow3D([t[0], t[0] + R[0,1] * arrow_len], [t[1],t[1] + R[1,1] * arrow_len], [t[2],t[2] + R[2,1] * arrow_len], mutation_scale=10, lw=3, arrowstyle="-|>", color="g", alpha=0.7)
    # ax.add_artist(y)
    # z = Arrow3D([t[0], t[0] + R[0,2] * arrow_len], [t[1],t[1] + R[1,2] * arrow_len], [t[2],t[2] + R[2,2] * arrow_len], mutation_scale=10, lw=3, arrowstyle="-|>", color="b", alpha=0.7)
    # ax.add_artist(z)
    # ax.text(t[0] + R[0,0] * arrow_len,t[1] + R[1,0] * arrow_len,t[2] + R[2,0] * arrow_len, "x", color='r'); ax.text(t[0] + R[0,1] * arrow_len,t[1] + R[1,1] * arrow_len,t[2] + R[2,1] * arrow_len, "y", color='g'); ax.text(t[0] + R[0,2] * arrow_len,t[1] + R[1,2] * arrow_len,t[2] + R[2,2] * arrow_len, "z", color='b')


# def picker_callback(picker_obj):
#     """
#     http://docs.enthought.com/mayavi/mayavi/auto/example_pick_on_surface.html
#
#     fig.on_mouse_pick(picker_callback)
#     """
#     cursor3d = mlab.points3d(0., 0., 0., mode='axes',
#                                 color=(0, 0, 0),
#                                 scale_factor=0.5)
#     picked = picker_obj.actors
#     if mesh.actor.actor._vtk_obj in [o._vtk_obj for o in picked]:
#         # m.mlab_source.points is the points array underlying the vtk
#         # dataset. GetPointId return the index in this array.
#         x_, y_ = np.lib.index_tricks.unravel_index(picker_obj.point_id,
#                                                                 s.shape)
#         print("Data indices: %i, %i" % (x_, y_))
#         n_x, n_y = s.shape
#         cursor.mlab_source.reset(x=x_ - n_x/2.,
#                                y=y_ - n_y/2.)
#         cursor3d.mlab_source.reset(x=x[x_, y_],
#                                  y=y[x_, y_],
#                                  z=z[x_, y_])



def same_length_3D(ax):
    print('    ... same_length_3D')
    delta_list = [np.abs(ax.get_xlim()[1] - ax.get_xlim()[0]), np.abs(ax.get_ylim()[1] - ax.get_ylim()[0]),
                  np.abs(ax.get_zlim()[1] - ax.get_zlim()[0])]
    mean_list = [ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0]) / 2,
                 ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) / 2,
                 ax.get_zlim()[0] + (ax.get_zlim()[1] - ax.get_zlim()[0]) / 2]
    plus_minus = delta_list[int(np.argmax(delta_list))] * .6
    ax.set_xlim3d(mean_list[0] - plus_minus, mean_list[0] + plus_minus);
    ax.set_ylim3d(mean_list[1] - plus_minus, mean_list[1] + plus_minus);
    ax.set_zlim3d(mean_list[2] - plus_minus, mean_list[2] + plus_minus)


def plot_voxel(img_sitk, title='', factor=1, same_length='yes'):
    print('    ... plot_voxel')
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    if type(img_sitk) != np.ndarray:
        voxels = sitk.GetArrayFromImage(img_sitk).transpose(2, 1, 0)[::factor, ::factor, ::factor]
    else:
        voxels = img_sitk[::factor, ::factor, ::factor]
    ax.voxels(voxels, edgecolor='k')
    same_length_3D(ax)
    plt.title(title)
    if same_length.upper() == 'yes':
        same_length_3D(ax)


def plt_scatter(coordinates, factor=1):
    ax = m3d.Axes3D(plt.figure())
    ax.scatter(coordinates[::factor, 0],
               coordinates[::factor, 1],
               coordinates[::factor, 2], '^', c='black')
    same_length_3D(ax)


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


def plot_arrow_R_t(ax, R, t, arrow_len=0):
    print('    plot_arrow_R_t')
    x = Arrow3D([t[0], t[0] + R[0, 0] * arrow_len], [t[1], t[1] + R[1, 0] * arrow_len],
                [t[2], t[2] + R[2, 0] * arrow_len], mutation_scale=10, lw=3, arrowstyle="-|>", color="r", alpha=0.7)
    ax.add_artist(x)
    y = Arrow3D([t[0], t[0] + R[0, 1] * arrow_len], [t[1], t[1] + R[1, 1] * arrow_len],
                [t[2], t[2] + R[2, 1] * arrow_len], mutation_scale=10, lw=3, arrowstyle="-|>", color="g", alpha=0.7)
    ax.add_artist(y)
    z = Arrow3D([t[0], t[0] + R[0, 2] * arrow_len], [t[1], t[1] + R[1, 2] * arrow_len],
                [t[2], t[2] + R[2, 2] * arrow_len], mutation_scale=10, lw=3, arrowstyle="-|>", color="b", alpha=0.7)
    ax.add_artist(z)
    ax.text(t[0] + R[0, 0] * arrow_len, t[1] + R[1, 0] * arrow_len, t[2] + R[2, 0] * arrow_len, "x", color='r');
    ax.text(t[0] + R[0, 1] * arrow_len, t[1] + R[1, 1] * arrow_len, t[2] + R[2, 1] * arrow_len, "y", color='g');
    ax.text(t[0] + R[0, 2] * arrow_len, t[1] + R[1, 2] * arrow_len, t[2] + R[2, 2] * arrow_len, "z", color='b')


def plot_origin(ax, arrow_len=50):
    print('    plot_origin')
    R = np.eye(3)
    t = np.zeros(3)
    x = Arrow3D([t[0], t[0] + R[0, 0] * arrow_len], [t[1], t[1] + R[1, 0] * arrow_len],
                [t[2], t[2] + R[2, 0] * arrow_len], mutation_scale=10, lw=3, arrowstyle="-|>", color="r", alpha=0.7)
    ax.add_artist(x)
    y = Arrow3D([t[0], t[0] + R[0, 1] * arrow_len], [t[1], t[1] + R[1, 1] * arrow_len],
                [t[2], t[2] + R[2, 1] * arrow_len], mutation_scale=10, lw=3, arrowstyle="-|>", color="g", alpha=0.7)
    ax.add_artist(y)
    z = Arrow3D([t[0], t[0] + R[0, 2] * arrow_len], [t[1], t[1] + R[1, 2] * arrow_len],
                [t[2], t[2] + R[2, 2] * arrow_len], mutation_scale=10, lw=3, arrowstyle="-|>", color="b", alpha=0.7)
    ax.add_artist(z)
    ax.text(t[0] + R[0, 0] * arrow_len, t[1] + R[1, 0] * arrow_len, t[2] + R[2, 0] * arrow_len, "x", color='r');
    ax.text(t[0] + R[0, 1] * arrow_len, t[1] + R[1, 1] * arrow_len, t[2] + R[2, 1] * arrow_len, "y", color='g');
    ax.text(t[0] + R[0, 2] * arrow_len, t[1] + R[1, 2] * arrow_len, t[2] + R[2, 2] * arrow_len, "z", color='b')


def plot_arrow_3d(ax, v, t, arrow_len=50, color='r'):
    x = Arrow3D([t[0], t[0] + v[0] * arrow_len], [t[1], t[1] + v[1] * arrow_len], [t[2], t[2] + v[2] * arrow_len],
                mutation_scale=10, lw=3, arrowstyle="-|>", color=color, alpha=0.7)
    ax.add_artist(x)


def plot_ax():
    import mpl_toolkits.mplot3d as m3d
    return m3d.Axes3D(plt.figure())


def histo(img):
    print('    ... histo')
    if type(img) != np.ndarray:  # check type of img (numpy or simpleitk)
        pass
    else:
        img = sitk.GetImageFromArray(img)
    intensity_values = sitk.GetArrayViewFromImage(img).flatten()
    plt.figure()
    plt.hist(intensity_values, bins=500)
    n, bins, patches = plt.hist(intensity_values, bins=500)
    plt.ylim(n[np.argsort(n)[0]], n[np.argsort(n)[-2]] * 1.4)
    plt.show()


@jit
def downscale_kernel(size4func, factor, img_np_cropped, downscaled_np, type_img):
    cnt = 0
    for i in range(size4func[0] / factor):
        # print(' % s / %s' % (i, size4func[0]/factor))
        cnt += 1
        for j in range(size4func[1] / factor):
            for k in range(size4func[2] / factor):
                if type_img == 'binary':
                    downscaled_np[k, j, i] = int(np.rint(np.mean(
                        img_np_cropped[k * factor:k * factor + factor, j * factor:j * factor + factor,
                        i * factor:i * factor + factor])))
                else:
                    downscaled_np[k, j, i] = np.mean(
                        img_np_cropped[k * factor:k * factor + factor, j * factor:j * factor + factor,
                        i * factor:i * factor + factor])
    return downscaled_np


def downscale_numba(img, factor=2, type_img='grey'):
    print('    ... downscale_numba')
    if type(img) != np.ndarray:  # check type of img (numpy or simpleitk)
        img_sitk = img
    else:
        img_sitk = sitk.GetImageFromArray(img)
    size = img_sitk.GetSize()
    size4func = img_sitk.GetSize()
    resolution4func = img_sitk.GetSpacing()
    first_slice = np.array([0, 0, 0])
    last_slice = np.array([0, 0, 0])
    for i in range(3):  # look at dimensions divisible by the factor:
        dimension_scaled = int(np.int(size[i] / float(factor))) * factor
        first_slice[i] = int(np.rint(int(np.rint(size4func[i] / 2.)) - dimension_scaled / 2.))
        last_slice[i] = first_slice[i] + dimension_scaled
    img_sitk_cropped = img_sitk[first_slice[0]:last_slice[0], first_slice[1]:last_slice[1],
                       first_slice[2]:last_slice[2]]
    img_np_cropped = sitk.GetArrayFromImage(img_sitk_cropped)
    size4func = img_sitk_cropped.GetSize()
    del img_sitk_cropped
    downscaled_np = np.zeros((int(size4func[2] / factor), int(size4func[1] / factor), int(size4func[0] / factor)))
    print('before numba')
    downscaled_np = downscale_kernel(size4func, factor, img_np_cropped, downscaled_np, type_img)
    print('numba finished')
    downscaled_sitk = sitk.GetImageFromArray(downscaled_np)
    print('downscaled_sitk finished')
    del downscaled_np
    downscaled_sitk.SetSpacing((resolution4func[0] * factor, resolution4func[1] * factor, resolution4func[2] * factor))
    return downscaled_sitk


def downscale_sitk(img_target, resolution_target):
    if isinstance(resolution_target, tuple):
        ratio = resolution_target[0] / img_target.GetSpacing()[0]
    else:
        ratio = resolution_target / img_target.GetSpacing()[0]
        resolution_target = (resolution_target, resolution_target, resolution_target)
    new_size = np.rint(
        [img_target.GetSize()[0] / ratio, img_target.GetSize()[1] / ratio, img_target.GetSize()[2] / ratio]).astype(
        'int').tolist()
    reference_properties = sitk.Image(new_size, img_target.GetPixelIDValue())
    reference_properties.SetOrigin(img_target.GetOrigin())
    reference_properties.SetDirection(img_target.GetDirection())
    reference_properties.SetSpacing(resolution_target)
    return sitk.Resample(img_target, reference_properties, sitk.AffineTransform(3), sitk.sitkNearestNeighbor, 0)


def rescale_isotropic(img_sitk, resolution='max_spacing', type_interpolation='linear'):
    print('    ... rescale_isotropic')
    #        - if the spacing (resolution) is x = 0.488, y = 0.488, z = 0.400, the isotropic max spaxing will be [0.48, 0.488, 0.488]
    #        - if the image is in grayscale, take a linear interpolation (stik.sitklinear) otherwise, with a binary image, take sitkNearestNeighbor
    original_spacing = img_sitk.GetSpacing()
    if original_spacing[0] == original_spacing[1] and original_spacing[1] == original_spacing[2]:
        print('    the image is already isotropic')
        return img_sitk
    else:
        original_size = img_sitk.GetSize()
        min_spacing = min(img_sitk.GetSpacing())
        max_spacing = max(img_sitk.GetSpacing())
        if resolution == 'max_spacing':
            new_spacing = [max_spacing, max_spacing, max_spacing]
            new_size = [int(round(original_size[0] * (original_spacing[0] / max_spacing))),
                        int(round(original_size[1] * (original_spacing[1] / max_spacing))),
                        int(round(original_size[2] * (original_spacing[2] / max_spacing)))]
        else:
            new_spacing = [min_spacing, min_spacing, min_spacing]
            new_size = [int(round(original_size[0] * (original_spacing[0] / min_spacing))),
                        int(round(original_size[1] * (original_spacing[1] / min_spacing))),
                        int(round(original_size[2] * (original_spacing[2] / min_spacing)))]
        resampleSliceFilter = sitk.ResampleImageFilter()
        if type_interpolation == 'linear':
            sitk_isotropic = resampleSliceFilter.Execute(img_sitk, new_size, sitk.Transform(), sitk.sitkLinear,
                                                         img_sitk.GetOrigin(),
                                                         new_spacing, img_sitk.GetDirection(), 0, img_sitk.GetPixelID())
        else:
            sitk_isotropic = resampleSliceFilter.Execute(img_sitk, new_size, sitk.Transform(), sitk.sitkNearestNeighbor,
                                                         img_sitk.GetOrigin(),
                                                         new_spacing, img_sitk.GetDirection(), 0, img_sitk.GetPixelID())
        return sitk_isotropic


def rotate(img, angles, ang_type='rad', center_rotation_x=0.5, center_rotation_y=0.5, center_rotation_z=0.5):
    print('    ... rotate')
    ang_x, ang_y, ang_z = angles
    if type(img) != np.ndarray:
        pass
    else:
        img = sitk.GetImageFromArray(img.transpose(2, 1, 0))
        print('         !! be careful: resolution lost because input = np image !!')

    def Rot_matrix(ang_x, ang_y, ang_z):
        def trig(angle):
            return np.cos(angle), np.sin(angle)

        xC, xS = trig(ang_x)
        yC, yS = trig(ang_y)
        zC, zS = trig(ang_z)
        return [yC * xC, -zC * xS + zS * yS * xC, zS * xS + zC * yS * xC, yC * xS, zC * xC + zS * yS * xS,
                -zS * xC + zC * yS * xS, -yS, zS * yC, zC * yC]

    if ang_type != 'rad':
        ang_x = np.deg2rad(ang_x)
        ang_y = np.deg2rad(ang_y)
        ang_z = np.deg2rad(ang_z)
    dimension = 3
    affine = sitk.AffineTransform(dimension)
    affine.SetMatrix(Rot_matrix(ang_x, ang_y, ang_z))
    affine.SetCenter((center_rotation_x * img.GetSpacing()[0] * img.GetSize()[0],
                      center_rotation_y * img.GetSpacing()[1] * img.GetSize()[1],
                      center_rotation_z * img.GetSpacing()[2] * img.GetSize()[2]))
    value_outside_translated_image = 0
    affine_rotate = sitk.Resample(img, img, affine, sitk.sitkLinear, value_outside_translated_image)
    affine_rotate.SetSpacing(img.GetSpacing())
    return affine_rotate


def rotate_R(img, R, t=(0, 0, 0), center_rotation_x=0.5, center_rotation_y=0.5, center_rotation_z=0.5,
             empty_layers=False):
    """
    rotate with a rotation matrix using SimpleItk the image
    :param img: sitk or numpy array
    :param R:
    :param t:
    :param center_rotation_x: default is the center (0.5)
    :param center_rotation_y: default is the center (0.5)
    :param center_rotation_z: default is the center (0.5)
    :param empty_layers: add empty layers to not loose image parts out of the original dimensions
    :return:
    """
    print('    ... rotate_R')
    if empty_layers:
        if type(img) != np.ndarray:
            img_np = sitk.GetArrayFromImage(img).transpose(2, 1, 0)
            img_sitk = img
        else:
            img_np = img
            img_sitk = sitk.GetImageFromArray(img.transpose(2, 1, 0))
            print('         !! be careful: resolution lost because input = np image !!')
        img_bigger_np = np.zeros(())
        dim_x, dim_y, dim_z = int(img_sitk.GetSize()[0] * 1.1), int(img_sitk.GetSize()[1] * 1.1), int(
            img_sitk.GetSize()[2] * 1.1)
        dimension_moving = img_sitk.GetSize()
        img_bigger_np = np.zeros((dim_x, dim_y, dim_z))
        img_bigger_np[
        int(dim_x / 2 - dimension_moving[0] / 2):int(dim_x / 2 - dimension_moving[0] / 2) + dimension_moving[0],
        int(dim_y / 2 - dimension_moving[1] / 2):int(dim_y / 2 - dimension_moving[1] / 2) + dimension_moving[1],
        int(dim_z / 2 - dimension_moving[2] / 2):int(dim_z / 2 - dimension_moving[2] / 2) + dimension_moving[
            2]] = img_np
        img_bigger_sitk = sitk.GetImageFromArray(img_bigger_np.transpose(2, 1, 0))
    else:
        img_bigger_sitk = img
    dimension = 3
    affine = sitk.AffineTransform(dimension)
    affine.SetMatrix(R.reshape(9).tolist())
    affine.SetCenter((center_rotation_x * img_bigger_sitk.GetSpacing()[0] * img_bigger_sitk.GetSize()[0],
                      center_rotation_y * img_bigger_sitk.GetSpacing()[1] * img_bigger_sitk.GetSize()[1],
                      center_rotation_z * img_bigger_sitk.GetSpacing()[2] * img_bigger_sitk.GetSize()[2]))
    affine.SetOffset(t)
    value_outside_translated_image = 0
    affine_rotate = sitk.Resample(img_bigger_sitk, img_bigger_sitk, affine, sitk.sitkLinear,
                                  value_outside_translated_image)
    affine_rotate.SetSpacing(img.GetSpacing())
    #    affine_rotate = sitk.Cast(affine_rotate, sitk.sitkFloat32)
    affine_rotate = bounding_box(affine_rotate)
    affine_rotate.SetOrigin((0, 0, 0))
    return affine_rotate


@jit
def kernel_rotate_R_personal(img_bigger_coor, new_coord, R, center_rotation_x, img_bigger_np_shape):
    for i in range(len(img_bigger_coor)):
        new_coord[i, :] = np.dot(R, img_bigger_coor[i, :]) + center_rotation_x * np.array(img_bigger_np_shape)
    return new_coord


def rotate_R_personal(img, R, t, center_rotation_x=0.5, center_rotation_y=0.5, center_rotation_z=0.5,
                      empty_layers=False):
    import psutil
    print('    ... rotate_R_personal')
    #    if empty_layers:
    if type(img) != np.ndarray:
        img_np = sitk.GetArrayFromImage(img).transpose(2, 1, 0)
        img_sitk = img
    else:
        img_np = img
        img_sitk = sitk.GetImageFromArray(img.transpose(2, 1, 0))
        print('         !! be careful: resolution lost because input = np image !!')
    print('memory in % 1: ', psutil.virtual_memory()[2])
    dim_x, dim_y, dim_z = int(img_sitk.GetSize()[0] * 1.5), int(img_sitk.GetSize()[1] * 1.5), int(
        img_sitk.GetSize()[2] * 1.5)
    dimension_moving = img_sitk.GetSize()
    img_bigger_np = np.zeros((dim_x, dim_y, dim_z))
    print('memory in % 2: ', psutil.virtual_memory()[2])
    img_bigger_np[
    int(dim_x / 2 - dimension_moving[0] / 2):int(dim_x / 2 - dimension_moving[0] / 2) + dimension_moving[0],
    int(dim_y / 2 - dimension_moving[1] / 2):int(dim_y / 2 - dimension_moving[1] / 2) + dimension_moving[1],
    int(dim_z / 2 - dimension_moving[2] / 2):int(dim_z / 2 - dimension_moving[2] / 2) + dimension_moving[2]] = img_np
    print('memory in % 3: ', psutil.virtual_memory()[2])
    #    plot_stack(img_bigger_np, 'img_bigger_np')
    img_bigger_coor = array_3d_to_coordinate(img_bigger_np, value=1)
    print('memory in % 5: ', psutil.virtual_memory()[2])
    img_bigger_np_shape = img_bigger_np.shape
    img_bigger_coor -= np.rint(0.5 * np.array(img_bigger_np_shape)).astype(int)
    new_coord = np.zeros((len(img_bigger_coor), 3))
    print('memory in % 6: ', psutil.virtual_memory()[2])
    print('memory in %     ... - start rotation')
    #    new_coord = kernel_rotate_R_personal(img_bigger_coor, new_coord, R, center_rotation_x, img_bigger_np_shape)
    new_coord = np.dot(R, img_bigger_coor.T).T + t
    print('memory in % 7: ', psutil.virtual_memory()[2])
    new_coord += center_rotation_x * .999 * np.array(img_bigger_np_shape)
    print('memory in % 8: ', psutil.virtual_memory()[2])
    #    for i in range(len(img_bigger_coor)):
    #        new_coord[i, :] = np.dot(R, img_bigger_coor[i, :]) + center_rotation_x * np.array(img_bigger_np.shape)
    print('memory in %     ... - end rotation')
    array_rotated_np = coordinate_to_array_3d(new_coord, dim='max_of_coordinate', value=1)
    print('memory in % 9: ', psutil.virtual_memory()[2])
    array_rotated_sitk = sitk.GetImageFromArray(array_rotated_np.transpose(2, 1, 0))
    #    plot_stack(array_rotated_np, 'array_rotated_np')
    #    array_rotated_sitk_closed = dilate_erode_ndimage(array_rotated_np, iterations=2)
    print('memory in % 10: ', psutil.virtual_memory()[2])
    del array_rotated_np
    #    print('memory in % 11: ', psutil.virtual_memory()[2]
    #    array_rotated_sitk = sitk.GetImageFromArray(array_rotated_np_closed.transpose(2, 1, 0))
    print('memory in % 11: ', psutil.virtual_memory()[2])
    array_rotated_sitk.SetSpacing(img_sitk.GetSpacing())
    del new_coord
    del img_bigger_coor
    del img_bigger_np
    del img
    print('memory in % 12: ', psutil.virtual_memory()[2])
    return array_rotated_sitk


# # img_bigger_sitk = sitk.GetImageFromArray(img_bigger_np.transpose(2, 1, 0))
# # else:
# #     img_bigger_sitk = img
# # img_bigger_np = sitk.GetArrayFromImage(img_bigger_sitk)
# dimension = 3
# affine = sitk.AffineTransform(dimension)
# affine.SetMatrix(R.reshape(9).tolist())
# affine.SetCenter((center_rotation_x * img_bigger_sitk.GetSpacing()[0] * img_bigger_sitk.GetSize()[0],
#                     center_rotation_y * img_bigger_sitk.GetSpacing()[1] * img_bigger_sitk.GetSize()[1],
#                     center_rotation_z * img_bigger_sitk.GetSpacing()[2] * img_bigger_sitk.GetSize()[2]))
# value_outside_translated_image = 0
# affine_rotate = sitk.Resample(img_bigger_sitk, img_bigger_sitk, affine, sitk.sitkLinear, value_outside_translated_image)
# affine_rotate.SetSpacing(img.GetSpacing())
# # affine_rotate = sitk.Cast(affine_rotate, sitk.sitkFloat32)
# # affine_rotate = bounding_box(affine_rotate)
# # affine_rotate.SetOrigin((0, 0, 0))
# return affine_rotate


def center_of_gravity(img, resolution=np.array([1, 1, 1])):
    import scipy.ndimage as ndimage
    if type(img) == np.ndarray:
        pass
    else:
        img = sitk.GetArrayFromImage(img).transpose(2, 1, 0)
    CG_temp = ndimage.measurements.center_of_mass(img)
    return [CG_temp[0] * resolution[0], CG_temp[1] * resolution[1], CG_temp[2] * resolution[2]]


def add_layers(img, add_layer):
    if type(img) != np.ndarray:
        img_np = sitk.GetArrayFromImage(img).transpose(2, 1, 0)
        img_sitk = img
    else:
        img_np = img
        img_sitk = sitk.GetImageFromArray(img.transpose(2, 1, 0))
        print('         !! be careful: resolution lost because input = np image !!')
    if np.array(add_layer).shape == ():
        add_layer = [add_layer] * 3
    dim_x = int(img_sitk.GetSize()[0] * add_layer[0])
    dim_y = int(img_sitk.GetSize()[1] * add_layer[1])
    dim_z = int(img_sitk.GetSize()[2] * add_layer[2])
    print('dim_x', dim_x)
    dimension_moving = img_sitk.GetSize()
    del img_sitk
    img_bigger_np = np.zeros((dim_x, dim_y, dim_z))
    img_bigger_np[
    int(dim_x / 2 - dimension_moving[0] / 2):int(dim_x / 2 - dimension_moving[0] / 2) + dimension_moving[0],
    int(dim_y / 2 - dimension_moving[1] / 2):int(dim_y / 2 - dimension_moving[1] / 2) + dimension_moving[1],
    int(dim_z / 2 - dimension_moving[2] / 2):int(dim_z / 2 - dimension_moving[2] / 2) + dimension_moving[2]] = img_np
    del img_np
    img_bigger_sitk = sitk.GetImageFromArray(img_bigger_np.transpose(2, 1, 0))
    try:
        img_bigger_sitk.SetSpacing(img.GetSpacing())
    except:
        pass
    return img_bigger_sitk


def add_layers_nb(img, add_layer):
    if type(img) != np.ndarray:
        img_np = sitk.GetArrayFromImage(img).transpose(2, 1, 0)
        img_sitk = img
    else:
        img_np = img
        img_sitk = sitk.GetImageFromArray(img.transpose(2, 1, 0))
    if np.array(add_layer).shape == ():
        add_layer = [add_layer] * 3
    dim_x = int(img_sitk.GetSize()[0] + 2 * add_layer[0])
    dim_y = int(img_sitk.GetSize()[1] + 2 * add_layer[1])
    dim_z = int(img_sitk.GetSize()[2] + 2 * add_layer[2])
    print('dim_x', dim_x)
    dimension_moving = img_sitk.GetSize()
    del img_sitk
    img_bigger_np = np.zeros((dim_x, dim_y, dim_z))
    img_bigger_np[
    int(dim_x / 2 - dimension_moving[0] / 2):int(dim_x / 2 - dimension_moving[0] / 2) + dimension_moving[0],
    int(dim_y / 2 - dimension_moving[1] / 2):int(dim_y / 2 - dimension_moving[1] / 2) + dimension_moving[1],
    int(dim_z / 2 - dimension_moving[2] / 2):int(dim_z / 2 - dimension_moving[2] / 2) + dimension_moving[2]] = img_np
    del img_np
    img_bigger_sitk = sitk.GetImageFromArray(img_bigger_np.transpose(2, 1, 0))
    try:
        img_bigger_sitk.SetSpacing(img.GetSpacing())
        return img_bigger_sitk
    except:
        return img_bigger_np


def add_layers_nb_np(img_np, add_layer):
    if np.array(add_layer).shape == ():
        add_layer = [add_layer] * 3
    dim_x = int(img_np.shape[0] + 2 * add_layer[0])
    dim_y = int(img_np.shape[1] + 2 * add_layer[1])
    dim_z = int(img_np.shape[2] + 2 * add_layer[2])
    dimension_moving = img_np.shape
    img_bigger_np = np.zeros((dim_x, dim_y, dim_z))
    img_bigger_np[
    int(dim_x / 2 - dimension_moving[0] / 2):int(dim_x / 2 - dimension_moving[0] / 2) + dimension_moving[0],
    int(dim_y / 2 - dimension_moving[1] / 2):int(dim_y / 2 - dimension_moving[1] / 2) + dimension_moving[1],
    int(dim_z / 2 - dimension_moving[2] / 2):int(dim_z / 2 - dimension_moving[2] / 2) + dimension_moving[2]] = img_np
    del img_np
    return img_bigger_np


def transform_R_t(img_sitk, R, t, center='gravity', add_layer=[1.2, 1.2, 1.2], interpo=sitk.sitkLinear, bbcut=False):
    print('c', img_min_max(img_sitk))
    print('    ... transform_R_t')
    mini, maxi = img_min_max(img_sitk)
    print('0 max = %s' % (maxi))
    img_sitk = add_layers(img_sitk, add_layer)
    mini, maxi = img_min_max(img_sitk)
    print('1 max = %s' % (maxi))
    affine = sitk.AffineTransform(3)
    affine.SetMatrix(R.ravel())
    if center == 'gravity':
        COG_voxel = np.array(center_of_gravity(img_sitk))
        COG_mm = (img_sitk.GetSpacing()[0] * COG_voxel[0], img_sitk.GetSpacing()[1] * COG_voxel[1],
                  img_sitk.GetSpacing()[2] * COG_voxel[2])
        affine.SetCenter(COG_mm)
        print('    ... center of rotation %s is: %s' % (center, COG_voxel))
    if center == 'center':
        center_image = (np.rint(np.array(img_sitk.GetSize())[0] / 2).astype('int'),
                        np.rint(np.array(img_sitk.GetSize())[1] / 2).astype('int'),
                        np.rint(np.array(img_sitk.GetSize())[2] / 2).astype('int'))
        center_image_mm = np.array(center_image) * img_sitk.GetSpacing()[0]
        affine.SetCenter(center_image_mm)
        print('    ... center of rotation %s is: %s' % (center, center_image))
    if not isinstance(center, string_types):
        affine.SetCenter(center)

    resampled_temp = sitk.Resample(img_sitk, img_sitk, affine, interpo, 0)
    #    Swissknife.plot_stack(resampled_temp, 'resampled_temp')
    mini, maxi = img_min_max(resampled_temp)
    print('2 max = %s' % (maxi))
    affine = sitk.AffineTransform(3)
    affine.SetTranslation((-1 * np.array(t)).tolist())
    resampled = sitk.Resample(resampled_temp, resampled_temp, affine, interpo, 0)
    del resampled_temp
    if bbcut:
        resampled = bounding_box(resampled)
    resampled.SetSpacing(img_sitk.GetSpacing())
    resampled = sitk.Cast(resampled, sitk.sitkFloat32)
    mini, maxi = img_min_max(resampled)
    print('3 max = %s' % (maxi))
    return resampled


def gaussian(img_sitk, sigma):
    print('    ... Gaussian (sigma = ) filtering' % sigma)
    gaussian = sitk.SmoothingRecursiveGaussianImageFilter()
    gaussian.SetSigma(sigma)
    return gaussian.Execute(img_sitk)


def segmentation_otsu(img):
    print('    ... segmentation_otsu')
    if type(img) != np.ndarray:
        pass
    else:
        img = sitk.GetImageFromArray(img.transpose(2, 1, 0))
    otsu_filter = sitk.OtsuThresholdImageFilter()
    otsu_filter.SetInsideValue(0)
    otsu_filter.SetOutsideValue(1)
    seg = otsu_filter.Execute(img)
    seg = otsu_filter.Execute(img_same_properties(img, seg) * img)
    print('otsu threshold = ', otsu_filter.GetThreshold())
    return seg


def segmentation_overview(img, lower_threshold=5, upper_threshold=800):
    print('    ... segmentation_overview')
    if type(img) != np.ndarray:
        pass
    else:
        img = sitk.GetImageFromArray(img.transpose(2, 1, 0))
        print('         !! be careful: resolution lost because input = np image !!')
    resolution = img.GetSize()
    j = 0
    figZ, axZ = plt.subplots(nrows=4, ncols=4, sharex=True, sharey=True, figsize=(8, 8))
    figX, axX = plt.subplots(nrows=4, ncols=4, sharex=True, sharey=True, figsize=(8, 8))
    axZ = axZ.ravel()
    axX = axX.ravel()
    for i in np.rint(np.linspace(lower_threshold, upper_threshold, 16)):
        upperThreshold = i
        print('threshold: ', upperThreshold)
        seg_threshold_sitk = sitk.BinaryThreshold(img, lowerThreshold=float(np.min(sitk.GetArrayFromImage(img))),
                                                  upperThreshold=upperThreshold, insideValue=0, outsideValue=1)
        size = seg_threshold_sitk.GetSize()
        imgZ_sitk = seg_threshold_sitk[:, :, int(size[2] / 2)]
        imgX_sitk = seg_threshold_sitk[int(size[0] / 2), :, :]
        imageZ = sitk.GetArrayFromImage(imgZ_sitk)
        imageX = sitk.GetArrayFromImage(imgX_sitk)
        extentZ = (0, imageZ.shape[1] * resolution[1], imageZ.shape[0] * resolution[0], 0)
        extentX = (0, imageX.shape[1] * resolution[1], imageX.shape[0] * resolution[0], 0)
        axZ[j].imshow(imageZ, extent=extentZ, interpolation=None)
        axZ[j].set_title(' % s' % int(upperThreshold), fontsize=7)
        axZ[j].set_axis_off()
        axX[j].imshow(imageX, extent=extentX, interpolation=None)
        axX[j].set_title(' % s' % int(upperThreshold), fontsize=7)
        axX[j].set_axis_off()
        j += 1
    plt.tight_layout()
    print('    Please choose an appropriate threshold on the image an then close the image, thank you\n')
    figZ.savefig('./temp_figZ_seg.png', dpi=512, pad_inches=0)
    figX.savefig('./temp_figX_seg.png', dpi=512, pad_inches=0)
    if socket.gethostname() == 'ben-PC':
        os.system('gwenview temp_figZ_seg.png')
        os.system('gwenview temp_figX_seg.png')
    else:
        os.system('xdg-open temp_figZ_seg.png')
        os.system('xdg-open temp_figX_seg.png')
    # upperThreshold = int(raw_input("\nWhich threshold is correct? "))  # python 2.7
    upperThreshold = float(input("\nWhich threshold is correct? "))  # python 3
    seg_threshold = sitk.BinaryThreshold(img, lowerThreshold=float(np.min(sitk.GetArrayFromImage(img))),
                                         upperThreshold=upperThreshold, insideValue=0, outsideValue=1)
    os.system('rm temp_figZ_seg.png')
    os.system('rm temp_figX_seg.png')
    return seg_threshold


def remove_small_object_old(img):
    print('    ... remove_small_object')
    if type(img) != np.ndarray:
        img_sitk = sitk.Cast(img, sitk.sitkInt32)
    else:
        img_sitk = sitk.Cast(sitk.GetImageFromArray(img.transpose(2, 1, 0)), sitk.sitkInt32)
        print('         !! be careful: resolution lost because input = np image !!')
    reso = img_sitk.GetSpacing()
    img_label_sitk = sitk.ConnectedComponent(img_sitk)
    img_label_np = sitk.GetArrayFromImage(img_label_sitk)  # .transpose(2, 1, 0)
    unique, count = np.unique(img_label_np, return_counts=True)
    print('     count:  ', np.sort(count)[::-1][:7])
    for i in np.arange(2) + 1:
        if np.argsort(count)[-i] != 0:
            label_to_keep = np.argsort(count)[-i]
    img_label_np[img_label_np != label_to_keep] = 0
    img_label_np[img_label_np != 0] = 1
    img_label_sitk = sitk.GetImageFromArray(img_label_np)
    img_label_sitk.SetSpacing(reso)
    return img_label_sitk


def remove_small_object(img):
    print('    ... remove_small_object')
    if type(img) != np.ndarray:
        img_sitk = sitk.Cast(img, sitk.sitkInt32)
    else:
        img_sitk = sitk.Cast(sitk.GetImageFromArray(img.transpose(2, 1, 0)), sitk.sitkInt32)
        print('         !! be careful: resolution lost because input = np image !!')
    reso = img_sitk.GetSpacing()
    label_sitk = sitk.ConnectedComponent(img_sitk)
    del img_sitk
    label_np = sitk.GetArrayFromImage(label_sitk).transpose(2, 1, 0)
    del label_sitk
    unique, count = np.unique(label_np, return_counts=True)
    img_sitk = sitk.GetImageFromArray(np.array(label_np == np.argsort(count)[-2]).astype(int).transpose(2, 1, 0))
    img_sitk.SetSpacing(reso)
    return img_sitk


def cv2_loop(inputs):
    seg_np = inputs[0]
    radius = inputs[1]
    dilated_eroded_z = np.zeros(seg_np.shape)
    kernel_2d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius))
    for i in np.arange(seg_np.shape[2]):
        dilated_eroded_z[:, :, i] = cv2.morphologyEx(seg_np[:, :, i], cv2.MORPH_CLOSE, kernel_2d)
    return dilated_eroded_z


def masking_parallel(seg_np, radius=60, nb_cpu=12):
    if type(seg_np) != np.ndarray:
        reso = seg_np.GetSpacing()
        seg_np = sitk.GetArrayFromImage(seg_np).transpose(2, 1, 0)
    else:
        pass
    # kernel_2d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(radius, radius))
    # dilated_eroded_z = np.zeros(seg_np.shape)
    slices_to_process = np.array_split(seg_np, nb_cpu, axis=2)
    radii = [radius] * nb_cpu
    slices_to_process = [x for x in zip(slices_to_process, radii)]
    pool = Pool(processes=nb_cpu)
    result = pool.map(cv2_loop, slices_to_process)
    dilated_eroded_z = np.concatenate(result, axis=2) * (-1) + 1
    img_sitk = sitk.Cast(sitk.GetImageFromArray(dilated_eroded_z.transpose(2, 1, 0)), sitk.sitkInt32)
    # compute connected component:
    label_sitk = sitk.ConnectedComponent(img_sitk)
    # # transform it into a numpy array:
    label_np = sitk.GetArrayFromImage(label_sitk).transpose(2, 1, 0)
    # count the labels:
    unique, count = np.unique(label_np, return_counts=True)
    # sort the label index and take the second higher:
    label_to_keep = np.argsort(count).astype(int)[-1]
    # take the label which are equal to the second higher label
    mask_np = np.array(label_np == label_to_keep) * (-1) + 1
    # return sitk.GetImageFromArray(mask_np.transpose(2, 1, 0))
    return mask_np


def cv2_dilate_loop(inputs):
    seg_np = inputs[0].astype(np.uint8)
    radius = inputs[1]
    dilated_z = np.zeros(seg_np.shape).astype(np.uint8)
    kernel_2d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius))
    for i in np.arange(seg_np.shape[2]):
        dilated_z[:, :, i] = cv2.dilate(seg_np[:, :, i], kernel_2d)
    return dilated_z


def dilate_cv2(seg_np, radius=60, nb_cpu=12):
    print('    ... start dilation with cv2 with %s CPUs' % nb_cpu)
    if type(seg_np) != np.ndarray:
        reso = seg_np.GetSpacing()
        seg_np = sitk.GetArrayFromImage(seg_np).transpose(2, 1, 0)
    else:
        pass
    slices_to_process = np.array_split(seg_np, nb_cpu, axis=2)
    radii = [radius] * nb_cpu
    slices_to_process = [x for x in zip(slices_to_process, radii)]
    pool = Pool(processes=nb_cpu)
    result = pool.map(cv2_dilate_loop, slices_to_process)
    dilated_z = np.concatenate(result, axis=2) * (-1) + 1
    img_sitk = sitk.Cast(sitk.GetImageFromArray(dilated_z.transpose(2, 1, 0)), sitk.sitkInt32)
    # compute connected component:
    label_sitk = sitk.ConnectedComponent(img_sitk)
    # # transform it into a numpy array:
    label_np = sitk.GetArrayFromImage(label_sitk).transpose(2, 1, 0)
    # count the labels:
    unique, count = np.unique(label_np, return_counts=True)
    # sort the label index and take the second higher:
    label_to_keep = np.argsort(count).astype(int)[-1]
    # take the label which are equal to the second higher label
    mask_np = np.array(label_np == label_to_keep) * (-1) + 1
    # return sitk.GetImageFromArray(mask_np.transpose(2, 1, 0))
    return mask_np


def get_mask_from_seg(seg_np, radius=25):
    if type(seg_np) != np.ndarray:
        reso = seg_np.GetSpacing()
        seg_np = sitk.GetArrayFromImage(seg_np).transpose(2, 1, 0)
    else:
        pass
    kernel_2d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius))
    dilated_eroded_z = np.zeros(seg_np.shape)
    for i in np.arange(seg_np.shape[2]):
        dilated_eroded_z[:, :, i] = cv2.morphologyEx(seg_np[:, :, i], cv2.MORPH_CLOSE, kernel_2d)
    dilated_eroded_z = dilated_eroded_z * (-1) + 1
    img_sitk = sitk.Cast(sitk.GetImageFromArray(dilated_eroded_z.transpose(2, 1, 0)), sitk.sitkInt32)
    # compute connected component:
    label_sitk = sitk.ConnectedComponent(img_sitk)
    # # transform it into a numpy array:
    label_np = sitk.GetArrayFromImage(label_sitk).transpose(2, 1, 0)
    # count the labels:
    unique, count = np.unique(label_np, return_counts=True)
    # sort the label index and take the second higher:
    label_to_keep = np.argsort(count).astype(int)[-1]
    # take the label which are equal to the second higher label
    mask_np = np.array(label_np == label_to_keep) * (-1) + 1
    return sitk.GetImageFromArray(mask_np.transpose(2, 1, 0))


def smooth_binary(img, sigma=2.0, threshold=0.95):
    """
    Smooth the binary image with sigma (sigma=2.0), takes the binaries with a threshold (0.95),
    and dilate (radisu = sigma + 1) to recover the initial "thickness".

    :param img: numpy or sitk
    :param sigma: smoothing factor
    :param threshold: binary threshold after smoothing
    :return:
    """
    if type(img) != np.ndarray:
        img_sitk = sitk.Cast(img, sitk.sitkInt32)
    else:
        img_sitk = sitk.Cast(sitk.GetImageFromArray(img.transpose(2, 1, 0)), sitk.sitkInt32)
    del img

    smoothing_filter = sitk.SmoothingRecursiveGaussianImageFilter()
    smoothing_filter.SetSigma(sigma)
    smoothed_sitk = smoothing_filter.Execute(img_sitk)
    smoothed_binary_sitk = img_same_properties(img_sitk, smoothed_sitk > threshold)
    smoothed_binary_sitk = dilate(smoothed_binary_sitk, int(sigma * 1.7 + 1))
    return smoothed_binary_sitk


def bounding_box(img):
    print('    ... creating bounding_box')
    if type(img) != np.ndarray:
        img_sitk = img
    else:
        img_sitk = sitk.GetImageFromArray(img.transpose(2, 1, 0))
    reso = img_sitk.GetSpacing()
    img_sitk = sitk.Cast(img_sitk, sitk.sitkInt16)
    img_description(img_sitk)
    label_shape_analysis = sitk.LabelShapeStatisticsImageFilter()
    label_shape_analysis.Execute(img_sitk)
    bounding_box = label_shape_analysis.GetBoundingBox(1)
    img_sitk = img_sitk[bounding_box[0]:bounding_box[0] + bounding_box[3],
               bounding_box[1]:bounding_box[1] + bounding_box[4], bounding_box[2]:bounding_box[2] + bounding_box[5]]
    img_sitk.SetSpacing(reso)
    return img_sitk


def bounding_box_start_end(img):
    print('    ... creating bounding_box')
    if type(img) != np.ndarray:
        img_sitk = img
    else:
        img_sitk = sitk.GetImageFromArray(img.transpose(2, 1, 0))
    reso = img_sitk.GetSpacing()
    img_sitk = sitk.Cast(img_sitk, sitk.sitkInt16)
    img_description(img_sitk)
    label_shape_analysis = sitk.LabelShapeStatisticsImageFilter()
    label_shape_analysis.Execute(img_sitk)
    bounding_box = label_shape_analysis.GetBoundingBox(1)
    return bounding_box[0], bounding_box[0] + bounding_box[3], bounding_box[1], bounding_box[1] + bounding_box[4], \
           bounding_box[2], bounding_box[2] + bounding_box[5]


def sphere_array_old(dimensions, radius, center, resolution=(1, 1, 1)):
    sphere_np = np.zeros((dimensions)).astype('uint8')
    cz, cy, cx = center[0], center[1], center[2]
    z, y, x = np.ogrid[-radius: radius, -radius: radius, -radius: radius]
    index = x ** 2 + y ** 2 + z ** 2 <= radius ** 2
    sphere_np[cz - radius:cz + radius, cy - radius:cy + radius, cx - radius:cx + radius][index] = 1
    sphere_sitk = sitk.GetImageFromArray(sphere_np.transpose(2, 1, 0))
    sphere_sitk.SetSpacing(resolution)
    return sphere_sitk


def sphere_array(shape, radius, position):
    print('    ... creating sphere_array')
    semisizes = (float(radius),) * 3
    grid = [slice(-x0, dim - x0) for x0, dim in zip(position, shape)]
    position = np.ogrid[grid]
    arr = np.zeros(np.asarray(shape).astype(int), dtype=float)
    for x_i, semisize in zip(position, semisizes):
        arr += (np.abs(x_i / semisize) ** 2)
    return (arr <= 1.0).astype('int')


def sphere_itk(shape, radius, position, resolution=(1, 1, 1)):
    print('    ... creating sphere_itk')
    semisizes = (float(radius),) * 3
    grid = [slice(-x0, dim - x0) for x0, dim in zip(position, shape)]
    position = np.ogrid[grid]
    arr = np.zeros(np.asarray(shape).astype(int), dtype=float)
    for x_i, semisize in zip(position, semisizes):
        arr += (np.abs(x_i / semisize) ** 2)
    sphere_np = arr <= 1.0
    sphere_sitk = sitk.GetImageFromArray(sphere_np.astype(int).transpose(2, 1, 0))
    sphere_sitk.SetSpacing(resolution)
    return sitk.Cast(sphere_sitk, sitk.sitkInt16)


def hollow_sphere_array(shape, outer_radius, inner_radius, center, resolution=(1, 1, 1)):
    print('    ... creating hollow_sphere_array')
    semisizes = (float(outer_radius),) * 3
    grid = [slice(-x0, dim - x0) for x0, dim in zip(center, shape)]
    position = np.ogrid[grid]
    arr = np.zeros(np.asarray(shape).astype(int), dtype=float)
    for x_i, semisize in zip(position, semisizes):
        arr += (np.abs(x_i / semisize) ** 2)
    sphere_np = arr <= 1.0
    sphere_out = sphere_np.astype(int).transpose(2, 1, 0)
    semisizes = (float(inner_radius),) * 3
    grid = [slice(-x0, dim - x0) for x0, dim in zip(center, shape)]
    position = np.ogrid[grid]
    arr = np.zeros(np.asarray(shape).astype(int), dtype=float)
    for x_i, semisize in zip(position, semisizes):
        arr += (np.abs(x_i / semisize) ** 2)
    sphere_np = arr <= 1.0
    sphere_inner = sphere_np.astype(int).transpose(2, 1, 0)
    return sphere_out - sphere_inner


def hollow_sphere_coordinate(outer_radius, inner_radius, center=np.array([0, 0, 0]), resolution=(1, 1, 1)):
    print('    ... creating hollow_sphere_coordinate')
    semisizes = [float(outer_radius), ] * 3
    shape = [float(outer_radius * 2 + 1), ] * 3
    center_no_shifted = [float(outer_radius), ] * 3
    grid = [slice(-x0, dim - x0) for x0, dim in zip(center_no_shifted, shape)]
    position = np.ogrid[grid]
    arr = np.zeros(np.asarray(shape).astype(int), dtype=float)
    for x_i, semisize in zip(position, semisizes):
        arr += (np.abs(x_i / semisize) ** 2)
    sphere_np = arr <= 1.0
    sphere_out = sphere_np.astype(int).transpose(2, 1, 0)
    semisizes = (float(inner_radius),) * 3
    grid = [slice(-x0, dim - x0) for x0, dim in zip(center_no_shifted, shape)]
    position = np.ogrid[grid]
    arr = np.zeros(np.asarray(shape).astype(int), dtype=float)
    for x_i, semisize in zip(position, semisizes):
        arr += (np.abs(x_i / semisize) ** 2)
    sphere_np = arr <= 1.0
    sphere_inner = sphere_np.astype(int).transpose(2, 1, 0)
    sphere_hollow_array = sphere_out - sphere_inner
    return array_3d_to_coordinate(sphere_hollow_array) + center


def hollow_sphere_array_old(dimensions, outer_radius, inner_radius, center, resolution=(1, 1, 1)):
    print('    ... hollow_sphere_array_old')
    sphere_outer_np = np.zeros((dimensions)).astype('uint8')
    sphere_inner_np = np.zeros((dimensions)).astype('uint8')
    cz, cy, cx = center[0], center[1], center[2]
    z, y, x = np.ogrid[-outer_radius: outer_radius, -outer_radius: outer_radius, -outer_radius: outer_radius]
    index = x ** 2 + y ** 2 + z ** 2 <= outer_radius ** 2
    sphere_outer_np[cz - outer_radius:cz + outer_radius, cy - outer_radius:cy + outer_radius,
    cx - outer_radius:cx + outer_radius][index] = 1
    z, y, x = np.ogrid[- inner_radius: inner_radius, - inner_radius: inner_radius, - inner_radius: inner_radius]
    index = x ** 2 + y ** 2 + z ** 2 <= inner_radius ** 2
    sphere_inner_np[cz - inner_radius:cz + inner_radius, cy - inner_radius:cy + inner_radius,
    cx - inner_radius:cx + inner_radius][index] = 1
    sphere_hollow_sitk = sitk.GetImageFromArray((sphere_outer_np - sphere_inner_np).transpose(2, 1, 0))
    sphere_hollow_sitk.SetSpacing(resolution)
    return sphere_hollow_sitk


def same_value_in_array(k, k2):
    print('    ... same_value_in_array')
    if k.shape[1] == 3:
        return k[np.in1d(k.copy().view(dtype='f8,f8,f8').reshape(k.shape[0]),
                         k2.copy().view(dtype='f8,f8,f8').reshape(k2.shape[0]))]
    elif k.shape[1] == 2:
        return k[np.in1d(k.copy().view(dtype='f8,f8').reshape(k.shape[0]),
                         k2.copy().view(dtype='f8,f8').reshape(k2.shape[0]))]
    else:
        print('error: array of shape[1] is not equal to 2 or 3...')
        return None



# ************************************************************************************************************************************************************************************

# #@jit
# def coordinate_binary_image(img):
#    import scipy.ndimage as ndimage
#    if type(img) == np.ndarray:
#        pass
#    else:
#        img_np   = sitk.GetArrayFromImage(img)
#    coordinate_A_x = []
#    coordinate_A_y = []
#    coordinate_A_z = []
#    for i in range(img_np.shape[0]):
#        for j in range(img_np.shape[1]):
#            for k in range(img_np.shape[2]):
#                if img_np[i,j,k] == 1:
#                    coordinate_A_x.append(i)
#                    coordinate_A_y.append(j)
#                    coordinate_A_z.append(k)
#    return np.vstack((np.vstack((np.array(coordinate_A_x), np.array(coordinate_A_y))), np.array(coordinate_A_z))).T


@jit
# def kernel_coordinate
def coordinate_binary_image_old(img):
    if type(img) == np.ndarray:
        pass
    else:
        img = sitk.GetArrayFromImage(img).transpose(2, 1, 0)
    if len(img.shape) == 3:
        coordinate_A_x = []
        coordinate_A_y = []
        coordinate_A_z = []
        #        kernel_coordinate
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                for k in range(img.shape[2]):
                    if img[i, j, k] == 1:
                        coordinate_A_x.append(i)
                        coordinate_A_y.append(j)
                        coordinate_A_z.append(k)
        return np.vstack((np.vstack((np.array(coordinate_A_x), np.array(coordinate_A_y))), np.array(coordinate_A_z))).T
    if len(img.shape) == 2:
        coordinate_A_x = []
        coordinate_A_y = []
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img[i, j] == 1:
                    coordinate_A_x.append(i)
                    coordinate_A_y.append(j)
        return np.vstack((np.array(coordinate_A_x), np.array(coordinate_A_y))).T
    else:
        print('error: image shape dimensions are wrong in function ""coordinate_binary_image')
        return None


def coordinate_binary_image(img):
    print('    ... coordinate_binary_image')
    if type(img) == np.ndarray:
        pass
    else:
        img = sitk.GetArrayFromImage(img).transpose(2, 1, 0)
    return np.argwhere(img == 1)


def img_same_dim(img_fixed, img_small):
    print('    ... img_same_dim')
    if type(img_fixed) != np.ndarray:
        img_fixed_sitk = img_fixed
        img_small_sitk = img_small
    else:
        img_fixed_sitk = sitk.GetImageFromArray(img_fixed)
        img_small_sitk = sitk.GetImageFromArray(img_small)
    img_small_np = sitk.GetArrayFromImage(img_small_sitk).transpose(2, 1, 0)
    dimension_fixed = img_fixed_sitk.GetSize()
    dimension_small = img_small_sitk.GetSize()
    img_empty_np = np.zeros((dimension_fixed[0], dimension_fixed[1], dimension_fixed[2]))
    print('dimension_empty:', img_empty_np.shape)
    print('dimension_fixed:', dimension_fixed)
    print('dimension_small:', dimension_small)
    print('img_empty_np:', img_empty_np.shape)
    print(' % s:%s, %s:%s, %s:%s ' % (int(dimension_fixed[0] / 2 - dimension_small[0] / 2),
                                      int(dimension_fixed[0] / 2 - dimension_small[0] / 2) + dimension_small[0],
                                      int(dimension_fixed[1] / 2 - dimension_small[1] / 2),
                                      int(dimension_fixed[1] / 2 - dimension_small[1] / 2) + dimension_small[1],
                                      int(dimension_fixed[2] / 2 - dimension_small[2] / 2),
                                      int(dimension_fixed[2] / 2 - dimension_small[2] / 2) + dimension_small[2]))
    img_empty_np[
    int(dimension_fixed[0] / 2 - dimension_small[0] / 2):int(dimension_fixed[0] / 2 - dimension_small[0] / 2) +
                                                         dimension_small[0],
    int(dimension_fixed[1] / 2 - dimension_small[1] / 2):int(dimension_fixed[1] / 2 - dimension_small[1] / 2) +
                                                         dimension_small[1],
    int(dimension_fixed[2] / 2 - dimension_small[2] / 2):int(dimension_fixed[2] / 2 - dimension_small[2] / 2) +
                                                         dimension_small[2]] = img_small_np
    img_empty_sitk = sitk.GetImageFromArray(img_empty_np.transpose(2, 1, 0))
    dimension_final = img_empty_sitk.GetSize()
    print('dimension_final:', dimension_final)
    img_empty_sitk.SetSpacing(img_fixed_sitk.GetSpacing())
    return img_empty_sitk


def img_same_properties(img_origin, img, force2notflip='No'):
    print('    ... img_same_properties')
    if type(img) != np.ndarray:
        pass
    else:
        img = sitk.GetImageFromArray(img.transpose(2, 1, 0))
    try:
        img.CopyInformation(img_origin)
    except:
        print('dimension or size are not the same')
    size_origin = img_origin.GetSize()
    reso_origin = img_origin.GetSpacing()
    type_origin_string = img_origin.GetPixelIDTypeAsString()
    origin_origin = img_origin.GetOrigin()
    if size_origin != img.GetSize() and force2notflip.upper() == 'NO':
        print('dimensions does not match, transpose 2 1 0 applied')
        img = sitk.Flip(img, [True, True, True])
        img_np = sitk.GetArrayFromImage(img)
        img = sitk.GetImageFromArray(img_np.transpose(2, 1, 0))
    if '8' in type_origin_string:
        if 'INTEGER' in type_origin_string.upper():
            if 'UNSIGNED' in type_origin_string.upper():
                img = sitk.Cast(img, sitk.sitkUInt8)
            else:
                img = sitk.Cast(img, sitk.sitkInt8)
    elif '16' in type_origin_string:
        if 'INTEGER' in type_origin_string.upper():
            if 'UNSIGNED' in type_origin_string.upper():
                img = sitk.Cast(img, sitk.sitkUInt16)
            else:
                img = sitk.Cast(img, sitk.sitkInt16)
    elif '32' in type_origin_string:
        if 'FLOAT' in type_origin_string.upper():
            img = sitk.Cast(img, sitk.sitkFloat32)
        else:
            if 'INTEGER' in type_origin_string.upper():
                if 'UNSIGNED' in type_origin_string.upper():
                    img = sitk.Cast(img, sitk.sitkUInt32)
                else:
                    img = sitk.Cast(img, sitk.sitkInt32)
    elif '64' in type_origin_string:
        if 'FLOAT' in type_origin_string.upper():
            img = sitk.Cast(img, sitk.sitkFloat64)
        else:
            if 'INTEGER' in type_origin_string.upper():
                if 'UNSIGNED' in type_origin_string.upper():
                    img = sitk.Cast(img, sitk.sitkUInt64)
                else:
                    img = sitk.Cast(img, sitk.sitkInt64)
    else:
        'print(type image not understood...'
    img.SetSpacing(reso_origin)
    img.SetOrigin(origin_origin)
    return img


def cylinder_mask(resolution, dimensions, outer_diameter, inner_diameter, length):
    print('    ... cylinder_mask')
    """
    outer_diameter in mm, inner_diameter in mm, length in mm
    """
    nx = dimensions[0]
    ny = dimensions[1]
    nz = dimensions[2]
    #    center_x, center_y = int(np.rint(outer_diameter/2.0/resolution[0])), int(np.rint(outer_diameter/2.0/resolution[1]))
    center_x, center_y = nx / 2, ny / 2

    dimensions
    outer_radius = int(np.rint(outer_diameter / 2 / resolution[0]))
    y1, x1 = np.ogrid[-center_x:nx - center_x, -center_y:ny - center_y]
    mask1 = x1 ** 2 + y1 ** 2 <= outer_radius ** 2
    array1 = np.zeros((dimensions[0], dimensions[1]))
    array1[mask1] = 1
    inner_radius = int(np.rint(inner_diameter / 2 / resolution[0]))
    y2, x2 = np.ogrid[-center_x:nx - center_x, -center_y:ny - center_y]
    mask2 = x2 ** 2 + y2 ** 2 <= inner_radius ** 2
    array2 = np.zeros((nx, ny))
    array2[mask2] = 1
    cylinder_flat = array1 - array2
    cylinder_np = np.zeros((nx, ny, nz))
    for i in np.arange(int(np.rint(length / resolution[2]))):
        cylinder_np[:, :, nz - i - 1] = cylinder_flat
    cylinder_sitk = sitk.GetImageFromArray(cylinder_np.transpose(2, 1, 0))
    del cylinder_np
    cylinder_sitk.SetSpacing(resolution)
    return cylinder_sitk  # sitk.Cast(cylinder_sitk ,sitk.sitkInt16)


def img_registration_sitk(img_fixed, img_moving):
    print('    ... img_registration')
    initial_transform = sitk.CenteredTransformInitializer(img_fixed, img_moving, sitk.Euler3DTransform(),
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)
    registration_method = sitk.ImageRegistrationMethod()
    # Similarity metric settings.
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)
    registration_method.SetInterpolator(sitk.sitkLinear)
    # Optimizer settings.
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100,
                                                      convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()
    # Setup for the multi-resolution framework.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    # Don't optimize in-place, we would possibly like to run this cell multiple times.
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    final_transform = registration_method.Execute(sitk.Cast(img_fixed, sitk.sitkFloat32),
                                                  sitk.Cast(img_moving, sitk.sitkFloat32))
    # get the information of the transformation sitk result:
    #    print(final_transform)
    final_transform_text = str(final_transform)
    angles = [float(i) for i in [final_transform_text.split('AngleX=')[1].split(' ')[0],
                                 final_transform_text.split('AngleY=')[1].split(' ')[0],
                                 final_transform_text.split('AngleZ=')[1].split('\n')[0]]][::-1]
    matrix = [float(i) for i in
              final_transform_text.split('Matrix: \n       ')[1].split(' \n     Offset:')[0].replace('\n       ',
                                                                                                     '').split(
                  ' ')]  # ! 2, 1, 0 ordered!!
    offset = [float(i) for i in final_transform_text.split('Offset: [')[1].split(']')[0].split(', ')][::-1]
    center = [float(i) for i in final_transform_text.split('Center: [')[1].split(']')[0].split(', ')][::-1]
    translation = [float(i) for i in final_transform_text.split('Translation: [')[1].split(']')[0].split(', ')][::-1]
    #    use img_sitk_rotated = Swissknife.rotate(img_moving, angles[0], angles[1], angles[2], ang_type='rad', center[0], center[1], center[2])   to rotate back
    return angles, matrix, offset, center, translation, final_transform


def registration_sitk_2(img_fixed, img_moving):
    print('    ... registration')
    if img_fixed.GetSize() != img_moving.GetSize():  # if images do not have the same dimensions: create a similar empty larger image and insert the images inside
        dimension_moving = img_moving.GetSize()
        dimension_fixed = img_fixed.GetSize()
        img_moving_np = sitk.GetArrayFromImage(img_moving).transpose(2, 1, 0)
        if img_fixed.GetSize()[0] > img_moving.GetSize()[0]:
            dim_x = img_fixed.GetSize()[0]
        else:
            dim_x = img_moving.GetSize()[0]
        if img_fixed.GetSize()[1] > img_moving.GetSize()[1]:
            dim_y = img_fixed.GetSize()[1]
        else:
            dim_y = img_moving.GetSize()[1]
        if img_fixed.GetSize()[2] > img_moving.GetSize()[2]:
            dim_z = img_fixed.GetSize()[2]
        else:
            dim_z = img_moving.GetSize()[2]
        img_moving_big_np = np.zeros((dim_x, dim_y, dim_z))
        img_moving_big_np[
        int(dim_x / 2 - dimension_moving[0] / 2):int(dim_x / 2 - dimension_moving[0] / 2) + dimension_moving[0],
        int(dim_y / 2 - dimension_moving[1] / 2):int(dim_y / 2 - dimension_moving[1] / 2) + dimension_moving[1],
        int(dim_z / 2 - dimension_moving[2] / 2):int(dim_z / 2 - dimension_moving[2] / 2) + dimension_moving[
            2]] = img_moving_np
        img_moving = sitk.GetImageFromArray(img_moving_big_np.transpose(2, 1, 0))
        if img_fixed.GetSize()[0] < img_moving.GetSize()[0] or img_fixed.GetSize()[1] < img_moving.GetSize()[1] or \
                img_fixed.GetSize()[2] < img_moving.GetSize()[2]:
            img_fixed_np = sitk.GetArrayFromImage(img_fixed).transpose(2, 1, 0)
            img_fixed_big_np = sitk.GetImageFromArray(np.zeros((dim_x, dim_y, dim_z)).transpose(2, 1, 0))
            print(' % s:%s,%s:%s,%s:%s' % (int(dim_x / 2 - dimension_fixed[0] / 2),
                                           int(dim_x / 2 - dimension_fixed[0] / 2) + dimension_fixed[0],
                                           int(dim_y / 2 - dimension_fixed[1] / 2),
                                           int(dim_y / 2 - dimension_fixed[1] / 2) + dimension_fixed[1],
                                           int(dim_z / 2 - dimension_fixed[2] / 2),
                                           int(dim_z / 2 - dimension_fixed[2] / 2) + dimension_fixed[2]))
            img_fixed_big_np[
            int(dim_x / 2 - dimension_fixed[0] / 2):int(dim_x / 2 - dimension_fixed[0] / 2) + dimension_fixed[0],
            int(dim_y / 2 - dimension_fixed[1] / 2):int(dim_y / 2 - dimension_fixed[1] / 2) + dimension_fixed[1],
            int(dim_z / 2 - dimension_fixed[2] / 2):int(dim_z / 2 - dimension_fixed[2] / 2) + dimension_fixed[
                2]] = img_fixed_np
            img_fixed = sitk.GetImageFromArray(img_fixed_big_np.transpose(2, 1, 0))
    print('img_fixed dimensions:', img_fixed.GetSize())
    print('img_moving dimensions:', img_moving.GetSize())
    img_description(img_fixed)
    img_description(img_moving)
    img_fixed = sitk.Cast(img_fixed, sitk.sitkFloat32)
    img_moving = sitk.Cast(img_moving, sitk.sitkFloat32)
    img_moving.SetSpacing(img_fixed.GetSpacing())
    initial_transform = sitk.CenteredTransformInitializer(img_fixed, img_moving, sitk.Euler3DTransform(),
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)
    registration_method = sitk.ImageRegistrationMethod()
    # Similarity metric settings.
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.005)
    registration_method.SetInterpolator(sitk.sitkLinear)
    # Optimizer settings.
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100,
                                                      convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()
    # Setup for the multi-resolution framework.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    # Don't optimize in-place, we would possibly like to run this cell multiple times.
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    final_transform = registration_method.Execute(sitk.Cast(img_fixed, sitk.sitkFloat32),
                                                  sitk.Cast(img_moving, sitk.sitkFloat32))
    # get the information of the transformation sitk result:
    final_transform_text = str(final_transform)
    angles = [float(i) for i in [final_transform_text.split('AngleX=')[1].split(' ')[0],
                                 final_transform_text.split('AngleY=')[1].split(' ')[0],
                                 final_transform_text.split('AngleZ=')[1].split('\n')[0]]][::-1]
    matrix = [float(i) for i in
              final_transform_text.split('Matrix: \n       ')[1].split(' \n     Offset:')[0].replace('\n       ',
                                                                                                     '').split(
                  ' ')]  # ! 2, 1, 0 ordered!!
    offset = [float(i) for i in final_transform_text.split('Offset: [')[1].split(']')[0].split(', ')][::-1]
    center = [float(i) for i in final_transform_text.split('Center: [')[1].split(']')[0].split(', ')][::-1]
    translation = [float(i) for i in final_transform_text.split('Translation: [')[1].split(']')[0].split(', ')][::-1]
    return angles, matrix, offset, center, translation, img_fixed, img_moving


def mask_image(size, r_factor, img, resolution, center_x='center', center_y='center'):
    print('    ... mask_image')
    if type(img) != np.ndarray:
        img_sitk = img
    else:
        img_sitk = sitk.GetImageFromArray(img.transpose(2, 1, 0))
    reso = img_sitk.GetSpacing()
    if center_x == 'center':
        center_x = size[0] / 2
    else:
        center_x = size[0] / 2 * center_x * 2
    if center_y == 'center':
        center_y = size[1] / 2
    else:
        center_y = size[1] / 2 * center_y * 2
    len_x = size[0]
    len_y = size[1]
    len_z = size[2]
    radius = r_factor * np.mean(np.array([len_x, len_y])) / 2
    y, x = np.ogrid[-center_x:len_x - center_x, -center_y:len_y - center_y]
    mask = x ** 2 + y ** 2 <= radius ** 2
    disk = np.zeros((len_x, len_y))
    disk[mask] = 1
    mask = np.zeros((len_z, len_y, len_x))
    mask = sitk.GetImageFromArray(mask)
    mask.SetSpacing(resolution)
    mask_np = np.array(sitk.GetArrayFromImage(mask)).transpose(2, 1, 0)
    for i in np.arange(len_z):
        mask_np[:, :, i] = disk
    mask_sitk = sitk.GetImageFromArray(mask_np.transpose(2, 1, 0))
    mask_sitk.SetSpacing(reso)
    cut_3d_2_images(img_sitk, mask_sitk, size, color='b', title='')
    if type(img_sitk) == np.ndarray:
        return sitk.GetImageFromArray(img_sitk * mask)
    else:
        return sitk.GetImageFromArray(
            (sitk.GetArrayFromImage(img_sitk) * mask_np.transpose(2, 1, 0)).transpose(0, 1, 2))


def bvtv_mask(size, r_factor, img, start_z, end_z, resolution, center_x='center', center_y='center'):
    print('    ... bvtv_mask')
    resolution = img.GetSpacing(img)
    start_z = int(np.rint(start_z * size[2]))
    end_z = int(np.rint(end_z * size[2]))
    if center_x == 'center':
        center_x = size[0] / 2
    else:
        center_x = size[0] / 2 * center_x * 2
    if center_y == 'center':
        center_y = size[1] / 2
    else:
        center_y = size[1] / 2 * center_y * 2
    len_x = size[0]
    len_y = size[1]
    len_z = size[2]
    radius = r_factor * np.mean(np.array([len_x, len_y])) / 2
    y, x = np.ogrid[-center_x:len_x - center_x, -center_y:len_y - center_y]
    mask = x ** 2 + y ** 2 <= radius ** 2
    disk = np.zeros((len_x, len_y))
    disk[mask] = 1
    mask = np.zeros((len_z, len_y, len_x))
    mask = sitk.GetImageFromArray(mask)
    mask.SetSpacing(resolution)
    mask_np = np.array(sitk.GetArrayFromImage(mask)).transpose(2, 1, 0)
    for i in np.arange(start_z, end_z):
        mask_np[:, :, i] = disk
    mask_sitk = sitk.GetImageFromArray(mask_np.transpose(2, 1, 0))
    mask_sitk.SetSpacing(resolution)
    cut_3d_2_images(img, mask_sitk, size, color='r', title='')
    unique, counts = np.unique(mask_np * 3 + sitk.GetArrayFromImage(img).transpose(2, 1, 0), return_counts=True)
    print(' ... BV / TV VOI = %.2f' % float(float(counts[-1]) / (float(counts[-2]) + float(counts[-1]))))
    return float(float(counts[-1]) / (float(counts[-2]) + float(counts[-1])))


def dilate(img, radius=20):
    if type(img) == np.ndarray:
        img = sitk.GetImageFromArray(img.transpose(2, 1, 0))
    else:
        pass
    if img.GetPixelIDTypeAsString() != '16-bit int':
        img = sitk.Cast(img, sitk.sitkInt16)
    print('    ... start dilation sitk with radius %s mm' % radius)
    dilation = sitk.BinaryDilateImageFilter()
    dilation.SetKernelRadius(radius)
    return dilation.Execute(img)


def erode(img, radius=20):
    if type(img) == np.ndarray:
        try:
            img = sitk.GetImageFromArray(img.transpose(2, 1, 0))
        except:
            img = sitk.GetImageFromArray(img.transpose(1, 0))
    else:
        pass
    if img.GetPixelIDTypeAsString() != '16-bit int':
        img = sitk.Cast(img, sitk.sitkInt16)
    print('    ... start dilation sitk with radius %s mm' % radius)
    erodation = sitk.BinaryErodeImageFilter()
    erodation.SetKernelRadius(radius)
    return erodation.Execute(img)


def dilate_erode(img, radius=20):
    if type(img) == np.ndarray:
        img = sitk.GetImageFromArray(img.transpose(2, 1, 0))
    else:
        pass
    if img.GetPixelIDTypeAsString() != '16-bit int':
        img = sitk.Cast(img, sitk.sitkInt16)
    print('    ... start dilation sitk with radius %s mm' % radius)
    dilation = sitk.BinaryDilateImageFilter()
    dilation.SetKernelRadius(radius)
    dil = dilation.Execute(img)
    print('    ... start erosion sitk with radius %s mm' % radius)
    erodation = sitk.BinaryErodeImageFilter()
    erodation.SetKernelRadius(radius)
    return erodation.Execute(dil)


def closing(img, radius=20):
    if type(img) == np.ndarray:
        img = sitk.GetImageFromArray(img.transpose(2, 1, 0))
    else:
        pass
    if img.GetPixelIDTypeAsString() != '16-bit int':
        img = sitk.Cast(img, sitk.sitkInt16)
    vectorRadius = (radius, radius, radius)
    kernel = sitk.sitkBall
    closed = sitk.BinaryMorphologicalClosing(img, vectorRadius, kernel)
    return closed


def dilate_erode_ndimage(image, iterations=10):
    from scipy import ndimage
    if type(image) == np.ndarray:
        pass
    else:
        image = sitk.GetArrayFromImage(image).transpose(2, 1, 0)
    struct = ndimage.generate_binary_structure(3, 2)
    print('    ... start dilation ndimage with  %s iterations' % iterations)
    dilated = ndimage.binary_dilation(image, structure=struct, iterations=iterations).astype('int')
    print('    ... start erosion ndimage with  %s iterations' % iterations)
    eroded_np = ndimage.binary_erosion(dilated, structure=struct, iterations=iterations).astype('int')
    img_sitk = sitk.GetImageFromArray(eroded_np.transpose(2, 1, 0))
    try:
        img_sitk.SetSpacing(image.GetSPacing())
    except:
        pass
    return img_sitk


def erode_dilate(img, radius=20):
    if type(img) == np.ndarray:
        try:
            img = sitk.GetImageFromArray(img.transpose(2, 1, 0))
        except:
            img = sitk.GetImageFromArray(img.transpose(1, 0))
    else:
        pass
    if img.GetPixelIDTypeAsString() != '16-bit int':
        img = sitk.Cast(img, sitk.sitkInt16)
    print('    ... start erosion with radius %s mm' % radius)
    erodation = sitk.BinaryErodeImageFilter()
    erodation.SetKernelRadius(radius)
    ero = erodation.Execute(img)
    print('    ... start dilation with radius %s mm' % radius)
    dilation = sitk.BinaryDilateImageFilter()
    dilation.SetKernelRadius(radius)
    return dilation.Execute(ero)


def center_gravity(image, resolution=(1, 1, 1)):
    import scipy.ndimage as ndimage
    if type(image) == np.ndarray:
        if image.ndim == 3:
            dim = 3
        else:
            dim = 2
    else:
        if image.GetDimension() == 3:
            image = sitk.GetArrayFromImage(image).transpose(2, 1, 0)
            dim = 3
        else:
            image = sitk.GetArrayFromImage(image).transpose(1, 0)
            dim = 2

    CG_temp = ndimage.measurements.center_of_mass(image)
    if dim == 2:
        return [CG_temp[1] * resolution[0], CG_temp[0] * resolution[1]]
    else:
        return [CG_temp[2] * resolution[0], CG_temp[1] * resolution[1], CG_temp[0] * resolution[2]]


def array_3d_to_coordinate(array_3d, value=1):
    if type(array_3d) == np.ndarray:
        pass
    else:
        array_3d = sitk.GetArrayFromImage(array_3d).transpose(2, 1, 0)
    return np.rint(np.array(np.where(array_3d == value)).T).astype('int')


@jit
def kernel_coordinate_to_array_3d(coordinate, array_3d, value):
    for j in range(coordinate.shape[0]):
        array_3d[int(coordinate[j, 0]), int(coordinate[j, 1]), int(coordinate[j, 2])] = value
    return array_3d


def coordinate_to_array_3d_old_old(coordinate, dim='max_of_coordinate', value=1):
    print('    ... coordinate_to_array_3d')
    mini = np.min(coordinate)
    if mini < 0:
        print('        ... negative coordinates detected, -> shifted (np.abs) %s' % (mini))
        if np.min(coordinate[:, 0]) < 0:
            coordinate[:, 0] += np.abs(np.min(coordinate[:, 0]))
        if np.min(coordinate[:, 1]) < 0:
            coordinate[:, 1] += np.abs(np.min(coordinate[:, 1]))
        if np.min(coordinate[:, 2]) < 0:
            coordinate[:, 2] += np.abs(np.min(coordinate[:, 2]))
    if dim == 'max_of_coordinate':
        # dim = [int(np.max(coordinate[:,0]))+int(np.abs(np.min(coordinate[:,0])))+1,
        #        int(np.max(coordinate[:,1]))+int(np.abs(np.min(coordinate[:,1])))+1,
        #        int(np.max(coordinate[:,2]))+int(np.abs(np.min(coordinate[:,2])))+1]
        dim = [int(np.max(coordinate[:, 0])) + 1,
               int(np.max(coordinate[:, 1])) + 1,
               int(np.max(coordinate[:, 2])) + 1]
    array_3d = np.zeros((dim[0], dim[1], dim[2]))
    array_3d = kernel_coordinate_to_array_3d(coordinate, array_3d, value)
    return array_3d


# @pysnooper.snoop()
def coordinate_to_array_3d(coordinate, dim='max_of_coordinate', value=1):
    if type(coordinate) != np.ndarray:
        coordinate = np.array(coordinate)
    print('    ... coordinate_to_array_3d')
    mini = np.min(coordinate)
    if mini < 0:
        print('        ... negative coordinates detected, -> shifted (np.abs) %s' % (mini))
        if np.min(coordinate[:, 0]) < 0:
            coordinate[:, 0] += np.abs(np.min(coordinate[:, 0]))
        if np.min(coordinate[:, 1]) < 0:
            coordinate[:, 1] += np.abs(np.min(coordinate[:, 1]))
        if np.min(coordinate[:, 2]) < 0:
            coordinate[:, 2] += np.abs(np.min(coordinate[:, 2]))
    if dim == 'max_of_coordinate':
        dim = np.array(
            [int(np.max(coordinate[:, 0]) + 1), int(np.max(coordinate[:, 1]) + 1), int(np.max(coordinate[:, 2]) + 1)])
    array_3d = np.zeros((dim[0], dim[1], dim[2]))
    array_3d[coordinate[:, 0].astype(int), coordinate[:, 1].astype(int), coordinate[:, 2].astype(int)] = value
    return array_3d


def coordinate_to_array_3d_new(coordinates, reso=1):
    ## https://github.com/mikedh/trimesh/blob/master/trimesh/voxel.py : def sparse_to_matrix(sparse)
    points = np.asanyarray(coordinates, dtype=np.float64)
    reso = float(reso)
    # find the minimum value of points for origin
    origin = points.min(axis=0)
    # convert points to occupied voxel cells
    index = ((points - origin) / reso).round().astype(np.int64)
    # convert voxel indices to a matrix
    index = np.asanyarray(index, dtype=np.int)
    shape = index.max(axis=0) + 1
    matrix = np.zeros(np.product(shape), dtype=np.bool)
    multiplier = np.array([np.product(shape[1:]), shape[2], 1])
    index_ = (index * multiplier).sum(axis=1)
    matrix[index_] = True
    dense = matrix.reshape(shape)
    return dense.astype(int)

    # ## https://github.com/mikedh/trimesh/blob/master/trimesh/voxel.py : def sparse_to_matrix(sparse)
    # coordinate = np.asanyarray(coordinate, dtype=np.int)
    # shape = coordinate.max(axis=0) + 1
    # matrix = np.zeros(np.product(shape), dtype=np.bool)
    # multiplier = np.array([np.product(shape[1:]), shape[2], 1])
    # index = (coordinate * multiplier).sum(axis=1)
    # matrix[index] = True
    # array_3d = matrix.reshape(shape)
    # return array_3d.astype(int)


def write_image_from_coordinates_offset(coordinate, path):
    mini = np.min(coordinate, axis=0)
    origin = np.zeros(3)
    for i in range(len(mini)):
        if mini[i] < 0:
            origin[i] = mini[i] * -1
        else:
            origin[i] = 0
    img_np = coordinate_to_array_3d(coordinate)
    img_np_layers_added = np.zeros((np.array(img_np.shape) + 2))
    img_np_layers_added[1:-1, 1:-1, 1:-1] = img_np
    del img_np
    img_sitk = sitk.GetImageFromArray(img_np_layers_added.transpose(2, 1, 0))
    img_sitk.SetOrigin((-origin[0]-1, -origin[1]-1, -origin[2]-1))
    write_MHD(img_sitk, path)


def img_16_bit(img):
    return sitk.Cast(img, sitk.sitkInt16)


def normi(vector):
    norm_unit = np.linalg.norm(vector)
    return vector / norm_unit


def linear_regression_3d(data):
    datamean = data.mean(axis=0)
    uu, dd, vv = np.linalg.svd(data - datamean)
    return normi(vv[0])


def angle_between_vectors(v1, v2):
    return np.arccos(np.dot(v1, v2) / (np.sqrt(np.dot(v1, v1)) * np.sqrt(np.dot(v2, v2))))


def closet_point_between_vector(v1, v2, start_v1, start_v2):
    # morroworks.com/Content/Docs/Rays%20closest%20point.pdf
    print('    ... closet_point_between_vector')
    c = start_v2 - start_v1
    D = start_v1 + v1 * ((-np.dot(v1, v2) * np.dot(v2, c) + np.dot(v1, c) * np.dot(v2, v2)) / (
                np.dot(v1, v1) * np.dot(v2, v2) - np.dot(v1, v2) * np.dot(v1, v2)))
    E = start_v2 + v2 * ((np.dot(v1, v2) * np.dot(v1, c) - np.dot(v2, c) * np.dot(v1, v1)) / (
                np.dot(v1, v1) * np.dot(v2, v2) - np.dot(v1, v2) * np.dot(v1, v2)))
    return (D + E) / 2.


def find_new_point_perpendicular_to_vector_end_start_and_point_new_point(point, start, end):  # end = start + vector
    return start + np.dot((end - start) / np.linalg.norm(end - start),
                          1.0 / np.linalg.norm((end - start)) * (point - start)) * (end - start)


def img_description(img):
    if type(img) != np.ndarray:
        pass
    else:
        print('numpy array given, please provide a sitk image...')
    print('\nSize     : ', img.GetSize())
    print('Spacing  : ', img.GetSpacing())
    print('Type     : ', img.GetPixelIDTypeAsString())
    print('Origin   : ', img.GetOrigin())
    print('')


def img_min_max(img):
    if type(img) != np.ndarray:
        pass
    else:
        print('numpy array given, please provide a sitk image...')
    statistics_image_filter = sitk.StatisticsImageFilter()
    statistics_image_filter.Execute(img)
    return statistics_image_filter.GetMinimum(), statistics_image_filter.GetMaximum()


def sizeof_fmt(num, suffix='B'):
    ''' By Fred Cirera, after https://stackoverflow.com/a/1094933/1870254'''
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)



def coordinate_to_vertice(coordinates, reso=1, downscale_factor=1):
    coordinates = np.asanyarray(coordinates, dtype=np.float64)
    reso = float(reso)

    # find the minimum value of points for origin
    origin = coordinates.min(axis=0)
    # convert points to occupied voxel cells
    index = ((coordinates - origin) / reso).round().astype(np.int64)

    # convert voxel indices to a matrix
    img_np = coordinate_to_array_3d_new(index)

    pad_width = 1
    img_np_padded = np.pad(img_np[::downscale_factor, ::downscale_factor, ::downscale_factor],
                           pad_width=(pad_width),
                           mode='constant',
                           constant_values=(0))
    # Run marching cubes.
    meshed = measure.marching_cubes_lewiner(volume = img_np_padded,
                                            level = .5,  # it is a boolean voxel grid
                                            spacing = (reso, reso, reso))
    if len(meshed) == 2:
        print('using old marching cubes, may not be watertight!')
        vertices, faces = meshed
        normals = None
    elif len(meshed) == 4:
        vertices, faces, normals, vals = meshed

    vertices = np.subtract(np.add(vertices, origin), pad_width * reso)
    return vertices, faces, normals, vals


def vertice_to_vtk(vertices, faces, filename, comment=None):
    number_vertices = vertices.shape[0]
    number_faces = faces.shape[0]
    # make header and subheader dataframe
    header = ['# vtk DataFile Version 3.0',
              '%s' % comment,
              'ASCII',
              'DATASET POLYDATA',
              'POINTS %i float' % number_vertices
              ]
    header_df = pd.DataFrame(header)
    sub_header = ['POLYGONS %i %i' % (number_faces, 4 * number_faces)]
    sub_header_df = pd.DataFrame(sub_header)
    # make dataframe from vertices
    vertex_df = pd.DataFrame(vertices)
    # make dataframe from faces, appending first row of 3's (indicating the polygons are triangles)
    triangles = np.reshape(3 * (np.ones(number_faces)), (number_faces, 1))
    triangles = triangles.astype(int)
    faces = faces.astype(int)
    faces_df = pd.DataFrame(np.concatenate((triangles, faces), axis=1))
    # write dfs to csv
    header_df.to_csv(filename, header=None, index=False)
    with open(filename, 'a') as f:
        vertex_df.to_csv(f, header=False, index=False, float_format='%.3f', sep=' ')
    with open(filename, 'a') as f:
        sub_header_df.to_csv(f, header=False, index=False)
    with open(filename, 'a') as f:
        faces_df.to_csv(f, header=False, index=False, float_format='%.0f', sep=' ')



def vertice_to_stl(vertices, faces, normals, filename):
    # https://github.com/mikedh/trimesh/blob/master/trimesh/exchange/stl.py  : def export_stl_ascii(mesh)

    triangles = vertices.view(np.ndarray)[faces]
    vectors = np.diff(triangles, axis=1)
    triangles_cross = np.cross(vectors[:, 0], vectors[:, 1])

    # normals, valid = triangles.normals(triangles, triangles_cross)

    # unit, valid = util.unitize(triangles_cross, check_valid=True)

    triangles_cross = np.asanyarray(triangles_cross)
    threshold = np.finfo(np.float64).resolution * 100
    norm = np.sqrt(np.dot(triangles_cross * triangles_cross,
                          [1.0] * triangles_cross.shape[1]))
    # non-zero norms
    valid = norm > threshold
    # in-place reciprocal of nonzero norms
    norm[valid] **= -1
    # tile reciprocal of norm
    tiled = np.tile(norm, (triangles_cross.shape[1], 1)).T
    # multiply by reciprocal of norm
    unit = triangles_cross * tiled
    # return unit[valid], valid
    normals = unit[valid]

    if valid.all():
        face_normals = normals
    else:
        print('error')
        padded = np.zeros((len(triangles), 3),
                          dtype=np.float64)
        padded[valid] = normals
        face_normals = padded

    blob = np.zeros((len(faces), 4, 3))
    blob[:, 0, :] = face_normals
    blob[:, 1:, :] = triangles

    # create a lengthy format string for the data section of the file
    format_string = 'facet normal {} {} {}\nouter loop\n'
    format_string += 'vertex {} {} {}\n' * 3
    format_string += 'endloop\nendfacet\n'
    format_string *= len(faces)

    # concatenate the header, data, and footer
    stl = 'solid \n'
    stl += format_string.format(*blob.reshape(-1))
    stl += 'endsolid'

    file = open(filename, 'w')
    file.write(stl)
    file.close()


def coordinate_to_vtk(coordinates, filename, data=None, comment=None, downscale_factor=1, reso=1):
    '''
    Creates ASCII coded vtk file from numpy arrays using pandas.
    Inputs:
    -------
    (mandatory)
    * filename: str, path to location where vtk file should be stored
    * vertices: numpy array with vertex coordinates,  shape (n_vertices, 3)
    * faces: numpy array with face specifications, shape (n_faces, 3)
    (optional)
    * data: numpy array with data points, shape (n_vertices, n_datapoints)
        NOTE: n_datapoints can be =1 but cannot be skipped (n_vertices,)
    * comment: str, is written into the comment section of the vtk file
    Usage:
    ---------------------
    write_vtk('/path/to/vtk/file.vtk', v_array, f_array)
    '''

    coordinates = np.asanyarray(coordinates, dtype=np.float64)
    reso = float(reso)

    # find the minimum value of points for origin
    origin = coordinates.min(axis=0)
    # convert points to occupied voxel cells
    index = ((coordinates - origin) / reso).round().astype(np.int64)

    # convert voxel indices to a matrix
    img_np = coordinate_to_array_3d(index)

    pad_width = 1
    img_np_padded = np.pad(img_np[::downscale_factor, ::downscale_factor, ::downscale_factor],
                           pad_width=(pad_width),
                           mode='constant',
                           constant_values=(1))
    # Run marching cubes.
    meshed = measure.marching_cubes_lewiner(volume = img_np_padded,
                                            level = .5,  # it is a boolean voxel grid
                                            spacing = (reso, reso, reso))
    if len(meshed) == 2:
        print('using old marching cubes, may not be watertight!')
        vertices, faces = meshed
        normals = None
    elif len(meshed) == 4:
        vertices, faces, normals, vals = meshed

    vertices = np.subtract(np.add(vertices, origin), pad_width * reso)

    # infer number of vertices and faces
    number_vertices = vertices.shape[0]
    number_faces = faces.shape[0]
    if data is not None:
        number_data = data.shape[0]
    # make header and subheader dataframe
    header = ['# vtk DataFile Version 3.0',
              '%s' % comment,
              'ASCII',
              'DATASET POLYDATA',
              'POINTS %i float' % number_vertices
              ]
    header_df = pd.DataFrame(header)
    sub_header = ['POLYGONS %i %i' % (number_faces, 4 * number_faces)]
    sub_header_df = pd.DataFrame(sub_header)
    # make dataframe from vertices
    vertex_df = pd.DataFrame(vertices)
    # make dataframe from faces, appending first row of 3's (indicating the polygons are triangles)
    triangles = np.reshape(3 * (np.ones(number_faces)), (number_faces, 1))
    triangles = triangles.astype(int)
    faces = faces.astype(int)
    faces_df = pd.DataFrame(np.concatenate((triangles, faces), axis=1))
    # write dfs to csv
    header_df.to_csv(filename, header=None, index=False)
    with open(filename, 'a') as f:
        vertex_df.to_csv(f, header=False, index=False, float_format='%.3f', sep=' ')
    with open(filename, 'a') as f:
        sub_header_df.to_csv(f, header=False, index=False)
    with open(filename, 'a') as f:
        faces_df.to_csv(f, header=False, index=False, float_format='%.0f', sep=' ')
    # if there is data append second subheader and data
    if data != None:
        datapoints = data.shape[1]
        sub_header2 = ['POINT_DATA %i' % number_data,
                       'SCALARS EmbedVertex float %i' % datapoints,
                       'LOOKUP_TABLE default']
        sub_header_df2 = pd.DataFrame(sub_header2)
        data_df = pd.DataFrame(data)
        with open(filename, 'a') as f:
            sub_header_df2.to_csv(f, header=False, index=False)
        with open(filename, 'a') as f:
            data_df.to_csv(f, header=False, index=False, float_format='%.16f', sep=' ')


def mhd_to_vtk_smooth_decimate(path, output_name=None, write_stl=None):
    filePath = path
    reader = vtk.vtkMetaImageReader()
    reader.SetFileName(filePath)
    reader.Update()

    dmc = vtk.vtkDiscreteMarchingCubes()
    dmc.SetInputConnection(reader.GetOutputPort())
    dmc.GenerateValues(1, 1, 1)
    dmc.Update()

    smoothingIterations = 75
    passBand = 0.0000001
    featureAngle = 120.0
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputConnection(dmc.GetOutputPort())
    smoother.SetNumberOfIterations(smoothingIterations)
    # smoother.BoundarySmoothingOn()
    smoother.BoundarySmoothingOff()
    # smoother.FeatureEdgeSmoothingOn()
    smoother.FeatureEdgeSmoothingOff()
    smoother.SetFeatureAngle(featureAngle)
    smoother.SetPassBand(passBand)
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.Update()

    decimate = vtk.vtkQuadricClustering()
    decimate.SetInputData(smoother.GetOutput())
    decimate.SetNumberOfXDivisions(32)
    decimate.SetNumberOfYDivisions(32)
    decimate.SetNumberOfZDivisions(32)
    # decimate.SetTargetReduction(.40)
    decimate.UseFeatureEdgesOn()
    decimate.Update()


    smoothingIterations = 2
    # passBand = 0.1
    featureAngle = 120.0
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputConnection(decimate.GetOutputPort())
    smoother.SetNumberOfIterations(smoothingIterations)
    smoother.BoundarySmoothingOn()
    # smoother.BoundarySmoothingOff()
    smoother.FeatureEdgeSmoothingOn()
    # smoother.FeatureEdgeSmoothingOff()
    smoother.SetFeatureAngle(featureAngle)
    # smoother.SetPassBand(passBand)
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.Update()


    # import optimesh
    # optimesh.cvt.quasi_newton_uniform_full()
    cleaned = vtk.vtkCleanPolyData()
    cleaned.SetInputData(smoother.GetOutput())
    cleaned.Update()


    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(smoother.GetOutput())
    # writer.SetInputData(cleaned.GetOutput())
    if output_name:
        writer.SetFileName(output_name)
    else:
        writer.SetFileName(path.replace('.mhd', '.vtk'))
    writer.Update()

    if write_stl:
        print('print_stl')
        writer = vtk.vtkSTLWriter()
        writer.SetInputConnection(cleaned.GetOutputPort())
        writer.SetFileTypeToBinary()
        if output_name:
            writer.SetFileName(output_name.replace('.vtk', '.stl'))
            print(output_name)
        else:
            writer.SetFileName(path.replace('.mhd', '.stl'))
        writer.Update()


def icp(currSource_filename, currTarget_filename):
    currSource = vtk.vtkDataSetReader()
    currSource.SetFileName(currSource_filename)
    currSource.Update()

    currTarget = vtk.vtkDataSetReader()
    currTarget.SetFileName(currTarget_filename)
    currTarget.Update()

    icp = vtk.vtkIterativeClosestPointTransform()

    icp.SetSource(currSource.GetOutput())
    icp.SetTarget(currTarget.GetOutput())

    icp.SetMeanDistanceModeToAbsoluteValue()
    icp.SetMeanDistanceModeToRMS()

    icp.GetLandmarkTransform().SetModeToAffine()
    # icp.GetLandmarkTransform().SetModeToRigidBody()
    # icp.GetLandmarkTransform().SetModeToSimilarity()

    # m_icp_iterations = 1000000
    # icp.SetMaximumNumberOfIterations(m_icp_iterations)
    icp.SetMaximumMeanDistance(0.00001)

    icp.StartByMatchingCentroidsOn()
    # icp.StartByMatchingCentroidsOff()

    icp.CheckMeanDistanceOn()
    # icp.CheckMeanDistanceOff()

    icp.Modified()
    icp.Update()

    transform = vtk.vtkTransform()
    transform.SetMatrix(icp.GetMatrix())

    print(icp.GetMatrix())
    # m_initialMatrix = vtk.DeepCopy(icp.GetMatrix())

    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetInputConnection(currSource.GetOutputPort())
    transformFilter.SetTransform(transform)
    transformFilter.Update()

    # transformFilter = vtk.vtkTransformFilter()
    # transformFilter.SetInputConnection(currTarget.GetOutputPort())
    # transformFilter.SetTransform(transform)

    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(transformFilter.GetOutput())
    writer.SetFileName(currTarget_filename.replace('.vtk', '_transformed.vtk'))
    writer.Update()


def stl_binary_to_ascii(input_path, output_path):
    """
    https://github.com/IsseiMori/binary-stl-toASCII/blob/master/BinaryToASCII.py
    :param input_path:
    :param output_path:
    :return:
    """

    infile = open('binary.stl')  # import binary stl file
    out = open('ASCII.stl', 'w')  # export ascii stl file

    data = infile.read()
    out.write("solid ")
    for x in range(0, 80):
        if not ord(data[x]) == 0:
            out.write(struct.unpack('c', data[x])[0])
        else:
            pass
    out.write("\n")

    number = data[80] + data[81] + data[82] + data[83]
    faces = struct.unpack('I', number)[0]

    for x in range(0, faces):
        out.write("facet normal ")
        xc = data[84 + x * 50] + data[85 + x * 50] + data[86 + x * 50] + data[87 + x * 50]
        yc = data[88 + x * 50] + data[89 + x * 50] + data[90 + x * 50] + data[91 + x * 50]
        zc = data[92 + x * 50] + data[93 + x * 50] + data[94 + x * 50] + data[95 + x * 50]
        out.write(str(struct.unpack('f', xc)[0]) + " ")
        out.write(str(struct.unpack('f', yc)[0]) + " ")
        out.write(str(struct.unpack('f', zc)[0]) + "\n")
        out.write("outer loop\n")

        for y in range(1, 4):
            out.write("vertex ")
            xc = data[84 + y * 12 + x * 50] + data[85 + y * 12 + x * 50] + data[86 + y * 12 + x * 50] + data[
                87 + y * 12 + x * 50]
            yc = data[88 + y * 12 + x * 50] + data[89 + y * 12 + x * 50] + data[90 + y * 12 + x * 50] + data[
                91 + y * 12 + x * 50]
            zc = data[92 + y * 12 + x * 50] + data[93 + y * 12 + x * 50] + data[94 + y * 12 + x * 50] + data[
                95 + y * 12 + x * 50]
            out.write(str(struct.unpack('f', xc)[0]) + " ")
            out.write(str(struct.unpack('f', yc)[0]) + " ")
            out.write(str(struct.unpack('f', zc)[0]) + "\n")
        out.write("endloop\n")
        out.write("endfacet\n")

    out.close()


def vtk_read_stl(fname):
    """
    https://gist.github.com/Hodapp87/8874941
    Load the given STL file, and return a vtkPolyData object for it
    :param fname:
    :return:
    """
    print('    ... read mesh vtk')
    reader = vtk.vtkSTLReader()
    reader.SetFileName(fname)
    reader.Update()
    polydata = reader.GetOutput()
    return polydata


def vtk_decimate(mesh_vtk, target_reduction=0.01):
    """
    https://stackoverflow.com/questions/38197212/mesh-decimation-in-python
    :param mesh_vtk:
    :param target_reduction:
    :return:
    """
    print('    ... decimate mesh vtk')
    from vtk import (vtkPolyData, vtkDecimatePro)
    decimate = vtkDecimatePro()
    decimate.SetInputData(mesh_vtk)
    decimate.SetTargetReduction(target_reduction)
    decimate.PreserveTopologyOn()
    decimate.Update()
    decimatedPoly = vtkPolyData()
    decimatedPoly.ShallowCopy(decimate.GetOutput())
    return decimatedPoly


def vtk_write_vtk(mesh_vtk, path):
    """
    https://public.kitware.com/pipermail/vtkusers/2009-May/052012.html
    :param mesh_vtk:
    :param path:
    :return:
    """
    print('    ... write mesh vtk')
    w = vtk.vtkPolyDataWriter()
    w.SetInputData(mesh_vtk)
    w.SetFileName(path)
    w.Write()


def sitk_to_vtk(img, path):
    import itk
    PixelType = img.GetPixelIDValueAsString()
    Dimension = img.GetSize()
    ImageType = itk.Image[PixelType, Dimension]
    reader = itk.ImageFileReader[ImageType].New()
    reader.SetFileName(img)
    itkToVtkFilter = itk.ImageToVTKImageFilter[ImageType].New()
    itkToVtkFilter.SetInput(reader.GetOutput())
    itkToVtkFilter.Update()
    myvtkImageData = itkToVtkFilter.GetOutput()


def _datacheck_peakdetect(x_axis, y_axis):
    if x_axis is None:
        x_axis = range(len(y_axis))

    if len(y_axis) != len(x_axis):
        raise (ValueError,
               'Input vectors y_axis and x_axis must have same length')

    # needs to be a numpy array
    y_axis = np.array(y_axis)
    x_axis = np.array(x_axis)
    return x_axis, y_axis


def peakdetect(y_axis, x_axis=None, lookahead=10, delta=0.1, plot=False):
    """
    Converted from/based on a MATLAB script at:
    http://billauer.co.il/peakdet.html

    function for detecting local maximas and minmias in a signal.
    Discovers peaks by searching for values which are surrounded by lower
    or larger values for maximas and minimas respectively

    keyword arguments:
    y_axis -- A list containg the signal over which to find peaks
    x_axis -- (optional) A x-axis whose values correspond to the y_axis list
        and is used in the return to specify the postion of the peaks. If
        omitted an index of the y_axis is used. (default: None)
    lookahead -- (optional) distance to look ahead from a peak candidate to
        determine if it is the actual peak (default: 200)
        '(sample / period) / f' where '4 >= f >= 1.25' might be a good value
    delta -- (optional) this specifies a minimum difference between a peak and
        the following points, before a peak may be considered a peak. Useful
        to hinder the function from picking up false peaks towards to end of
        the signal. To work well delta should be set to delta >= RMSnoise * 5.
        (default: 0)
            delta function causes a 20% decrease in speed, when omitted
            Correctly used it can double the speed of the function

    return -- two lists [max_peaks, min_peaks] containing the positive and
        negative peaks respectively. Each cell of the lists contains a tupple
        of: (position, peak_value)
        to get the average peak value do: np.mean(max_peaks, 0)[1] on the
        results to unpack one of the lists into x, y coordinates do:
        x, y = zip(*tab)
    """
    max_peaks = []
    min_peaks = []
    dump = []  # Used to pop the first hit which almost always is false

    # check input data
    x_axis, y_axis = _datacheck_peakdetect(x_axis, y_axis)
    # store data length for later use
    length = len(y_axis)

    # perform some checks
    if lookahead < 1:
        raise ValueError("Lookahead must be '1' or above in value")
    if not (np.isscalar(delta) and delta >= 0):
        raise ValueError("delta must be a positive number")

    # maxima and minima candidates are temporarily stored in
    # mx and mn respectively
    mn, mx = np.Inf, -np.Inf

    # Only detect peak if there is 'lookahead' amount of points after it
    for index, (x, y) in enumerate(zip(x_axis[:-lookahead],
                                       y_axis[:-lookahead])):
        if y > mx:
            mx = y
            mxpos = x
        if y < mn:
            mn = y
            mnpos = x

        ####look for max####
        if y < mx - delta and mx != np.Inf:
            # Maxima peak candidate found
            # look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index:index + lookahead].max() < mx:
                try:
                    max_peaks.append([mxpos, mx[0]])
                except:
                    max_peaks.append([mxpos, mx])
                dump.append(True)
                # set algorithm to only find minima now
                mx = np.Inf
                mn = np.Inf
                if index + lookahead >= length:
                    # end is within lookahead no more peaks can be found
                    break
                continue
            # else:  #slows shit down this does
            #    mx = ahead
            #    mxpos = x_axis[np.where(y_axis[index:index+lookahead]==mx)]

        ####look for min####
        if y > mn + delta and mn != -np.Inf:
            # Minima peak candidate found
            # look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index:index + lookahead].min() > mn:
                try:  # added by me (Benjamin 29.05.2017) beacause when mn has only one value (length = 1), mn[0] fails!!
                    min_peaks.append([mnpos, mn[0]])
                except:
                    min_peaks.append([mnpos, mn])
                dump.append(False)
                # set algorithm to only find maxima now
                mn = -np.Inf
                mx = -np.Inf
                if index + lookahead >= length:
                    # end is within lookahead no more peaks can be found
                    break
            # else:  #slows shit down this does
            #    mn = ahead
            #    mnpos = x_axis[np.where(y_axis[index:index+lookahead]==mn)]

    # Remove the false hit on the first value of the y_axis
    try:
        if dump[0]:
            max_peaks.pop(0)
        else:
            min_peaks.pop(0)
        del dump
    except IndexError:
        # no peaks were found, should the function return empty lists?
        pass
    max_peaks = np.array(max_peaks)
    min_peaks = np.array(min_peaks)
    if plot == True:
        plot_peaks(y_axis, min_peaks, max_peaks)
    return max_peaks, min_peaks  # return [max_peaks, min_peaks]   -> changed to have them separated (BV) 13.04.2017


def plot_peaks(y, min_peaks, max_peaks):
    plt.figure()
    plt.plot(np.arange(len(y)), y)
#    max peaks:
    plt.plot(max_peaks[:,0], max_peaks[:,1], 'or')
#    minpeaks:
    plt.plot(min_peaks[:,0], min_peaks[:,1], 'ob')
    plt.show()

# ************************************************************************************************************************************************************************************
# ************************************************************************************************************************************************************************************
