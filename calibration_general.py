"""
Alice Dudle
10.2.2022
"""

import os
import glob
import numpy as np
import SimpleITK as sitk
import copy
import Swissknife
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
from calibration_fct import detect_circles, plot_example, compute_last_circle, compute_average, \
    compute_average_hip, plot_final

"""INITIALIZATION"""

"""Path to folder containing calibration scans"""
calib_folder = "/home/aliced/Desktop/Images/Measurements_Phantom/"

"""Interactive part: complete calibation path & choose scans to analyze"""
folder = input("Complete the calibration folder path : "+calib_folder)

list_dir = os.listdir(os.path.join(calib_folder, folder))
list_dir_text = ""
for i in range(len(list_dir)):
    list_dir_text = list_dir_text + str(i) + ": " + list_dir[i] + "\n"
test = input("This folder includes the following files: \n" + list_dir_text +
             "\n Do you want to analyze all of them? (y/n/p: path incomplete) \n")
while test not in {"y", "n", "p"}:
    test = input("Sorry, answer unknown. Try y, n or p. \n")
list_dir = np.asarray(list_dir)
while test == "p":
    subfolder = input("Which subfolder contains the calibration images? (0, 1, 2, ...) \n")
    folder = os.path.join(folder, list_dir[int(subfolder)])
    list_dir = os.listdir(os.path.join(calib_folder, folder))
    list_dir_text = ""
    for i in range(len(list_dir)):
        list_dir_text = list_dir_text + str(i) + ": " + list_dir[i] + "\n"
    print("Folder: ", folder)
    test = input("This folder includes the following files: \n" + list_dir_text +
                 "\n Do you want to analyze all of them? (y/n/p: path incomplete) \n")
    while test not in {"y", "n", "p"}:
        test = input("Sorry, answer unknown. Try y, n or p. \n")
    list_dir = np.asarray(list_dir)

site = folder.split("/")[0]
scanner = folder.split("/")[1]
print(scanner)

if test == "y":
    new_list_dir = list_dir
while test == "n":
    new_list = input("Which ones do you want to analyze? \n"
                     "Please enter a list of positions (e.g. 3, 5, 7) \n")
    new_list = new_list.replace("[", "").replace("]", "").split(",")
    new_list = np.asarray(new_list).astype(int)
    new_list_dir = list_dir[new_list]
    test = input("Is this the list of folders you want to analyze?\n"+str(new_list_dir)+"\n (y/n)\n")
small_phantom = input("Is the small phantom present in the images? (y/n) \n")


"""Prepare dataframe columns"""
if small_phantom == "y":
    small_phantom = True
    columns = ["Path", "Study date", "Study time", "Series time",
               "LP1 avg", "LP1 std", "LP2 avg", "LP2 std", "LP3 avg", "LP3 std",
               "LP4 avg", "LP4 std", "LP5 avg", "LP5 std", "LP6 avg", "LP6 std",
               "HP1 avg", "HP1 std", "HP2 avg", "HP2 std", "HP3 avg", "HP3 std",
               "SP1 avg", "SP1 std", "SP2 avg", "SP2 std", "SP3 avg", "SP3 std",
               "LP slope", "LP intercept", "LP correl", "HP slope", "HP intercept", "HP correl",
               "SP slope", "SP intercept", "SP correl", "All slope", "All intercept", "All correl"]
else:
    small_phantom = False
    columns = ["Path", "Study date", "Study time", "Series time",
               "LP1 avg", "LP1 std", "LP2 avg", "LP2 std", "LP3 avg", "LP3 std",
               "LP4 avg", "LP4 std", "LP5 avg", "LP5 std", "LP6 avg", "LP6 std",
               "HP1 avg", "HP1 std", "HP2 avg", "HP2 std", "HP3 avg", "HP3 std",
               "LP slope", "LP intercept", "LP correl", "HP slope", "HP intercept", "HP correl",
               "All slope", "All intercept", "All correl"]

"""Choose between seeing some plots as a check (yes), no plots (no), or all plots to debug (d)"""
show_plots = input("Do you want to see the plots as a safety check? (y/n/d: debugging) \n")
if show_plots == "y":
    show_plots = False
    show_plots_final = True
elif show_plots == "d":
    show_plots = True
    show_plots_final = True
else:
    show_plots = False
    show_plots_final = False

df_path = os.path.join(calib_folder, folder, "new_list_calib_img.csv")
if os.path.isfile(df_path):
    df = pd.read_csv(df_path, header=0, index_col=0)
else:
    df = pd.DataFrame(index=new_list_dir, columns=columns)


"""Iteration over calibration images"""
for row in new_list_dir:
    print("Analyzing: ", row)
    filepath = os.path.join(calib_folder, folder, row)
    df.loc[row, "Path"] = filepath

    """Get metadata from the header of one slice"""
    reader = sitk.ImageFileReader()
    dicom_files = glob.glob(filepath + "/*.dcm")
    if (len(dicom_files) < 5) & (scanner != "Trois_Chenes"):
        print("Skip: " + str(len(dicom_files)) + " dicom files.")
        df = df.drop(index=row)
        continue
    reader.SetFileName(os.path.join(filepath, dicom_files[0]))
    reader.ReadImageInformation()
    df.loc[row, "Study date"] = reader.GetMetaData("0008|0020")
    df.loc[row, "Study time"] = reader.GetMetaData("0008|0030")
    df.loc[row, "Series time"] = reader.GetMetaData("0008|0031")

    com = reader.GetMetaData("0020|4000")
    l = len(com)
    while True:
        try:
            print("Image comments: ", reader.GetMetaData("0020|4000")[:l])
            df.loc[row, "Image comments"] = reader.GetMetaData("0020|4000")[:l]
            break
        except UnicodeEncodeError as err:
            print("Unicode error", err, "for l=", l)
            l -= 1

    if (scanner == "Trois_Chenes") | (scanner == "Cantonal_4"):
        df.loc[row, "Series number"] = reader.GetMetaData("0020|0011")
        df.loc[row, "Acquisition number"] = reader.GetMetaData("0020|0012")

    if scanner != "Trois_Chenes":
        df.loc[row, "Pitch"] = reader.GetMetaData("0018|9311")
        df.loc[row, "Voltage"] = reader.GetMetaData("0018|0060")
        df.loc[row, "Kernel"] = reader.GetMetaData("0018|1210")
        df.loc[row, "Slice thickness"] = reader.GetMetaData("0018|0050")
        df.loc[row, "Table height"] = reader.GetMetaData("0018|1130")

    df.loc[row, "Body part examined"] = reader.GetMetaData("0018|0015")
    df.to_csv(df_path)

    """Read whole image"""
    img_raw_sitk, additData, is_it_downscaled, file_type = Swissknife.read(filepath, forced='yes')
    # reader = sitk.ImageFileReader()
    # filename = os.listdir(filepath)[0]
    # reader.SetFileName(os.path.join(filepath, filename))
    # img_raw_sitk = reader.Execute()
    x_max, y_max, z_max = img_raw_sitk.GetSize()
    dx, dy, dz = img_raw_sitk.GetSpacing()
    img_raw_np = sitk.GetArrayFromImage(img_raw_sitk).transpose(2, 1, 0)

    """Prepare image for circle detection: clip and convert to uint8"""
    img_clip_np = copy.deepcopy(img_raw_np)
    img_clip_np = np.clip(img_clip_np, -100, 700)
    img_clip_np -= img_clip_np.min()
    img_clip_np = img_clip_np / 10
    img_8bit_np = img_clip_np.astype('uint8')

    """LARGE PHANTOM"""
    if site == "Bern":
        HU_values = [797.0, 601.2, 399.6, 201.4, 100.8, 0]
    elif site == "HUG":
        HU_values = [799.6, 596.8, 397.4, 197.0, 101.4, 0]
    elif site == "Forensic":
        HU_values = [790.3, 599.8, 400.8, 204.2, 101.4, 0]
    else:
        raise ValueError("The site name is unknown:", site)
    minRadius = int(np.rint(8/dx))
    maxRadius = int(np.rint(10/dx))
    dist_min_btw_circle = int(np.rint(18/dx))
    x0 = int(0.25 * x_max)
    x1 = int(0.75 * x_max)
    y0 = int(0.65 * y_max)
    y1 = int(0.80 * y_max)
    z0 = 0

    if show_plots:
        Swissknife.plot_special(img_8bit_np[x0:x1, y0:y1, :])
        plt.show()

    """Detect 4-5 circles with highest density"""
    print("LARGE PHANTOM")
    average_circles, nb_detected = detect_circles(img_8bit_np, minRadius, maxRadius, dist_min_btw_circle,
                                                  x0, x1, y0, y1, z0, filter=True)

    """Compute position of missing circles"""
    average_circles = compute_last_circle(average_circles, img_8bit_np, dx)
    nb_detected = average_circles.shape[0]
    if show_plots:
        plot_example(img_raw_np, average_circles, nb=4)

    """Compute intensity within circles"""
    average_values, std_values, = compute_average(img_raw_np, average_circles, z0=z0)
    for i in range(6):
        df.loc[row, "LP"+str(i+1)+" avg"] = average_values[i]
        df.loc[row, "LP"+str(i+1)+" std"] = std_values[i]

    """Linear regression between intensity and density values"""
    slope, intercept, r_value, p_value, std_err = stats.linregress(average_values[:nb_detected],
                                                                   HU_values[:nb_detected])
    df.loc[row, "LP slope"] = slope
    df.loc[row, "LP intercept"] = intercept
    df.loc[row, "LP correl"] = r_value

    """Plot regression"""
    if show_plots:
        x = np.arange(average_values.min() - 50, average_values.max() + 50, 10)
        fig, ax = plt.subplots(1, 1)
        ax.plot(x, intercept + slope * x)
        ax.errorbar(average_values[:nb_detected], HU_values[:nb_detected], xerr=std_values[:nb_detected], fmt='.')
        ax.set_ylabel("HU value")
        ax.set_xlabel("Image intensity")
        plt.show()

    """HIP PHANTOM"""
    if site == "Bern":
        HU_values_hip = [194.5, 97.7, 47.9]
    elif site == "HUG":
        HU_values_hip = [202.0, 104.2, 51.7]
    else:
        raise ValueError("The site name is unknown:", site)
    minRadius = int(np.rint(12/dx))
    maxRadius = int(np.rint(15/dx))
    dist_min_btw_circle = int(np.rint(20/dx))
    x0 = int(0.1 * x_max)
    x1 = int(0.9 * x_max)
    y0 = int(0.4 * y_max)
    y1 = int(0.6 * y_max)

    if show_plots:
        Swissknife.plot_special(img_8bit_np[x0:x1, y0:y1, :])
        plt.show()

    """Detect two circles inside hip phantom"""
    print("HIP PHANTOM")
    average_circles2, nb_detected2 = detect_circles(img_8bit_np, minRadius, maxRadius, dist_min_btw_circle,
                                                    x0, x1, y0, y1, z0)
    if show_plots:
        plot_example(img_raw_np, average_circles2, nb=4)

    """Compute intensity values within circles (3 different values for each circle)"""
    average_values2, std_values2 = compute_average_hip(img_raw_np, average_circles2, z0=z0,
                                                       show_plot=show_plots)
    for i in range(3):
        df.loc[row, "HP"+str(i+1)+" avg"] = np.mean(average_values2[i])
        df.loc[row, "HP"+str(i+1)+" std"] = np.mean(std_values2[i])

    """Linear regression between intensity and density values"""
    slope, intercept, r_value, p_value, std_err = stats.linregress(np.mean(average_values2, axis=1),
                                                                   HU_values_hip)
    df.loc[row, "HP slope"] = slope
    df.loc[row, "HP intercept"] = intercept
    df.loc[row, "HP correl"] = r_value

    """Plot linear regression"""
    if show_plots:
        fig, ax = plt.subplots(1, 1)
        x = np.arange(average_values2.min() - 50, average_values2.max() + 50, 10)
        ax.plot(x, intercept + slope * x)
        ax.errorbar(np.mean(average_values2, axis=1), HU_values_hip, xerr=np.mean(std_values2, axis=1), fmt='.')
        ax.set_ylabel("HU value")
        ax.set_xlabel("Image intensity")
        plt.show()

    if small_phantom:
        """SMALL PHANTOM (Next to hip)"""
        if scanner == "ASH":
            HU_values_small = [797.8, 551.4, 297.2]
        elif scanner == "CT1":
            HU_values_small = [799.0, 554.2, 298.8]
        elif scanner == "CT2":
            HU_values_small = [807.2, 560.2, 303.4]
        else:
            raise ValueError("Scanner name unknown")
        minRadius = 3
        maxRadius = 7
        dist_min_btw_circle = 4
        x0 = int(0.85 * x_max)
        x1 = int(0.95 * x_max)
        y0 = int(0.55 * y_max)
        y1 = int(0.68 * y_max)

        if show_plots:
            Swissknife.plot_special(img_8bit_np[x0:x1, y0:y1, :])
            plt.show()

        """Detect circles (probably only 2 with highest intensity)"""
        print("SMALL PHANTOM")
        average_circles3, nb_detected3 = detect_circles(img_8bit_np, minRadius, maxRadius, dist_min_btw_circle,
                                                        x0, x1, y0, y1, z0, sort=1, filter=False)
        """Compute position of third circle depending on how the phantom is positioned"""
        average_values3, std_values3 = compute_average(img_raw_np, average_circles3, order=False)
        if (len(average_values3) < 3) & (average_values3[0] > average_values3[1]):
            c3 = 2 * average_circles3[0:1, :] - average_circles3[1:2, :]
            c3[0, 3] = np.min(average_circles3[:, 3])
            average_circles3 = np.append(average_circles3, c3, axis=0)
        elif (len(average_values3) < 3) & (average_values3[0] < average_values3[1]):
            c3 = 2 * average_circles3[1:2, :] - average_circles3[0:1, :]
            c3[0, 3] = np.min(average_circles3[:, 3])
            average_circles3 = np.append(c3, average_circles3, axis=0)
        if show_plots:
            plot_example(img_raw_np, average_circles3, nb=4)

        """Compute intensity values within circles"""
        average_values3, std_values3 = compute_average(img_raw_np, average_circles3, z0=z0, order=True)
        for i in range(3):
            df.loc[row, "SP"+str(i+1)+" avg"] = average_values3[i]
            df.loc[row, "SP"+str(i+1)+" std"] = std_values3[i]

        """Linear regression between intensity and density values"""
        slope3, intercept3, r_value3, p_value3, std_err3 = stats.linregress(average_values3, HU_values_small)
        df.loc[row, "SP slope"] = slope3
        df.loc[row, "SP intercept"] = intercept3
        df.loc[row, "SP correl"] = r_value3

        """Plot linear regression"""
        if show_plots:
            fig, ax = plt.subplots(1, 1)
            x = np.arange(average_values3.min() - 50, average_values3.max() + 50, 10)
            ax.plot(x, intercept3 + slope3 * x)
            ax.errorbar(average_values3, HU_values_small, xerr=std_values3, fmt='.')
            ax.set_ylabel("HU value")
            ax.set_xlabel("Image intensity")
            plt.show()

    """ALL PHANTOMS"""
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    if small_phantom:
        all_values = np.concatenate((average_values,  np.mean(average_values2, axis=1),
                                     average_values3), axis=0)
        all_std = np.concatenate((std_values, np.mean(std_values2, axis=1), std_values3), axis=0)
        all_HU_values = np.concatenate((HU_values, HU_values_hip, HU_values_small), axis=0)
    else:
        all_values = np.concatenate((average_values,  np.mean(average_values2, axis=1)), axis=0)
        all_std = np.concatenate((std_values, np.mean(std_values2, axis=1)), axis=0)
        all_HU_values = np.concatenate((HU_values, HU_values_hip), axis=0)

    """Linear regression with all phantoms together"""
    slope, intercept, r_value, p_value, std_err = stats.linregress(all_values, all_HU_values)
    df.loc[row, "All slope"] = slope
    df.loc[row, "All intercept"] = intercept
    df.loc[row, "All correl"] = r_value

    """Plot linear regression and save as png"""
    x = np.arange(all_values.min() - 50, all_values.max() + 50, 10)
    ax.plot(x, intercept + slope * x)
    ax.errorbar(average_values, HU_values, xerr=std_values, fmt='g.')
    ax.errorbar(np.mean(average_values2, axis=1), HU_values_hip,
                xerr=np.mean(std_values2, axis=1), fmt='.')
    if small_phantom:
        ax.errorbar(average_values3, HU_values_small, xerr=std_values3, fmt='r.')
        ax.legend(["Lin reg", "Large", "Hip", "Small"])
    else:
        ax.legend(["Lin reg", "Large", "Hip"], loc=0)
    ax.set_ylabel("Density [mg/cm3]")
    ax.set_xlabel("Image intensity")
    ax.grid()
    ax.set_title("Calibration curve")
    plt.savefig(os.path.join(filepath, "lr_all_phant.png"))
    if show_plots_final:
        plt.show()
    else:
        plt.close()

    """Plot an example slice with detected circles and save as png"""
    if small_phantom:
        plot_final(img_raw_np, np.concatenate([average_circles, average_circles2, average_circles3]),
                   save=os.path.join(filepath, "plot_ex_all.png"), show_plot=show_plots_final)
    else:
        plot_final(img_raw_np, np.concatenate([average_circles, average_circles2]),
                   save=os.path.join(filepath, "plot_ex_all.png"), show_plot=show_plots_final)
    df.sort_values(by=["Series time"], inplace=True)
    df.to_csv(df_path)
