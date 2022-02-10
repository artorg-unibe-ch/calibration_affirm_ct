"""
Alice Dudle
9.2.2022

"""

import numpy as np
import cv2
import matplotlib.pyplot as plt


def detect_circles(img, minRadius, maxRadius, dist_min_btw_circle, x0, x1, y0, y1, z0=0, z1=None,
                   sort=0, filter=True):
    """
    Detect circles within an image
    :param img: 3D numpy array with image intensities
    :param minRadius, maxRadius: minimal and maximal radii of the circles to be detected
    :param dist_min_btw_circle: minimal distance between 2 circles to be detected
    :param x0, x1, y0, y1, z0, z1: range within which the circles should be detected
    :param sort: axis along which circles are aligned (large phantom: 0, small phantom: 1)
    :return: list and number of detected circles
    """
    reso = 1
    p1 = 56
    p2 = 12
    if z1 is None:
        z1 = img.shape[2]
    list_circles = np.zeros((1, 4))
    slice_nb_range = np.arange(z0, z1)
    if filter:
        j = 0
        while img.shape[2] > j * 490:
            img[:, :, 490*j:min(img.shape[2], (j+1)*490)] = cv2.GaussianBlur(img[:, :, 490*j:min(img.shape[2], (j+1)*490)],
                                                                             (5, 5), 0)
            j += 1

    """Iterate over slices and detect circles"""
    for slice_nb in slice_nb_range:
        found_circles = cv2.HoughCircles(img[x0:x1, y0: y1, slice_nb],
                                         cv2.HOUGH_GRADIENT, reso, dist_min_btw_circle,
                                         param1=p1, param2=p2,
                                         minRadius=minRadius, maxRadius=maxRadius)
        if (found_circles is None) & (slice_nb < 10):
            print("No circles found in slice nb" + str(slice_nb))
            continue
        elif (found_circles is None) & (len(list_circles) > 1):
            continue
        elif found_circles is None:
            raise ValueError("No circles found in slice nb" + str(slice_nb))

        found_circles = found_circles[0, ...]
        found_circles = np.concatenate((found_circles[:, 1:2], found_circles[:, 0:1],
                                        slice_nb * np.ones((found_circles.shape[0], 1)),
                                        found_circles[:, 2:3]), axis=1)
        # if slice_nb in [0, 30, 60, 90, 120]:
        #     color_list = ['y', 'm', 'r', 'g', 'b',  'c', 'y', 'm', 'r', 'g', 'b']
        #     fig, ax = plt.subplots(1, 1)
        #     ax.imshow(img[x0:x1, y0:y1, slice_nb].transpose(1, 0))
        #     for j in range(found_circles.shape[0]):
        #         center_x = found_circles[j, 0]
        #         center_y = found_circles[j, 1]
        #         radius = found_circles[j, 3]
        #         # print("Radius: ", radius)
        #         circle = plt.Circle((center_x, center_y), radius, color=color_list[j], fill=False)
        #         ax.add_artist(circle)
        #     plt.show()

        found_circles[:, 0] += x0
        found_circles[:, 1] += y0
        list_circles = np.append(list_circles, found_circles, axis=0)

    list_circles = list_circles[1:, ...]
    if len(list_circles) < (z1 - z0) / 2:
        raise ValueError("Circles detected in less than half of the slices")
    list_circles = list_circles[np.argsort(list_circles[:, sort])]
    # print(list_circles)

    cleaned_list_circles = []
    average_list = []
    min_i = 0
    max_i = 1

    """Group slices into groups according to their positions, and remove outliers"""
    for i in range(1, list_circles.shape[0]):
        dx = list_circles[i, 0] - list_circles[i - 1, 0]
        dy = list_circles[i, 1] - list_circles[i - 1, 1]
        if dx * dx + dy * dy < minRadius * minRadius:
            max_i = i + 1
        else:
            if max_i - min_i > len(slice_nb_range)/3:
                temp = list_circles[min_i:max_i, :]
                cleaned_list_circles.append(temp[np.argsort(temp[:, 2])])
                average_list.append(temp.mean(axis=0))
            min_i = i
            max_i = i + 1
    if max_i - min_i > len(slice_nb_range)/3:
        temp = list_circles[min_i:max_i, :]
        cleaned_list_circles.append(temp[np.argsort(temp[:, 2])])
        average_list.append(temp.mean(axis=0))
    nb_detected = len(average_list)
    print("Number of circles detected: ", nb_detected)
    """Return average circle positions or all circles positions"""
    average_list = np.asarray(average_list)
    return average_list, nb_detected


def compute_last_circle(average_list, img, dx=0.8):
    """
    Given 4 or 5 circles detected within the large phantom, computes the position of 1 or 2 missing ones
    :param average_list: list of detected circles positions
    :return: list of 6 circle positions
    """
    last_z = np.mean(average_list[:, 2])
    last_r = np.min(average_list[:, 3])
    position0 = average_list[0, :].astype(int)
    intensity0 = img[position0[0] - 4: position0[0] + 4, position0[1] - 4: position0[1] + 4, position0[2]].mean()
    position1 = average_list[1, :].astype(int)
    intensity1 = img[position1[0] - 4: position1[0] + 4, position1[1] - 4: position1[1] + 4, position1[2]].mean()

    """Compute position 1 or 2 last circles depending on position of detected circles"""
    if intensity1 > intensity0:
        v = average_list[-4, :] - average_list[-3, :]
        if average_list.shape[0] < 5:
            sec_last_y = average_list[-2, 1] + 3 * v[1]
            sec_last_x = average_list[-2, 0] + 3 * v[0]
            average_list = np.append([[sec_last_x, sec_last_y, last_z, last_r]], average_list, axis=0)
        if average_list.shape[0] < 6:
            last_y = average_list[-1, 1] + 5 * v[1]
            last_x = average_list[-1, 0] + 5 * v[0]
            average_list = np.append([[last_x, last_y, last_z, last_r]], average_list, axis=0)
    else:
        v = average_list[3, :] - average_list[2, :]
        if average_list.shape[0] < 5:
            sec_last_y = average_list[1, 1] + 3 * v[1]
            sec_last_x = average_list[1, 0] + 3 * v[0]
            average_list = np.append(average_list, [[sec_last_x, sec_last_y, last_z, last_r]], axis=0)
        if average_list.shape[0] < 6:
            last_y = average_list[0, 1] + 5 * v[1]
            last_x = average_list[0, 0] + 5 * v[0]
            average_list = np.append(average_list, [[last_x, last_y, last_z, last_r]], axis=0)
    return average_list


def plot_example(img_raw, average_list, z0=0, z1=None, nb=4, save=None):
    """
    Plot some example slices with detected circles
    :param img_raw: 3D numpy array with image intensities
    :param average_list: list of circles positions
    :param z0, z1: range of slices from which to pick examples
    :param nb: number of examples
    :param save: path for saving image
    """
    color_list = ['y', 'm', 'r', 'g', 'b',  'c', 'y', 'm', 'r', 'g', 'b']
    if z1 is None:
        z1 = img_raw.shape[2]
    fig, ax = plt.subplots(1, nb, figsize=(16, 4))
    x = np.random.randint(z0, z1, nb)
    for i in range(nb):
        ax[i].imshow(img_raw[:, :, x[i]].transpose(1, 0))
        for j in range(average_list.shape[0]):
            center_x = average_list[j, 0]
            center_y = average_list[j, 1]
            radius = average_list[j, 3]
            circle = plt.Circle((center_x, center_y), radius, color=color_list[j], fill=False)
            ax[i].add_artist(circle)

    if save is not None:
        plt.savefig(save)
    plt.show()


def plot_final(img_raw, average_list, z0=0, z1=None, save=None, show_plot=True):
    """
    Plot all detected circles on an example slice.
    The colors should match those of the points on the linear regression in calibration_general.py
    :param img_raw: 3D numpy array with image intensites
    :param average_list: list of circles positions
    :param z0, z1: slice range from which to pick example slice
    :param save: path for saving image
    :param show_plot: if True, shows plot
    """
    color_list = ['g', 'g', 'g', 'g', 'g', 'g', 'orange', 'orange', 'r', 'r', 'r']
    if z1 is None:
        z1 = img_raw.shape[2]
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    x = np.random.randint(z0, z1)
    ax.imshow(img_raw[:, :, x].transpose(1, 0), cmap="gray")
    for j in range(average_list.shape[0]):
        center_x = average_list[j, 0]
        center_y = average_list[j, 1]
        radius = average_list[j, 3]
        circle = plt.Circle((center_x, center_y), radius, color=color_list[j], fill=False)
        ax.add_artist(circle)
    if save is not None:
        plt.savefig(save)
    if show_plot:
        plt.show()
    else:
        plt.close()


def compute_average(img, average_list, z0=0, z1=None, order=True, save_detail=None):
    """
    Compute intensity values for a list of circle positions
    :param img: 3D numpy array with image intensities
    :param average_list: list of circle positions
    :param z0, z1: range of slices where intensities should be computed
    :param order: if True, intensities are sorted in increasing order
    :param save_detail: save plots with evolution of intensities over slice range
    :return: average intensity and standard deviation values for each calibration tube
    """

    x_max, y_max, z_max = img.shape
    if z1 is None:
        z1 = z_max
    X, Y, Z = np.ogrid[0: x_max, 0: y_max, 0: z_max]
    radius_ratio = 0.5
    average_values = np.zeros(average_list.shape[0])
    std_values = np.zeros(average_list.shape[0])
    list_values = np.zeros((average_list.shape[0], z1 - z0))
    list_std_values = np.zeros((average_list.shape[0], z1 - z0))

    for i in range(average_list.shape[0]):
        center_x = np.rint(average_list[i, 0]).astype(int)
        center_y = np.rint(average_list[i, 1]).astype(int)
        radius = average_list[i, 3] * radius_ratio
        mask = ((X - center_x)**2 + (Y - center_y)**2 < radius**2) & (Z - z0 >= 0) & (z1 - Z > 0)
        average_values[i] = np.average(img[mask])
        std_values[i] = np.std(img[mask])
        if save_detail is not None:
            for z in range(z0, z1):
                mask = ((X - center_x)**2 + (Y - center_y)**2 < radius**2) & (Z == z)
                list_values[i, z - z0] = np.average(img[mask])
                list_std_values[i, z - z0] = np.std(img[mask])

    if save_detail is not None:
        x = np.arange(z1 - z0)
        fig, ax = plt.subplots(1, 2, figsize=(16, 8))
        for i in range(6):
            ax[0].plot(x, list_values[i, :], '.')
            ax[1].plot(x, list_std_values[i, :], '.')
        plt.savefig(save_detail)
        # plt.show()
    if order:
        average_values = np.sort(average_values)[::-1]
    return average_values, std_values


def compute_average_hip(img, average_circles, z0=0, z1=None, save_detail=None, show_plot=False):
    """
    Compute intensity values for a list of circle positions for the hip phantom
    :param img: 3D numpy array with image intensities
    :param average_circles: list of circle positions (2)
    :param z0, z1: slice range where intensity should be computed
    :param save_detail: save plots with evolution of intensities over slice range
    :param show_plot: for debugging
    :return: average intensities for each density region in a (3, 2) array
    """
    x_max, y_max, z_max = img.shape
    if z1 is None:
        z1 = z_max
    X, Y, Z = np.ogrid[0: x_max, 0: y_max, 0: z_max]
    intensity_values = np.zeros((z1 - z0, average_circles.shape[0]))
    list_std_values = np.zeros((z1 - z0, average_circles.shape[0]))
    radius_ratio = 0.5
    for i in range(average_circles.shape[0]):
        center_x = np.rint(average_circles[i, 0]).astype(int)
        center_y = np.rint(average_circles[i, 1]).astype(int)
        radius = average_circles[i, 3] * radius_ratio
        for z in range(z1 - z0):
            mask = ((X - center_x)**2 + (Y - center_y)**2 < radius**2) & (Z == z0 + z)
            intensity_values[z, i] = np.average(img[mask])
            list_std_values[z, i] = np.std(img[mask])

    if save_detail is not None:
        x = np.arange(z1 - z0)
        fig, ax = plt.subplots(1, 2, figsize=(16, 8))
        for i in range(2):
            ax[0].plot(x, intensity_values[:, i], '.')
            ax[1].plot(x, list_std_values[:, i], '.')
        plt.savefig(save_detail)
        # plt.show()

    max_diff = 4
    min_length = z_max / 6
    if intensity_values.shape[0] < 20:
        min_length = 2
    average_values = [0, 0, 0, 0]
    while (len(average_values) > 3) | (len(average_values) < 3):
        if max_diff > 15:
            raise ValueError("Average values can't be computed. Number of groups: ", len(average_values))
        max_diff += 1
        average_values = []
        std_values = []
        list_min_values = []
        list_max_values = []
        std = np.zeros(2)
        min_i = 0
        max_i = 1
        for i in range(1, intensity_values.shape[0]):
            dx = abs(intensity_values[i, 0] - intensity_values[i - 1, 0])
            dy = abs(intensity_values[i, 1] - intensity_values[i - 1, 1])
            if (dx < max_diff) & (dy < max_diff):
                max_i = i + 1
            else:
                if max_i - min_i > min_length:
                    average_values.append(np.mean(intensity_values[min_i:max_i], axis=0))
                    list_min_values.append(min_i)
                    list_max_values.append(max_i)
                    for j in range(2):
                        center_x = np.rint(average_circles[j, 0]).astype(int)
                        center_y = np.rint(average_circles[j, 1]).astype(int)
                        radius = average_circles[j, 3] * radius_ratio
                        mask = ((X - center_x)**2 + (Y - center_y)**2 < radius**2) & \
                               (Z >= z0 + min_i) & (Z < z0 + max_i)
                        std[j] = np.std(img[mask])
                    std_values.append([std[0], std[1]])
                min_i = i
                max_i = i + 1
        if max_i - min_i > min_length:
            average_values.append(np.mean(intensity_values[min_i:max_i], axis=0))
            list_min_values.append(min_i)
            list_max_values.append(max_i)
            for j in range(2):
                center_x = np.rint(average_circles[j, 0]).astype(int)
                center_y = np.rint(average_circles[j, 1]).astype(int)
                radius = average_circles[j, 3] * radius_ratio
                mask = ((X - center_x)**2 + (Y - center_y)**2 < radius**2) & (Z >= z0 + min_i) & \
                       (Z < z0 + max_i)
                std[j] = np.std(img[mask])
            std_values.append([std[0], std[1]])
        if show_plot:
            print(list_max_values)
            plt.plot(intensity_values, '.')
            for x in list_min_values:
                plt.axvline(x=x, color="g")
            for x in list_max_values:
                plt.axvline(x=x, color="r")
            plt.show()

    average_values = np.asarray(average_values)
    if average_values[0, 0] - average_values[1, 0] < 0:
        average_values[:, 0] = average_values[::-1, 0]
    elif average_values[0, 1] - average_values[1, 1] < 0:
        average_values[:, 1] = average_values[::-1, 1]
    return average_values, std_values
