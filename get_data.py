# -*- coding: utf-8 -*-
'''
Author: Lorces
Get data from figures in paper.
'''


import cv2
import matplotlib.pyplot as plt
import os


def on_buttondown(event, x, y,flags, param):
    cv2.imshow("figure", img)
    if event == cv2.EVENT_LBUTTONDOWN:
        fig_x.append(x)
        fig_y.append(y)
        cv2.circle(img, (x, y), 5, (0, 0, 255), thickness=-1)
    if event == cv2.EVENT_RBUTTONDOWN:
        ax_x.append(x)
        ax_y.append(y)
        cv2.circle(img, (x, y), 7, (255, 0, 0), thickness=-1)


def get_pic_data(X0,Y0,XN,YN):
    try:
        x0,y0 = ax_x[0],ax_y[0]
        xn,yn = ax_x[1],ax_y[2]
    except:
        print('Please click the right points on coordinate axis!')
        return

    xl, yl = xn - x0, yn - y0
    data_x,data_y = [],[]
    for i in range(len(fig_x)):
        xi = X0 + (XN - X0) * (fig_x[i] - x0) / xl
        yi = Y0 + (YN - Y0) * (fig_y[i] - y0) / yl
        data_x.append(xi)
        data_y.append(yi)

    return data_x,data_y


def check_fig(data_x, data_y):
    try:
        plt.figure()
        plt.scatter(data_x,data_y, c='r')
        plt.plot(data_x, data_y, 'b-',)
        plt.show()
    except:
        print('Please check the input list!')
        return


def save_data(filename,data_x, data_y):
    with open(filename,'w') as f:
        for i in range(len(data_x)):
            f.write('%16.8f' % data_x[i] + '%16.8f' % data_y[i] + '\n')


if __name__ == '__main__':
    AP = os.getcwd().replace('\\', '/')

    pic = str(AP + '/Experiment/RF_umid.png')
    filename = str(AP + '/Experiment/umid_rf4.dat')
    ax_x, ax_y = [], []
    fig_x, fig_y = [], []

    img = cv2.imread(pic)
    cv2.namedWindow('figure', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.setMouseCallback("figure", on_buttondown)
    cv2.imshow("figure", img)
    cv2.waitKey(0)

    X0, Y0, XN, YN = -5, -0.4, 15, 1

    try:
        X, Y = get_pic_data(X0, Y0, XN, YN)
        check_fig(X, Y)
        save_data(filename, X, Y)
    except:
        print('Please check data of X and Y!')

