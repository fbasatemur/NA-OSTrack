import pylab as plt
import math
import numpy as np


def display(self, title, fig_title_dict, cmaps, size = 4, plt_func = plt.show):
        len_dict = len(fig_title_dict)
        get_row_col_size = lambda _len_dict, _size: (math.ceil(_len_dict / _size), min(_len_dict, _size))
        (rows, cols) = get_row_col_size(len_dict, size)
        dis_fig = plt.figure()
        dis_fig.suptitle(title)

        for (title, fig, c_map, i) in zip(fig_title_dict.keys(), fig_title_dict.values(), cmaps, range(1, cols*rows +1)):
            dis_fig.add_subplot(rows, cols, i).set_title(title)
            plt.imshow(fig, cmap=None)
            if c_map:
                plt.colorbar()
            plt_func()


def display_3d(self, fig, plt_func=plt.show):
        fig_shape = fig.shape
        x = np.outer(np.linspace(0, fig_shape[0], fig_shape[0]), np.ones(fig_shape[0]))
        y = x.copy().T # transpose
        z = (fig)
        
        # Creating figure
        fig = plt.figure(figsize =(7, 5))
        ax = plt.axes(projection ='3d')
        
        # Creating plot
        ax.plot_surface(x, y, z)
        
        # show plot
        plt_func()