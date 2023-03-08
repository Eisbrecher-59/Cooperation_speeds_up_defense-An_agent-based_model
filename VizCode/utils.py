import os
import sys
import time
import math
import glob
import time
import random
import string
import os.path
import sys, os
import itertools
import subprocess
from re import search
import multiprocessing
import concurrent.futures

import numpy as np
import scipy as sc
import pandas as pd
import seaborn as sns
import scipy.interpolate
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits import mplot3d
import matplotlib.colors as colors
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from matplotlib.gridspec import SubplotSpec
from matplotlib.colors import ListedColormap
from matplotlib.legend_handler import HandlerPatch



from io import StringIO
from matplotlib import cm


def Grid_Transform_A(Grids, t, GridL):
    Transform = Grids.iloc[[t]]

    Initial = Transform.to_string()
    Initial = Initial.split("[")[-1]
    Initial = Initial.split("]")[0]

    # Initial = Initial.translate({ord(c): None for c in string.whitespace})
    Initial = Initial.replace("(", "")
    Initial = Initial.replace(")", "")
    Initial = Initial.replace("-1", "x")
    # print(Initial)


    StrArr = []
    StrArr2 = []
    ListCoord_x = []
    ListCoord_y = []
    ListCoord_effect = []


    k = 0

    for idi, i in enumerate(Initial):
        if i.isdigit() and Initial[idi + 1].isdigit():
            i = i + Initial[idi + 1]
            StrArr.append(i)
        elif i.isdigit():
            StrArr.append(i)
        elif i == "x":
            StrArr.append("x")

    for idi, i in enumerate(StrArr):
        if idi == 0:
            StrArr2.append(i)
        else:
            if len(StrArr[idi - 1]) != 2:
                StrArr2.append(i)

    for idi, i in enumerate(StrArr2):
        if k % 3 == 0:
            ListCoord_x.append(int(i))
        elif k % 3 == 1:
            ListCoord_y.append(int(i))
        elif k % 3 == 2:
            if i == "x":
                ListCoord_effect.append(float(0))
            else:
                ListCoord_effect.append(float(1))
        # print(ListCoord_x)
        # print(ListCoord_y)
        # print(ListCoord_effect)
        k += 1

    Grid_Matrix = np.zeros((GridL, GridL))
    for idx, x in enumerate(ListCoord_x):
        Grid_Matrix[ListCoord_y[idx],x] = ListCoord_effect[idx]
    
    return Grid_Matrix, ListCoord_x, ListCoord_y, ListCoord_effect

def Grid_Transform_B(Grids, t, GridL):
    Transform = Grids.iloc[[t]]

    Initial = Transform.to_string()
    Initial = Initial.split("[")[-1]
    Initial = Initial.split("]")[0]

    # Initial = Initial.translate({ord(c): None for c in string.whitespace})
    Initial = Initial.replace("(", "")
    Initial = Initial.replace(")", "")
    Initial = Initial.replace("-1", "x")
    Initial = Initial.replace(".0", "")
    # print(Initial)


    StrArr = []
    StrArr2 = []
    ListCoord_x = []
    ListCoord_y = []
    ListCoord_PD = []


    k = 0

    for idi, i in enumerate(Initial):
        if i.isdigit() and Initial[idi + 1].isdigit():
            i = i + Initial[idi + 1]
            StrArr.append(i)
        elif i.isdigit():
            StrArr.append(i)
        elif i == "x":
            StrArr.append("x")

    for idi, i in enumerate(StrArr):
        if idi == 0:
            StrArr2.append(i)
        else:
            if len(StrArr[idi - 1]) != 2:
                StrArr2.append(i)

    # print(StrArr2)

    for idi, i in enumerate(StrArr2):
        if k % 3 == 0:
            ListCoord_x.append(int(i))
        elif k % 3 == 1:
            ListCoord_y.append(int(i))
        elif k % 3 == 2:
            if i == "x":
                ListCoord_PD.append(int(-1))
            elif i == "0":
                ListCoord_PD.append(int(0))
            else:
                ListCoord_PD.append(int(1))
        # print(ListCoord_x)
        # print(ListCoord_y)
        # print(ListCoord_effect)
        k += 1

    Grid_Matrix = np.zeros((len(ListCoord_x), len(ListCoord_x)))
    for idx, x in enumerate(ListCoord_x):
        Grid_Matrix[ListCoord_y[idx], x] = ListCoord_PD[idx]
    
    return Grid_Matrix, ListCoord_x, ListCoord_y, ListCoord_PD

# class HandlerSquare(HandlerPatch):
#     def create_artists(self, legend, orig_handle,
#                        xdescent, ydescent, width, height, fontsize, trans):
#         center = xdescent + 0.5 * (width - height), ydescent
#         p = mpatches.Rectangle(xy=center, width=height,
#                                height=height, angle=0.0)
#         self.update_prop(p, orig_handle, legend)
#         p.set_transform(trans)
#         return [p]
    
def Grid_Transform_float(Grids, t, GridL):
    Transform = Grids.iloc[[t]]

    Initial = Transform.to_string()
    Initial = Initial.split("[")[-1]
    Initial = Initial.split("]")[0]
    Initial = Initial.replace("(", "")
    Initial = Initial.replace(" ", "x")
    Initial = Initial.replace(")", "x)")

    delim = ")"
    Str_split = [
        list(y) for x, y in itertools.groupby(Initial, lambda z: z == delim) if not x
    ]


    ListCoord_x = []
    ListCoord_y = []
    ListCoord_effect = []

    StrArr = []

    for idi, i in enumerate(Str_split):
        Sub_StrArr = []
        k = ""
        for idj, j in enumerate(Str_split[idi]):
            if j.isdigit() or j == "." or j == "-" or j == "e" or j == "+":
                k += j
            elif j == "x":
                Sub_StrArr.append(str(k))
                k = ""

        StrArr.append(Sub_StrArr)

    for idi, i in enumerate(StrArr):
        q = 0
        for idj, j in enumerate(i):
            if j.isdigit() and q == 0:
                ListCoord_x.append(int(j))
                q = 1
            elif j.isdigit() and q == 1:
                ListCoord_y.append(int(j))
                q = 2
            elif "." in j or j == "-1" or j.isdigit() and q == 2:
                if j == "-1":
                    ListCoord_effect.append(float(0))
                else:
                    ListCoord_effect.append(float(i[idj]))


    Grid_Matrix_float = np.zeros((len(ListCoord_x), len(ListCoord_x)))
    for idx, x in enumerate(ListCoord_x):
        Grid_Matrix_float[ListCoord_y[idx], x] = ListCoord_effect[idx]

    return Grid_Matrix_float, ListCoord_x, ListCoord_y, ListCoord_effect

def Upper_Lower_string(length): # define the function and pass the length as argument  
    # Print the string in Lowercase  
    result = ''.join((random.choice(string.ascii_lowercase) for x in range(length))) # run loop until the define length  
    return result 

def Simulation(f):
    os.chdir("/home/gillard/Bureau/Book_Project/Towards_a_response_time_of_zero/Script/AB-Model")
    k = Upper_Lower_string(10)
    print(k)
    os.system("mv Config/"+str(f)+" "+str(f))
    os.system('gcc AB-Model.cpp Agent-A.cpp Agent-B.cpp -lstdc++ -lbsd -lboost_program_options -lm -o'+str(k)+'.out')
    os.system("./"+str(k)+".out --config "+str(f))
    os.system("mv "+str(f)+" Config/"+str(f))
    return

# def Simulation_Graph(f):
#     os.chdir("/home/gillard/Bureau/Book_Project/Towards_a_response_time_of_zero/Script/AB-Model")
#     k = Upper_Lower_string(10)
#     print(k)
#     os.system("mv Config/"+str(f)+" "+str(f))
#     os.system('gcc AB-Model.cpp Agent-A.cpp Agent-B.cpp -lstdc++ -lbsd -lboost_program_options -lm -o'+str(k)+'.out')
#     os.system("./"+str(k)+".out --config "+str(f))
    
#     os.chdir("/home/gillard/Bureau/Book_Project/Towards_a_response_time_of_zero/Results/")
    
#     # time.sleep(5)
#     substring = str(f).split("del")[-1]
#     substring = str(f).split(".cfg")[0]
    
#     GridL = 50
    
#     for file_grid_A in os.listdir("grids_A"):
#         if search(substring, file_grid_A):
#             print(file_grid_A)
    
#     Grids_A = pd.read_csv("grids_A/"+str(file_grid_A),
#     header=None)
#     Grid_A_Matrix = Grid_Transform_A(Grids_A, 0, GridL)
#     Grid_A_Matrix_final = Grid_Transform_float(Grids_A, -1, GridL)
            
#     for file_grid_B in os.listdir("grids_B"):
#         if substring in file_grid_B:
#             print(file_grid_B)
            
#     Grids_B = pd.read_csv("grids_B/"+str(file_grid_B),
#     header=None)
#     Grid_B_Matrix = Grid_Transform_B(Grids_B, 0, GridL)
#     Grid_B_Matrix_final = Grid_Transform_B(Grids_B, -1, GridL)
    
#     for file_allmove_A in os.listdir("allmoves_A"):
#         if substring in file_allmove_A:
#             print (file_allmove_A)
            
#     Allmoves_A = pd.read_csv("allmoves_A/"+str(file_allmove_A))
      
#     for file_allmove_B in os.listdir("allmoves_B"):
#         if substring in file_allmove_B:
#             print (file_allmove_B)
            
#     Allmoves_B = pd.read_csv("allmoves_B/"+str(file_allmove_B))

#     Grid_A_Matrix_subset = Grid_A_Matrix[0:GridL, 0:GridL]
#     Grid_A_Matrix_final_subset = Grid_A_Matrix_final[0:GridL, 0:GridL]
    
#     Grid_B_Matrix_subset = Grid_B_Matrix[0:GridL, 0:GridL]
    
#     Grid_B_Matrix_final_subset = Grid_B_Matrix_final[0:GridL, 0:GridL]
    
#     max_effect_R = Allmoves_A["amount_dest_after"].max()
#     max_effect_M = Allmoves_A["new_amount_focal_A"].max()
#     max_effect = 0
#     if max_effect_R > max_effect_M:
#         max_effect = max_effect_R
#     else:
#         max_effect = max_effect_M
#     print(max_effect)
    
#     fig = plt.subplots(figsize=(35, 45))
#     fig = plt.suptitle('Config:'+str(f), fontsize=16)
#     gs = gridspec.GridSpec(3, 2, width_ratios=[1, 0.9], height_ratios=[1,1,1])
#     ax1 = plt.subplot(gs[0])
#     ax2 = plt.subplot(gs[1])
#     ax3 = plt.subplot(gs[2])
#     ax4 = plt.subplot(gs[3])
#     ax5 = plt.subplot(gs[4])
#     ax6 = plt.subplot(gs[5])

#     im = ax1.imshow(
#         Grid_A_Matrix_subset,
#         cmap="jet",
#         vmin=0.0,
#         vmax=max_effect,
#     )

#     im_ratio = Grid_A_Matrix_subset.shape[0] / Grid_A_Matrix_subset.shape[1]
#     cbar = plt.colorbar(im, fraction=0.0455 * im_ratio, ax=ax1)
#     cbar.ax.tick_params(labelsize=24)
#     cbar.set_label("Effect of agents A", rotation=270, labelpad=25, fontsize=24)
#     ax1.set_xlabel(r"$x$-axis", fontsize=24)
#     ax1.set_ylabel(r"$y$-axis", fontsize=24)
#     # ax1 = plt.gca()

#     # Major ticks
#     ax1.set_xticks(np.arange(1, GridL, 2))
#     ax1.set_yticks(np.arange(1, GridL, 2))

#     # Labels for major ticks
#     ax1.set_xticklabels(np.arange(2, GridL + 1, 2), rotation=45)
#     ax1.set_yticklabels(np.arange(2, GridL + 1, 2))

#     # Minor ticks
#     ax1.set_xticks(np.arange(-0.5, GridL, 1), minor=True)
#     ax1.set_yticks(np.arange(-0.5, GridL, 1), minor=True)

#     ax1.xaxis.set_tick_params(labelsize=24)
#     ax1.yaxis.set_tick_params(labelsize=24)

#     # Gridlines based on minor ticks
#     ax1.grid(which="minor", color="black", linestyle="-", linewidth=2)


#     cmap = colors.ListedColormap(["w", "red", "green"])
#     im = ax2.imshow(Grid_B_Matrix_subset, cmap=cmap)
#     # cmap = plt.colorbar(im, fraction=0.0455 * im_ratio, ax=ax2)
#     # cmap.ax.tick_params(labelsize=24)
#     ax2.set_xlabel(r"$x$-axis", fontsize=24)
#     ax2.set_ylabel(r"$y$-axis", fontsize=24)
#     # ax2 = plt.gca()

#     # Major ticks
#     ax2.set_xticks(np.arange(1, GridL, 2))
#     ax2.set_yticks(np.arange(1, GridL, 2))

#     # Labels for major ticks
#     ax2.set_xticklabels(np.arange(2, GridL + 1, 2), rotation=45)
#     ax2.set_yticklabels(np.arange(2, GridL + 1, 2))

#     # Minor ticks
#     ax2.set_xticks(np.arange(-0.5, GridL, 1), minor=True)
#     ax2.set_yticks(np.arange(-0.5, GridL, 1), minor=True)

#     ax2.xaxis.set_tick_params(labelsize=24)
#     ax2.yaxis.set_tick_params(labelsize=24)
#     # rects1 = ax2.bar(1, 1, 1, color="w")
#     # rects2 = ax2.bar(1, 1, 1, color="gray")
#     # rects3 = ax2.bar(10, 0, 1, color="black")
#     # ax2.legend(
#     #     (rects3[0], rects2[0], rects1[0]),
#     #     ("Cooperator", "Defector", "Empty site"),
#     #     handler_map={
#     #         rects1[0]: HandlerSquare(),
#     #         rects2[0]: HandlerSquare(),
#     #         rects3[0]: HandlerSquare(),
#     #     },
#     #     fontsize=24,
#     # )
#     # Gridlines based on minor ticks
#     ax2.grid(which="minor", color="black", linestyle="-", linewidth=2)
    
#     im = ax3.imshow(
#     Grid_A_Matrix_final_subset,
#     cmap="jet",
#     vmin=0.0,
#     vmax=max_effect,
#     )

#     im_ratio = Grid_A_Matrix_subset.shape[0] / Grid_A_Matrix_subset.shape[1]
#     cbar = plt.colorbar(im, fraction=0.0455 * im_ratio, ax=ax3)
#     cbar.ax.tick_params(labelsize=24)
#     cbar.set_label("Effect of agents A", rotation=270, labelpad=25, fontsize=24)
#     ax3.set_xlabel(r"$x$-axis", fontsize=24)
#     ax3.set_ylabel(r"$y$-axis", fontsize=24)
#     # ax1 = plt.gca()

#     # Major ticks
#     ax3.set_xticks(np.arange(1, GridL, 2))
#     ax3.set_yticks(np.arange(1, GridL, 2))

#     # Labels for major ticks
#     ax3.set_xticklabels(np.arange(2, GridL + 1, 2), rotation=45)
#     ax3.set_yticklabels(np.arange(2, GridL + 1, 2))

#     # Minor ticks
#     ax3.set_xticks(np.arange(-0.5, GridL, 1), minor=True)
#     ax3.set_yticks(np.arange(-0.5, GridL, 1), minor=True)

#     ax3.xaxis.set_tick_params(labelsize=24)
#     ax3.yaxis.set_tick_params(labelsize=24)

#     # Gridlines based on minor ticks
#     ax3.grid(which="minor", color="black", linestyle="-", linewidth=2)

#     im = ax4.imshow(Grid_B_Matrix_final_subset, cmap=cmap)
#     # cmap = plt.colorbar(im, fraction=0.0455 * im_ratio, ax=ax2)
#     # cmap.ax.tick_params(labelsize=24)
#     ax4.set_xlabel(r"$x$-axis", fontsize=24)
#     ax4.set_ylabel(r"$y$-axis", fontsize=24)
#     # ax6 = plt.gca()

#     # Major ticks
#     ax4.set_xticks(np.arange(1, GridL, 2))
#     ax4.set_yticks(np.arange(1, GridL, 2))

#     # Labels for major ticks
#     ax4.set_xticklabels(np.arange(2, GridL + 1, 2), rotation=45)
#     ax4.set_yticklabels(np.arange(2, GridL + 1, 2))

#     # Minor ticks
#     ax4.set_xticks(np.arange(-0.5, GridL, 1), minor=True)
#     ax4.set_yticks(np.arange(-0.5, GridL, 1), minor=True)

#     ax4.xaxis.set_tick_params(labelsize=24)
#     ax4.yaxis.set_tick_params(labelsize=24)

#     # Gridlines based on minor ticks
#     ax4.grid(which="minor", color="black", linestyle="-", linewidth=2)
    

#     ax5.scatter(Allmoves_A["simul_step"], Allmoves_A["occupied_sites"], s=1, color="blue")
#     ax5.set_xlabel("Timestep", fontsize=24)
#     ax5.set_ylabel("Number of occupied sites", fontsize=24)
#     ax5.xaxis.set_tick_params(labelsize=24)
#     ax5.yaxis.set_tick_params(labelsize=24)
    
#     CoopLevel = Allmoves_B["cooperators"] / (
#     Allmoves_B["cooperators"] + Allmoves_B["defectors"]
#     )
    
#     ax6.scatter(Allmoves_B["simul_step"], CoopLevel, s=1, color="blue")
#     ax6.set_xlabel("Timestep", fontsize = 24)
#     ax6.set_ylabel("Coop Level (%)", fontsize = 24)
#     ax6.xaxis.set_tick_params(labelsize=24)
#     ax6.yaxis.set_tick_params(labelsize=24)
    
#     figname = substring = str(f).split(".cfg")[0]
#     print(figname)
    
#     plt.savefig("Figures/"+str(figname)+".png",dpi='figure')
    
#     os.chdir(
#     "/home/gillard/Bureau/Book_Project/Towards_a_response_time_of_zero/Script/AB-Model"
#     )
#     os.system("mv "+str(f)+" Config/"+str(f))
#     return

# # def Graph(f):
#     # Couple with Simulation
 
# def Grid_A(grid, max_effect, t, subgrid = 20):
#     fig = plt.figure(figsize=(10, 10))
#     fig.suptitle(r"$t =$"+str(t))
#     spec = gridspec.GridSpec(
#         ncols=1,
#         nrows=1,
#         width_ratios=[1],
#         wspace=0.0,
#         hspace=0.0,
#         height_ratios=[1],
#     )

#     ax1 = fig.add_subplot(spec[0])

#     im_ratio = grid.shape[0] / grid.shape[1]

#     im1 = ax1.imshow(
#         grid,
#         cmap="jet",
#         vmin=0.0,
#         vmax=max_effect,
#     )

#     cb = plt.colorbar(im1, ax=ax1, fraction=0.0455 * im_ratio)
#     cb.set_label("Effect of agents A")
#     ax1.set_xlabel(r"$x$-axis")
#     ax1.set_ylabel(r"$y$-axis")
#     ax1.set_title(r"with interactions between agents A and B")
#     # ax1 = plt.gca()

#     # Major ticks
#     ax1.set_xticks(np.arange(0, subgrid, 1))
#     ax1.set_yticks(np.arange(0, subgrid, 1))

#     # Labels for major ticks
#     ax1.set_xticklabels(np.arange(1, subgrid + 1, 1))
#     ax1.set_yticklabels(np.arange(1, subgrid + 1, 1))

#     # Minor ticks
#     ax1.set_xticks(np.arange(-0.5, subgrid, 1), minor=True)
#     ax1.set_yticks(np.arange(-0.5, subgrid, 1), minor=True)

#     # subgridines based on minor ticks
#     ax1.grid(which="minor", color="black", linestyle="-", linewidth=2)
 
#     return