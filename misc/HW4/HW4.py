import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def extract_grid(grid_file:str, is_3D:bool=False, 
                 sep:str=', ', engine:str='python',
                 headers:list=['xcrds', 'ycrds', 'zcrds']):
    """
    Read in the grid file (.dat) and import into a Pandas DataFrame

    Args:
        grid_file (str): Name of the .dat to retrieve coordinate file
        is_3D (bool, optional): Add a 3rd/Z dimenions
        sep (str, optional): Formatting of the grid_file to read in.
            Defaults to 'False'
        engine (str, optional): File import engine. Defaults to 'python'.
        headers (list, optional): Name of the column headers.
            Defaults to ['xcrds', 'ycrds', 'zcrds'].

    Returns:
        DataFrame, list: Coordinate Data Dataframe and list
        with the grid size
    """
    # 2D is default
    if not is_3D:
        # dropping the z dimension if 2D
        headers.pop()
    # 1. Use pandas to extract each coordinate grid values
    df = pd.read_csv(grid_file, sep=sep, header=None, engine=engine)
    # read in grid size, very first line
    grid_size = [int(df[0][0]), int(df[1][0])]
    # only include coordinates
    df = df[1:]
    df.columns = headers
    # for differentiating between imported and generated (halo) cells
    df['src'] = 'import'
    # 2. return grid coords and size ; generalized for both 2D and 3D systems
    return df, grid_size
 
def add_halo_cells(grid_coords:pd.DataFrame, grid_size:list):
    """
    Add in halo cells

    Args:
        grid_coords (pandas DataFrame): Grid Coordinates stored inside
            dataframe
        grid_size (list): Grid Size

    Returns:
        DataFrame, list: New grid with generated halo cells and new
            cell size
    """
    new_grid = pd.DataFrame()
   
    x_size = grid_size[0]
    y_size = grid_size[1]

    headers = list(grid_coords.columns.values)

    x = grid_coords[headers[0]]
    y = grid_coords[headers[1]]

    xminval = min(x)
    xmaxval = max(x)
    yminval = min(y)
    ymaxval = max(y)

    # assuming the grids are evenly spaced out
    dx = abs(xmaxval - xminval) / x_size
    dy = abs(ymaxval - yminval) / y_size

    # add bottom halo vertices first
    temp = pd.DataFrame()
    temp[headers[0]] = x[0:x_size]
    temp[headers[1]] = y[0:x_size] - dy
    temp['src'] = 'halo'
    new_grid = pd.concat([new_grid, temp])

    # add the original grids
    temp = pd.DataFrame()
    temp[headers[0]] = x[0:x_size*y_size]
    temp[headers[1]] = y[0:x_size*y_size]
    temp['src'] = 'grid'
    new_grid = pd.concat([new_grid, temp])

    # add top halo vertices last
    temp = pd.DataFrame()
    temp[headers[0]] = x[(x_size*y_size)-x_size:x_size*y_size]
    temp[headers[1]] = y[(x_size*y_size)-x_size:x_size*y_size] + dy
    temp['src'] = 'halo'
    new_grid = pd.concat([new_grid, temp])

    # then add left and right halo vertices
    x = new_grid[headers[0]]
    y = new_grid[headers[1]]
    _grid = pd.DataFrame()
    idx = 0
    for _ in range(y_size+2):
        seg_end = idx+x_size-1

        temp = pd.DataFrame()
        temp[headers[0]] = pd.Series(x.iloc[idx] - dx)
        temp[headers[1]] = pd.Series(y.iloc[idx])
        temp['src'] = 'halo'
        _grid = pd.concat([_grid, temp, new_grid[idx:seg_end+1]])

        temp = pd.DataFrame()
        temp[headers[0]] = pd.Series(x.iloc[seg_end] + dx)
        temp[headers[1]] = pd.Series(y.iloc[seg_end])
        temp['src'] = 'halo'
        _grid = pd.concat([_grid, temp])
        
        idx = seg_end + 1
    
    # assume only 2 set of halo cells vertically and horizontally
    new_grid_size = [grid_size[0]+2, grid_size[1]+2]

    return _grid, new_grid_size    

def calculate_cell_volumes(grid_coords:pd.DataFrame, grid_size:list):
    """
    Calculate the cell volumes for each face center

    Args:
        grid_coords (pandas DataFrame): Grid coordinates of 
            Vertices (2D only)
        grid_size (list): Cell size in X and Y directions

    Returns:
        DataFrame: New DataFrame with face center 
            coordinates and volume
    """
    cell_vol_df = pd.DataFrame()

    # get headers and x and y coordinates
    headers = list(grid_coords.columns.values)
    x = grid_coords[headers[0]]
    y = grid_coords[headers[1]]
    
    # get coordinate size
    x_size = grid_size[0]
    y_size = grid_size[1]

    # size of x and y face center coordinates should be 2-less
    # since those coordinate do not include the size vertices
    xc_size = x_size - 2
    yc_size = y_size - 2

    # initialize arrays to store center coordinates and volume
    xc = np.zeros(xc_size*yc_size)
    yc = np.zeros(xc_size*yc_size)
    vol = np.zeros(xc_size*yc_size)

    # get the cell-center coordinates, starting from
    # the bottom row of vertices
    for iy in range(yc_size-2):
        ybotstart = iy*x_size
        ybotend = ybotstart+x_size-1
        ybot = y[ybotstart:ybotend]
        xbot = x[ybotstart:ybotend]
        
        ytopstart = ybotend+1
        ytopend = ytopstart+x_size-1
        ytop = y[ytopstart:ytopend]
        xtop = x[ytopstart:ytopend]

        for ix in range(xc_size-2):
            _xc = (xtop.iloc[ix] + xbot.iloc[ix+1]) * 0.5
            _yc = (ytop.iloc[ix] + ybot.iloc[ix]) * 0.5
            # calculate the volume for each of the cells, 
            # associated with a cell center
            _vol = 0.5 * abs(((xtop.iloc[ix+1] - xbot.iloc[ix])*
                              (ytop.iloc[ix+1] - ybot.iloc[ix]))
                             - ((xtop.iloc[ix] - xbot.iloc[ix+1])*
                                (ytop.iloc[ix] - ybot.iloc[ix+1])))
            currIdx = (iy*x_size) + ix
            xc[currIdx] = _xc
            yc[currIdx] = _yc
            vol[currIdx] = _vol

    # add the data into the dataframe 
    cell_vol_df[headers[0]] = pd.Series(xc)
    cell_vol_df[headers[1]] = pd.Series(yc)
    cell_vol_df['Vol'] = pd.Series(vol)
    return cell_vol_df

def calculate_cell_face_areas(grid_coords:pd.DataFrame, grid_size:list):
    """
    Calculate cell face areas in the 'eta' and 'xi' directions
    at each face centroid

    Args:
        grid_coords (pandas DataFrame): Grid coordinates of
            Vertices (2D only)
        grid_size (list): Cell size in X and Y directions

    Returns:
        DataFrame: Data at all the face centroids with their areas
    """
    area = pd.DataFrame()

    # get coordinate data
    headers = list(grid_coords.columns.values)
    x = grid_coords[headers[0]]
    y = grid_coords[headers[1]]
    
    # get coordinate size
    x_size = grid_size[0]
    y_size = grid_size[1]

    # size of x and y face center coordinates should be 2-less
    # since those coordinate do not include the size vertices
    xc_size = x_size - 2
    yc_size = y_size - 2

    # initialize coordinate data and calculated area arrays to store
    xc = np.zeros(xc_size*yc_size)
    yc = np.zeros(xc_size*yc_size)
    area_n = np.zeros(xc_size*yc_size)
    area_xi = np.zeros(xc_size*yc_size)

    # get the cell-center coordinates, starting from
    # the bottom row of vertices
    for iy in range(yc_size-2):
        ybotstart = iy*xc_size
        ybotend = ybotstart+xc_size-1
        ybot = y[ybotstart:ybotend]
        xbot = x[ybotstart:ybotend]
        
        ytopstart = ybotend+1
        ytopend = ytopstart+xc_size-1
        ytop = y[ytopstart:ytopend]
        xtop = x[ytopstart:ytopend]

        for ix in range(xc_size-2):
            # calculate the face areas for each face for
            # each cell, associated with a cell center
            xcrd = (xbot.iloc[ix] + xbot.iloc[ix+1]) * 0.5
            ycrd = (ybot.iloc[ix] + ytop.iloc[ix]) * 0.5

            y_xi = -(ytop.iloc[ix+1] - ytop.iloc[ix])
            x_xi = (xtop.iloc[ix+1] - xtop.iloc[ix])
            y_n = (ytop.iloc[ix+1] - ybot.iloc[ix+1])
            x_n = -(xtop.iloc[ix+1] - xbot.iloc[ix+1])

            currIdx = (iy*x_size) + ix
            xc[currIdx] = xcrd
            yc[currIdx] = ycrd
            area_n[currIdx] = np.sqrt((x_n**2) + (y_n**2))
            area_xi[currIdx] = np.sqrt((x_xi**2) + (y_xi**2))

    # add in all the data
    area[headers[0]] = pd.Series(xc)
    area[headers[1]] = pd.Series(yc)
    area['Area_eta'] = pd.Series(area_n)
    area['Area_xi'] = pd.Series(area_xi)

    return area

def plot_contour(grid_coords:pd.DataFrame, show_plot:bool=True, 
                 hue='src', cmap='viridis', colorbar_label='Cell Volume',
                 coord_loc:list=[0,0], coord_loc_2:list=[-0.495,0]):
    """
    Plots scatterplot contours for a 'secondary' value for each 
    coordinate location.

    Args:
        grid_coords (pd.DataFrame): Dataframe of coordinate data 
            and contour values
        show_plot (bool, optional): Show the plot on the screen.
            Defaults to True.
        hue (str, optional): What data source to use to color the contour.
            Defaults to 'src'.
        cmap (str, optional): Type of color mapping and theme.
            Defaults to 'viridis'.
        colorbar_label (str, optional): Label for color bar.
            Defaults to 'Cell Volume'.
        coord_loc (list, optional): The min and max are proportional
            to x,y size, not the actual coordinate values.
            Defaults to [0,0].
        coord_loc_2 (list, optional): The min and max are proportional
            to x,y size, not the actual coordinate values.
            Defaults to [-0.495,0].
    """
    headers = list(grid_coords.columns.values)

    ax = sns.scatterplot(data=grid_coords, x=headers[0], y=headers[1],
                         hue=hue, s=5, palette=cmap)
    plt.legend([],[],frameon=False) # keep legend invisible

    min_color = grid_coords[hue].min()
    max_color = grid_coords[hue].max()
    norm = plt.Normalize(min_color, max_color)

    # scale the coloring
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(
        sm,
        ax=plt.gca(),
        label=colorbar_label
    )

    xminval = min(grid_coords[headers[0]])
    xmaxval = max(grid_coords[headers[0]])
    yminval = min(grid_coords[headers[1]])
    ymaxval = max(grid_coords[headers[1]])

    # set the limits for x and y direction
    # and add spacing between plot border
    ax.set_xlim(xminval-0.05, xmaxval+0.05)
    ax.set_ylim(yminval-0.05, ymaxval+0.05)

    # for 1st (cartesian) coordinate axe
    xprop = abs(coord_loc[0] - xminval) / (xmaxval - xminval)
    yprop = abs(coord_loc[1] - yminval) / (ymaxval - yminval)
    ax.axvline(x = coord_loc[0], ymin=yprop, ymax=yprop+0.2, color='k')
    ax.axhline(y = coord_loc[1], xmin=xprop, xmax=xprop+0.15, color='k')
    ax.text(coord_loc[0], coord_loc[1]+(0.21*(ymaxval - yminval)), "y")
    ax.text(coord_loc[0]+(0.16*(xmaxval - xminval)), coord_loc[1], "x")

    # for 2nd coordinate axe
    xprop = abs(coord_loc_2[0] - xminval) / (xmaxval - xminval)
    yprop = abs(coord_loc_2[1] - yminval) / (ymaxval - yminval)
    ax.axvline(x = coord_loc_2[0], ymin=yprop, ymax=yprop+0.2, color='r')
    ax.axhline(y = coord_loc_2[1], xmin=xprop, xmax=xprop+0.15, color='r')
    ax.text(coord_loc_2[0], coord_loc_2[1]+(0.21*(ymaxval - yminval)),
            r"$\eta$")
    ax.text(coord_loc_2[0]+(0.16*(xmaxval - xminval)), coord_loc_2[1],
            r"$\xi$")
    
    if show_plot:
        plt.show()

def plot_grid(grid_coords:pd.DataFrame, show_plot:bool=True, hue='src',
              coord_loc:list=[0, 0], coord_loc_2:list=[-0.495,0]):
    """
        Plots scatterplot contours for a 'secondary' value for 
        each coordinate location.

        Args:
            grid_coords (pd.DataFrame): Dataframe of coordinate data
              and contour values
            show_plot (bool, optional): Show the plot on the screen.
                Defaults to True.
            hue (str, optional): What data source to use to color
                the contour. Defaults to 'src'.
            coord_loc (list, optional): The min and max are proportional
                to x,y size, not the actual coordinate values. 
                Defaults to [0,0].
            coord_loc_2 (list, optional): The min and max are proportional
                to x,y size, not the actual coordinate values.
                Defaults to [-0.495,0].
        """
    headers = list(grid_coords.columns.values)
    ax = sns.scatterplot(data=grid_coords, x=headers[0], y=headers[1], 
                         s=5, hue=hue)

    xminval = min(grid_coords[headers[0]])
    xmaxval = max(grid_coords[headers[0]])
    yminval = min(grid_coords[headers[1]])
    ymaxval = max(grid_coords[headers[1]])

    ax.set_xlim(xminval-0.05, xmaxval+0.05)
    ax.set_ylim(yminval-0.05, ymaxval+0.05)

    # for 1st (cartesian) coordinate axe
    xprop = abs(coord_loc[0] - xminval) / (xmaxval - xminval)
    yprop = abs(coord_loc[1] - yminval) / (ymaxval - yminval)
    ax.axvline(x = coord_loc[0], ymin=yprop, ymax=yprop+0.2, color='k')
    ax.axhline(y = coord_loc[1], xmin=xprop, xmax=xprop+0.15, color='k')
    ax.text(coord_loc[0], coord_loc[1]+(0.21*(ymaxval - yminval)), "y")
    ax.text(coord_loc[0]+(0.16*(xmaxval - xminval)), coord_loc[1], "x")

    # for 2nd coordinate axe
    xprop = abs(coord_loc_2[0] - xminval) / (xmaxval - xminval)
    yprop = abs(coord_loc_2[1] - yminval) / (ymaxval - yminval)
    ax.axvline(x = coord_loc_2[0], ymin=yprop, ymax=yprop+0.2, color='r')
    ax.axhline(y = coord_loc_2[1], xmin=xprop, xmax=xprop+0.15, color='r')
    ax.text(coord_loc_2[0], coord_loc_2[1]+(0.21*(ymaxval - yminval)),
            r"$\eta$")
    ax.text(coord_loc_2[0]+(0.16*(xmaxval - xminval)), coord_loc_2[1],
            r"$\xi$")
    
    if show_plot:
        plt.show()


if __name__ == "__main__":

    grid_file = "g65x49u.dat"
    grid_coords, grid_size = extract_grid(grid_file)
    print(f"Grid size: {grid_size[0]}, {grid_size[1]}")
    plot_grid(grid_coords=grid_coords)

    new_grid_coords, new_grid_size =\
        add_halo_cells(grid_coords=grid_coords, grid_size=grid_size)
    plot_grid(grid_coords=new_grid_coords)

    vol = calculate_cell_volumes(new_grid_coords, new_grid_size)
    vol = vol[vol['Vol'] > 0.0] # to format the colorbar scaling
    plot_contour(grid_coords=vol, hue='Vol', colorbar_label='Cell Volume')

    area = calculate_cell_face_areas(grid_coords, new_grid_size)
    area = area[area['Area_eta'] > 0.0]
    area = area[area['Area_xi'] > 0.0]
    plot_contour(area, hue='Area_eta', 
                 colorbar_label=r'Cell Face Area in $\eta$ Direction')
    plot_contour(area, hue='Area_xi',
                 colorbar_label=r'Cell Face Area in $\xi$ Direction')