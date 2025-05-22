import numpy as np
import math

def cells_on_unit_sphere(df, pixel_size):
    """
    
    Functions that returns input positions normalized on the unit sphere

    Parameters
    ----------
    df : dataframe
        DESCRIPTION.
    pixel_size : list, int
        DESCRIPTION.
    save : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    df : dataframe
        dataframe with positions normalized on the unit sphere.

    """

    def sphereFit(spX,spY,spZ):
        #   Assemble the A matrix
        spX = np.array(spX)
        spY = np.array(spY)
        spZ = np.array(spZ)
        A = np.zeros((len(spX),4))
        A[:,0] = spX*2
        A[:,1] = spY*2
        A[:,2] = spZ*2
        A[:,3] = 1

        #   Assemble the f matrix
        f = np.zeros((len(spX),1))
        f[:,0] = (spX*spX) + (spY*spY) + (spZ*spZ)
        C, residules, rank, singval = np.linalg.lstsq(A,f)

        #   solve for the radius
        t = (C[0]*C[0])+(C[1]*C[1])+(C[2]*C[2])+C[3]
        radius = math.sqrt(t)

        return radius, C[0], C[1], C[2]

    df1 = df.copy()

    # normalize position on unit sphere
    df1['z_unit'] = df1.z*pixel_size[0]
    df1['y_unit'] = df1.y*pixel_size[1]
    df1['x_unit'] = df1.x*pixel_size[2]
    
    r, c0, c1, c2 = sphereFit(df1.z_unit, df1.y_unit, df1.x_unit)

    df1.z_unit = (df1.z_unit-c0)/r
    df1.y_unit = (df1.y_unit-c1)/r
    df1.x_unit = (df1.x_unit-c2)/r

    return df1
