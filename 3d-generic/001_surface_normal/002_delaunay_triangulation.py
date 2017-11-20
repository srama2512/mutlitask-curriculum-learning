import math
import json
import argparse
import numpy as np
from scipy.spatial import Delaunay

def str2bool(v):
    return v.lower() in ['t', 'true', '1', 'y', 'yes']

parser = argparse.ArgumentParser()
parser.add_argument('--do_plot', default=False, type=str2bool)
opts = parser.parse_args()

if opts.do_plot:
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors
    from mpl_toolkits.mplot3d import Axes3D
    # Delaunay plotting imports
    import plotly.plotly as py
    from plotly.graph_objs import *

cluster_centers = np.array(json.load(open('temp_clusters.json')))

# Normalize the cluster centers
cluster_centers = cluster_centers / np.linalg.norm(cluster_centers, axis=1)[:, np.newaxis]

if opts.do_plot:
    # Create a sphere
    r = 1
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0*pi:100j]
    x = r*sin(phi)*cos(theta)
    y = r*sin(phi)*sin(theta)
    z = r*cos(phi)

xx = cluster_centers[:, 0]
yy = cluster_centers[:, 1]
zz = cluster_centers[:, 2]

if opts.do_plot:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(
        x, y, z,  rstride=1, cstride=1, color='c', alpha=0.6, linewidth=0)

    ax.scatter(xx,yy,zz,color="k",s=20)

    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_zlim([-1,1])
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.xlabel('X: Right positive')
    plt.ylabel('Y: Forward positive')

    plt.show()

phi_theta = np.zeros((xx.shape[0], 2)) # (Angle wrt Z axis, Angle of XY projection wrt X axis)

for i in range(xx.shape[0]):
    phi_theta[i, 0] = math.atan2(yy[i], xx[i])
    phi_theta[i, 1] = math.atan2(math.sqrt(yy[i]*yy[i] + xx[i]*xx[i]), zz[i])

dt = Delaunay(phi_theta)
json.dump(dt.simplices.tolist(), open('delaunay_vertices.json', 'w'))

if opts.do_plot: 
    # Functions obtained from https://plot.ly/python/surface-triangulation/
    def map_z2color(zval, colormap, vmin, vmax):
        #map the normalized value zval to a corresponding color in the colormap
        
        if vmin>vmax:
            raise ValueError('incorrect relation between vmin and vmax')
        t=(zval-vmin)/float((vmax-vmin))#normalize val
        R, G, B, alpha=colormap(t)
        return 'rgb('+'{:d}'.format(int(R*255+0.5))+','+'{:d}'.format(int(G*255+0.5))+\
               ','+'{:d}'.format(int(B*255+0.5))+')'   

    def tri_indices(simplices):
        #simplices is a numpy array defining the simplices of the triangularization
        #returns the lists of indices i, j, k
        
        return ([triplet[c] for triplet in simplices] for c in range(3))

    def plotly_trisurf(x, y, z, simplices, colormap=cm.RdBu, plot_edges=None):
        #x, y, z are lists of coordinates of the triangle vertices 
        #simplices are the simplices that define the triangularization;
        #simplices  is a numpy array of shape (no_triangles, 3)
        #insert here the  type check for input data
        
        points3D=np.vstack((x,y,z)).T
        tri_vertices=map(lambda index: points3D[index], simplices)# vertices of the surface triangles     
        zmean=[np.mean(tri[:,2]) for tri in tri_vertices ]# mean values of z-coordinates of 
                                                          #triangle vertices
        min_zmean=np.min(zmean)
        max_zmean=np.max(zmean)  
        facecolor=[map_z2color(zz,  colormap, min_zmean, max_zmean) for zz in zmean] 
        I,J,K=tri_indices(simplices)
        
        triangles=Mesh3d(x=x,
                         y=y,
                         z=z,
                         facecolor=facecolor, 
                         i=I,
                         j=J,
                         k=K,
                         name=''
                        )
        
        if plot_edges is None:# the triangle sides are not plotted 
            return Data([triangles])
        else:
            #define the lists Xe, Ye, Ze, of x, y, resp z coordinates of edge end points for each triangle
            #None separates data corresponding to two consecutive triangles
            lists_coord=[[[T[k%3][c] for k in range(4)]+[ None]   for T in tri_vertices]  for c in range(3)]
            Xe, Ye, Ze=[reduce(lambda x,y: x+y, lists_coord[k]) for k in range(3)]
            
            #define the lines to be plotted
            lines=Scatter3d(x=Xe,
                            y=Ye,
                            z=Ze,
                            mode='lines',
                            line=Line(color= 'rgb(50,50,50)', width=1.5)
                   )
            return Data([triangles, lines])

    data1 = plotly_trisurf(xx, yy, zz, dt.simplices, colormap=cm.RdBu, plot_edges=True)

    # Set layout of the plot
    axis = dict(
    showbackground=True,
    backgroundcolor="rgb(230, 230,230)",
    gridcolor="rgb(255, 255, 255)",
    zerolinecolor="rgb(255, 255, 255)",
        )

    layout = Layout(
             title='Delaunay Triagnulation of codebook',
             width=800,
             height=800,
             scene=Scene(
             xaxis=XAxis(axis),
             yaxis=YAxis(axis),
             zaxis=ZAxis(axis),
            aspectratio=dict(
                x=1,
                y=1,
                z=0.5
            ),
            )
            )

    fig1 = Figure(data=data1, layout=layout)
    py.iplot(fig1, filename='Delaunay-trisurf')
