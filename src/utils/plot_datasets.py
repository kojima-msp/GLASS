import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np

# muted basemap palette (data points keep the 'turbo' colormap so they stand out)
LAND_COLOR = '#e8e8e8'
OCEAN_COLOR = '#dde8f2'
COAST_COLOR = '#9aa0a6'

def _basemap(ax):
    # the coastline polygons are world-wide even where clipped away, so they are rasterized
    ax.add_feature(cfeature.OCEAN, facecolor=OCEAN_COLOR, zorder=0)
    ax.add_feature(cfeature.LAND, facecolor=LAND_COLOR, zorder=0)
    ax.coastlines(color=COAST_COLOR, linewidth=0.5, zorder=0)
    ax.set_rasterization_zorder(1)

def _marker_size(e, vmax, scale):
    return scale * np.clip(e / vmax, 0, 1)

def plot_colorbar(vmin,vmax,out_name):
    '''Stand-alone horizontal color bar, so that every panel is the same size.'''
    fig, ax = plt.subplots(figsize=(3.2,0.16))
    fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin,vmax), cmap='turbo'),
                 cax=ax, orientation='horizontal')
    fig.savefig(out_name,bbox_inches='tight')
    plt.close(fig)

def plot_syn(node_set,E,tv_signal,idx,out_name,vmin,vmax,mode,J=None):

    fig, ax = plt.subplots()
    ax.grid(True)
    ax.set_xlim([-5,105])
    ax.set_ylim([-5,105])
    ax.set_aspect('equal')
    ax.tick_params(labelbottom=False,labelleft=False)

    for edge in E:
        ax.plot([node_set[edge[0],0],node_set[edge[1],0]],[node_set[edge[0],1],node_set[edge[1],1]],color='dimgray',linewidth=0.5,zorder=1)

    if(mode=='original'):
        obs = J.astype(bool)
        ax.scatter(node_set[obs,0],node_set[obs,1],s=200,c=tv_signal[obs,idx],
                   vmin=vmin,vmax=vmax,cmap='turbo',zorder=3)
        ax.scatter(node_set[~obs,0],node_set[~obs,1],s=200,zorder=2,c='black')
    else:
        e = tv_signal[:,idx]
        order = np.argsort(e)
        ax.scatter(node_set[order,0],node_set[order,1],s=_marker_size(e[order],vmax,500),
                   c=e[order],vmin=vmin,vmax=vmax,cmap='turbo',zorder=2)

    fig.savefig(out_name,bbox_inches='tight')
    plt.close(fig)


def plot_nys(coordinates,A,tv_signal,idx,out_name,vmin,vmax,mode,J=None):

    proj = ccrs.PlateCarree()
    ax = plt.axes(projection=proj)

    ax.set_extent([-74.1, -73.7, 40.55, 40.95], crs=proj)
    _basemap(ax)

    lon = coordinates[:,0]
    lat = coordinates[:,1]

    for edge in np.array(np.where(A!=0)).T:
        ax.plot([lon[edge[0]],lon[edge[1]]],[lat[edge[0]],lat[edge[1]]],color='dimgray',zorder=1,transform=proj)

    if(mode=='original'):
        obs = J.astype(bool)
        # the markers overlap, so the busy stations are drawn last and stay visible
        v = tv_signal[obs,idx]
        order = np.argsort(v)
        ax.scatter(lon[obs][order],lat[obs][order],c=v[order],vmin=vmin,vmax=vmax,zorder=2,
                   cmap='turbo',s=100,transform=proj)
        ax.scatter(lon[~obs],lat[~obs],zorder=2,c='black',s=100,transform=proj)
    else:
        e = tv_signal[:,idx]
        order = np.argsort(e)
        ax.scatter(lon[order],lat[order],s=_marker_size(e[order],vmax,300),c=e[order],vmin=vmin,vmax=vmax,cmap='turbo',zorder=2,alpha=0.9,transform=proj)

    plt.savefig(out_name,bbox_inches='tight',dpi=300)
    plt.close()

def plot_covid(coordinates,A,tv_signal,idx,out_name,vmin,vmax,mode,J=None):

    proj = ccrs.PlateCarree()
    ax = plt.axes(projection=ccrs.Mercator())
    ax.set_extent([127.0, 147.0, 25.0, 46.0], crs=proj)
    _basemap(ax)

    lon = coordinates[:,0]
    lat = coordinates[:,1]

    for edge in np.array(np.where(A!=0)).T:
        ax.plot([lon[edge[0]],lon[edge[1]]],[lat[edge[0]],lat[edge[1]]],color='dimgray',zorder=1,transform=proj)

    if(mode=='original'):
        obs = J.astype(bool)
        v = tv_signal[obs,idx]
        order = np.argsort(v)
        ax.scatter(lon[obs][order],lat[obs][order],c=v[order],vmin=vmin,vmax=vmax,zorder=2,cmap='turbo',s=100,transform=proj)
        ax.scatter(lon[~obs],lat[~obs],zorder=2,c='black',s=100,transform=proj)
    else:
        e = tv_signal[:,idx]
        order = np.argsort(e)
        ax.scatter(lon[order],lat[order],s=_marker_size(e[order],vmax,300),c=e[order],vmin=vmin,vmax=vmax,cmap='turbo',zorder=2,alpha=0.9,transform=proj)

    plt.savefig(out_name,bbox_inches='tight',dpi=300)
    plt.close()
