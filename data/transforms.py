import numpy as np

# make_tiles, normalize99

def make_tiles(imgi, bsize=224, augment=False, tile_overlap=0.1):
    """ make tiles of image to run at test-time

    if augmented, tiles are flipped and tile_overlap=2.
        * original
        * flipped vertically
        * flipped horizontally
        * flipped vertically and horizontally

    Parameters
    ----------
    imgi : float32
        array that's nchan x Ly x Lx

    bsize : float (optional, default 224)
        size of tiles

    augment : bool (optional, default False)
        flip tiles and set tile_overlap=2.

    tile_overlap: float (optional, default 0.1)
        fraction of overlap of tiles

    Returns
    -------
    IMG : float32
        array that's ntiles x nchan x bsize x bsize

    ysub : list
        list of arrays with start and end of tiles in Y of length ntiles

    xsub : list
        list of arrays with start and end of tiles in X of length ntiles

    
    """

    nchan, Ly, Lx = imgi.shape
    if augment:
        bsize = np.int32(bsize)
        # pad if image smaller than bsize
        if Ly<bsize:
            imgi = np.concatenate((imgi, np.zeros((nchan, bsize-Ly, Lx))), axis=1)
            Ly = bsize
        if Lx<bsize:
            imgi = np.concatenate((imgi, np.zeros((nchan, Ly, bsize-Lx))), axis=2)
        Ly, Lx = imgi.shape[-2:]
        # tiles overlap by half of tile size
        ny = max(2, int(np.ceil(2. * Ly / bsize)))
        nx = max(2, int(np.ceil(2. * Lx / bsize)))
        ystart = np.linspace(0, Ly-bsize, ny).astype(int)
        xstart = np.linspace(0, Lx-bsize, nx).astype(int)

        ysub = []
        xsub = []

        # flip tiles so that overlapping segments are processed in rotation
        IMG = np.zeros((len(ystart), len(xstart), nchan,  bsize, bsize), np.float32)
        for j in range(len(ystart)):
            for i in range(len(xstart)):
                ysub.append([ystart[j], ystart[j]+bsize])
                xsub.append([xstart[i], xstart[i]+bsize])
                IMG[j, i] = imgi[:, ysub[-1][0]:ysub[-1][1],  xsub[-1][0]:xsub[-1][1]]
                # flip tiles to allow for augmentation of overlapping segments
                if j%2==0 and i%2==1:
                    IMG[j,i] = IMG[j,i, :,::-1, :]
                elif j%2==1 and i%2==0:
                    IMG[j,i] = IMG[j,i, :,:, ::-1]
                elif j%2==1 and i%2==1:
                    IMG[j,i] = IMG[j,i,:, ::-1, ::-1]
    else:
        tile_overlap = min(0.5, max(0.05, tile_overlap))
        bsizeY, bsizeX = min(bsize, Ly), min(bsize, Lx)
        bsizeY = np.int32(bsizeY)
        bsizeX = np.int32(bsizeX)
        # tiles overlap by 10% tile size
        ny = 1 if Ly<=bsize else int(np.ceil((1.+2*tile_overlap) * Ly / bsize))
        nx = 1 if Lx<=bsize else int(np.ceil((1.+2*tile_overlap) * Lx / bsize))
        ystart = np.linspace(0, Ly-bsizeY, ny).astype(int)
        xstart = np.linspace(0, Lx-bsizeX, nx).astype(int)

        ysub = []
        xsub = []
        IMG = np.zeros((len(ystart), len(xstart), nchan,  bsizeY, bsizeX), np.float32)
        for j in range(len(ystart)):
            for i in range(len(xstart)):
                ysub.append([ystart[j], ystart[j]+bsizeY])
                xsub.append([xstart[i], xstart[i]+bsizeX])
                IMG[j, i] = imgi[:, ysub[-1][0]:ysub[-1][1],  xsub[-1][0]:xsub[-1][1]]
        
    return IMG, ysub, xsub, Ly, Lx

def normalize99(Y, lower=1,upper=99):
    """ normalize image so 0.0 is 1st percentile and 1.0 is 99th percentile """
    X = Y.copy()
    x01 = np.percentile(X, lower)
    x99 = np.percentile(X, upper)
    X = (X - x01) / (x99 - x01)