'''
This module includes functions that would receive a noisy image and using Makov Chain Monte Carlo Sampling tries to
reconstruct the image. Also this module contains a function that tries to find the equilibrium value of the total magnetization
of an XY spin system, again using Monte Carlo Samplings (Markov Chain and Random Walk). These two problems are very similar in concept
and methods.
'''

from skimage import data, img_as_float, color
from skimage.util import random_noise
import numpy as np


class neighbor():

    def __init__(self, ix, iy, ix_max, iy_max):

        if ix < ix_max - 1:
            self.right = ix
        else:
            self.right = 0

        if ix > 0:
            self.left = ix - 1
        else:
            self.left = ix_max - 1

        if iy < iy_max - 1:
            self.up = iy + 1
        else:
            self.up = 0

        if iy > 0:
            self.down = iy - 1
        else:
            self.down = iy_max - 1


def ising_binary_reconstruct(current_image, image_binary, max_itr, J, tau=0.2,
                             random_walk=False, start_fresh=True):
    '''
    This function initializes a random or already prepared pixel set and using
    Markov Chain Monte Carlo converts it to the presented image with or without noise. This
    function uses Ising prior.

    Input:

        current_image: a one layer numpy array to be matched with the given image.
        image_binary: a one layer numpy array as the given binary image.
        max_itr: number of samplings.
        J : Ising prior parameter.
        tau: the likelihood model variance.
        random_walk: Boolean. It true it picks the next site randomly from one of the
        neighbors of the previous site otherwise it
        start_fresh: Boolean. It starts the current_image array with random numbers.

    '''

    # read image sizes:
    ix_max, iy_max = image_binary.shape
    
    # start with previous runs or not:
    if start_fresh:
        X = np.ones((ix_max, iy_max))
    else:
        X = current_image
        
    # initialize the sample pixel:
    ix = np.random.randint(low=0, high=ix_max)
    iy = np.random.randint(low=0, high=iy_max)
    
    # diff is the difference between the pixels in the next step from previous step.
    diff = []
    itr = 0
    if random_walk:
        while itr < max_itr:
            # read neighbor pixels and neighbor pixel values
            nb = neighbor(ix, iy, ix_max, iy_max)
            neighbor_sites = [(ix,nb.up), (ix, nb.down), (nb.left, iy), (nb.right, iy)]
            neighbor_thetas = [X[a[0], a[1]] for a in neighbor_sites]
            # flip the pixel value.
            Xp = -X[ix, iy]
            # calculate the likelihood ratio.
            like_ratio = np.exp(image_binary[ix, iy]*(Xp - X[ix, iy])/tau**2)
            d = [X[ix,iy] != t for t in neighbor_thetas]
            dp = [Xp != t for t in neighbor_thetas]
            # calculate ising prior ratio.
            ising_prior_ratio = np.exp(2*J*(sum(d) - sum(dp)))
            # calculate posterior ratio.
            prop_ratio = like_ratio*ising_prior_ratio
            # acceptance:
            r = np.random.rand()
            if r < prop_ratio:
                X[ix, iy] = Xp
            # take the difference for every 100 steps:
            if itr%100 == 0:
                dX = X - image_binary
                dX_flat = dX.flatten()
                diff.append(np.linalg.norm(dX_flat)/len(dX_flat))
            # pick randomly from one of the neighboring pixels.
            ix = np.random.choice([ix] + [a[0] for a in neighbor_sites])
            iy = np.random.choice([iy] + [a[1] for a in neighbor_sites])
            itr += 1
        return X, diff
    
    else:
        while itr < max_itr:
            # randomly pick a pixel to sample:
            ix = np.random.randint(low=0,high=ix_max)
            iy = np.random.randint(low=0,high=iy_max)
            # All the lines are the same as above block except where noted:
            nb = neighbor(ix,iy,ix_max,iy_max)
            neighbor_sites = [(ix,nb.up),(ix,nb.down),(nb.left,iy),(nb.right,iy)]
            neighbor_thetas = [X[a[0],a[1]] for a in neighbor_sites]
            Xp = -X[ix,iy]
            like_ratio = np.exp(image_binary[ix,iy]*(Xp - X[ix,iy])/tau**2)
            d = [X[ix,iy] != t for t in neighbor_thetas]
            dp = [Xp != t for t in neighbor_thetas]
            ising_prior_ratio = np.exp(2*J*(sum(d)-sum(dp)))
            prop_ratio = like_ratio*ising_prior_ratio
            r = np.random.rand()
            if r < prop_ratio:
                X[ix,iy] = Xp
            if itr%100 == 0:
                dX = X - image_binary
                dX_flat = dX.flatten()
                diff.append(np.linalg.norm(dX_flat)/len(dX_flat))
            itr += 1
        return X,diff

    
def mlv_reconstruct(current_image, image, max_itr, J, tau=0.2, gap = 0.1, epsilon=1.0,
                             random_walk=False, start_fresh=True):
    '''
    This function initializes a random or already prepared pixel set and using
    Markov Chain Monte Carlo converts it to the presented image with or without noise. This
    function uses multilevel prior in attempt to go beyond binary images.

    Input:

        current_image: a one layer numpy array to be matched with the given image.
        image: a one layer numpy array as the given image.
        max_itr: number of samplings.
        J : multilevel prior parameter.
        tau: the likelihood model variance.
        gap: multilevel prior inter-level spacing.
        epsilon: float. usuallu less than one. The strength of pixel change during each
        sampling.
        random_walk: Boolean. It true it picks the next site randomly from one of the
        neighbors of the previous site otherwise it
        start_fresh: Boolean. It starts the current_image array with random numbers.

    '''

    # initialize the image:
    ix_max, iy_max = image_binary.shape
    if start_fresh:
        X = np.ones((ix_max, iy_max))
    else:
        X = current_image
        
    # randomly start with a pixel:
    ix = np.random.randint(low=0, high=ix_max)
    iy = np.random.randint(low=0, high=iy_max)
    # diff is the difference between the pixels in the next step from previous step.
    
    diff = []
    itr = 0
    if random_walk:
        while itr < max_itr:
            nb = neighbor(ix, iy, ix_max, iy_max)
            # read neighbor pixels and neighbor pixel values
            neighbor_sites = [(ix, nb.up),(ix, nb.down), (nb.left, iy), (nb.right, iy)]
            neighbor_thetas = [X[a[0], a[1]] for a in neighbor_sites]
            # sample a new pixel value:
            delta_X = (np.random.rand() - 1)*epsilon
            Xp = X[ix, iy] + delta_X
            # calculate prior and likelihood ratios:
            exponent = image[ix, iy]*(Xp - X[ix, iy])
            exponent += Xp**2 - X[ix, iy]**2
            like_ratio = np.exp(exponent/tau**2)
            d = [(X[ix, iy] - t) > gap for t in neighbor_thetas]
            dp = [(Xp - t) > gap for t in neighbor_thetas]
            ising_prior_ratio = np.exp(2*J*(sum(d)-sum(dp)))
            prop_ratio = like_ratio*ising_prior_ratio
            # acceptance:
            r = np.random.rand()
            if r < prop_ratio:
                X[ix, iy] = Xp
            if itr%100 == 0:
                dX = X - image
                dX_flat = dX.flatten()
                diff.append(np.linalg.norm(dX_flat)/len(dX_flat))
            # pick a pixel randomly from the previous pixel neighbors:
            ix = np.random.choice([ix] + [a[0] for a in neighbor_sites])
            iy = np.random.choice([iy] + [a[1] for a in neighbor_sites])
            itr += 1
        return X,diff
    
    else:
        while itr < max_itr:
            # pick the pixel randomly every time:
            ix = np.random.randint(low=0, high=ix_max)
            iy = np.random.randint(low=0, high=iy_max)
            # the rest of the lines are the same as above:
            nb = neighbor(ix, iy, ix_max, iy_max)
            neighbor_sites = [(ix, nb.up), (ix,nb.down), (nb.left, iy), (nb.right, iy)]
            neighbor_thetas = [X[a[0], a[1]] for a in neighbor_sites]
            delta_X = (np.random.rand() - 1)*epsilon
            Xp = X[ix, iy] + delta_X
            exponent = image[ix, iy]*(Xp - X[ix, iy])
            exponent += Xp**2 - X[ix, iy]**2
            like_ratio = np.exp(exponent/tau**2)
            d = [np.abs(X[ix, iy] - t) > gap for t in neighbor_thetas]
            dp = [np.abs(Xp - t) > gap for t in neighbor_thetas]
            ising_prior_ratio = np.exp(2*J*(sum(d) - sum(dp)))
            prop_ratio = like_ratio*ising_prior_ratio
            r = np.random.rand()
            if r < prop_ratio:
                X[ix, iy] = Xp
            if itr%100 == 0:
                dX = X - image
                dX_flat = dX.flatten()
                diff.append(np.linalg.norm(dX_flat)/len(dX_flat))
            itr += 1
        return X,diff

def XY_magnetization(theta, max_itr, J, h, size, n_max=10,
                     compute_average=False, start_fresh=False):
    
    ix_max = size[0]
    iy_max = size[1]
    # diving the 2pi angels :
    angles = [2*np.pi*i/n_max for i in range(n_max)]
    if start_fresh:
        theta = np.random.choice(angles, size=size)
    itr = 0
    
    # picke a site at random to start:
    ix = np.random.randint(low=0, high=ix_max)
    iy = np.random.randint(low=0, high=iy_max)
    
    # record difference between mean cos(angle) from step to step:
    diff = []
    # average of the magnetization cos(angles) over the desired period:
    mag = 0
    # record the magnetization over a desired period:
    mag_track = []
    while itr < max_itr:
        # pick the site to sample randomly at each step:
        ix = np.random.randint(low=0, high=ix_max)
        iy = np.random.randint(low=0, high=iy_max)
        nb = neighbor(ix, iy, ix_max, iy_max)
        # identify the neighboring sites and values:
        neighbor_sites = [(ix, nb.up), (ix, nb.down),(nb.left, iy), (nb.right, iy)]
        neighbor_thetas = [theta[a[0], a[1]] for a in neighbor_sites]
        # pick a value to modify the picked site at each step:
        d_theta = np.random.choice(angles)
        thetap = theta[ix, iy] + d_theta
        d = [np.cos(theta[ix, iy] - t) for t in neighbor_thetas]
        dp = [np.cos(thetap - t) for t in neighbor_thetas]
        # calculate the probability ratio:
        d_energy = J*(sum(d)-sum(dp)) - h*np.cos(theta[ix, iy]) + h*np.cos(thetap)
        # record at every 100 steps:
        if itr%100 == 0:
            dt = np.cos(theta[ix, iy]) - np.cos(thetap)
            diff.append(dt)
        # does the new sample lower the energy:
        if d_energy < 0:
            theta[ix, iy] = thetap
        else:
        # accept the proposal:
            r = np.random.rand()
            prior_ratio = np.exp(-d_energy)
            if r < prior_ratio:
                theta[ix, iy] = thetap
        if compute_average:
            inst_mag = np.mean(np.cos(theta))
            mag_track.append(inst_mag)
            mag += inst_mag/max_itr

        itr += 1
    return theta, diff, mag_track, mag
