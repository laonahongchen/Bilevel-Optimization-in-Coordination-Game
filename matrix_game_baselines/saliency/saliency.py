from scipy.ndimage.filters import gaussian_filter
from scipy.spatial import distance
import matplotlib.pyplot as plt
from scipy.misc import imsave 
import scipy.stats as st
import numpy as np
import scipy
import cv2
import os

class Saliency(object):
    """ Pertubation based saliency based on https://arxiv.org/abs/1711.00138 """
    def __init__(self, c, net, sess):
        
        # Get 2d list of masks with 2d Gausian kernels
        # centered at each x and y coordinate. 
        self.m = self.generateMasks(kernlen=c.dim[0])
        self.y = c.dim[0]
        self.x = c.dim[1]
        self.obs_dist = c.dim[0]//2
        self.net = net
        self.sess = sess
        self.s = np.zeros((self.y, self.x))
        self.t = 0

    def generateMasks(self, kernlen=13, nsig=10):
        '''
        Returns an array of 2D Gaussian kernels that are
        expanded with identical values along the thrid 
        (chanel) dimension.
        :param int: kernlen, length of the kernels to be created
        :param int: nsig
        :return: 2d list of kernels
        '''
        center = kernlen // 2 
        interval = (2*nsig+1.)//(kernlen)
        x = np.linspace(-nsig-interval//2., nsig+interval//2., kernlen+1)
        kern1d = np.diff(st.norm.cdf(x))
        kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
        kernel = np.array(kernel_raw//kernel_raw.sum())
        kernels = []
        for yShift in range(kernlen):
            kernels.append([])
            for xShift in range(kernlen):
                kernels[len(kernels)-1].append(255.0*np.tile(np.roll(np.roll(kernel, center+1+yShift, axis=0),\
                                                         center+1+xShift, axis=1)[:, :, None], [1, 1, 3]))
        return kernels


    def blur(self, obs):
        '''
        Returns a blured observation.
        :param tensor: observation
        :return: blured tensor
        '''
        return gaussian_filter(obs, sigma=7)

    def output(self, obs):
        '''
        Returns output with saliency
        '''
        b = self.blur(obs)
        v = self.net.fetch('outputs', self.sess, [obs])[0]
        for y in range(self.y):
            for x in range(self.x):
                phi = np.multiply(obs, (1.0 - self.m[y][x])) + np.multiply(b, self.m[y][x])
                v_phi = self.net.fetch('outputs', self.sess, [phi])[0]
                self.s[y][x] = distance.euclidean(v, v_phi)
        obs[:,:,0] = obs[:,:,0] + self.s
        return obs 

    def coordinates(self, obs, coordinates, location, hw):
        '''
        Used to get saliency coordinates for agent
        :param tensor: Observation
        :param vector: coordinates for which saliency is to be loaded
        :param vector: Agent location
        '''
        b = self.blur(obs)
        v = self.net.fetch('outputs', self.sess, [obs])[0]
        saliency = []
        for x, y in coordinates:
            dist_x = x - location[0]
            dist_y = y - location[1]
            relx = (dist_x + self.obs_dist)%hw[1]
            rely = (dist_y + self.obs_dist)%hw[0]
            if relx < 0 or rely < 0 or relx > len(obs[1]) or rely > len(obs[0]):
                saliency.append(0.0)
            else:
                #print(obs[rely][relx])
                phi = np.multiply(obs, (1.0 - self.m[rely][relx])) + np.multiply(b, self.m[rely][relx])
                v_phi = self.net.fetch('outputs', self.sess, [phi])[0]
                saliency.append(distance.euclidean(v, v_phi))
        return saliency

    def render(self, obs):
        '''
        Used to render the env.
        '''
        r = 16
        try:
            img = np.repeat(np.repeat(obs, r, axis=0), r, axis=1).astype(np.uint8)
            cv2.imshow('image', img)
            k = cv2.waitKey(50)
            if k == 27:         # If escape was pressed exit
                cv2.destroyAllWindows()
        except AttributeError:
            pass
