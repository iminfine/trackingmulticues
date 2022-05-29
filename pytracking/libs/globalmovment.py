from .sekd import SEKD
from pytracking.evaluation.environment import env_settings
import numpy as np
import os
import cv2
import torch
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
def similarity_matcher(descriptors1, descriptors2, threshold=0.8):
    # Similarity threshold matcher for L2 normalized descriptors.
    device = descriptors1.device
    sim = descriptors1 @ descriptors2.t()
    nn_sim, nn12 = torch.max(sim, dim=1)
    ids1 = torch.arange(0, sim.shape[0], device=device)
    mask = (nn_sim >= threshold)
    matches = torch.stack([ids1[mask], nn12[mask]])
    return matches.t(),nn_sim[mask]

class globalmotion():
    def preprocess(self,img):
        img = img.astype('float32')/255.0      
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if self.resize_ratio<1:
            self.realwidth=int(self.resize_ratio * self.width)
            self.realheight=int(self.resize_ratio * self.height)
            img = cv2.resize(img, (self.realwidth,self.realheight))
        return img

    def plot_image_pair(self,imgs, dpi=100, size=3, pad=.5):
        n = len(imgs)
        figsize = (size*n, size*3/4) if size is not None else None
        try:
            self.ax[0].cla()
            self.ax[1].cla()
            self.ax[2].cla()
        except AttributeError:
            self.fig1, self.ax = plt.subplots(1, n, figsize=figsize, dpi=dpi)
            
        for i in range(n):
            self.ax[i].imshow(imgs[i], cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
            self.ax[i].get_yaxis().set_ticks([])
            self.ax[i].get_xaxis().set_ticks([])
            for spine in self.ax[i].spines.values():  # remove frame
                spine.set_visible(False)
        plt.tight_layout(pad=pad)

    def plot_keypoints(self,kpts0, kpts1, color='w', ps=2):
        self.ax[0].scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=ps)
        self.ax[1].scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=ps)


    def plot_matches(self,kpts0, kpts1, color, lw=1.5, ps=4):
        self.fig1.canvas.draw()

        transFigure = self.fig1.transFigure.inverted()
        fkpts0 = transFigure.transform(self.ax[0].transData.transform(kpts0))
        fkpts1 = transFigure.transform(self.ax[1].transData.transform(kpts1))

        self.fig1.lines = [matplotlib.lines.Line2D(
            (fkpts0[i, 0], fkpts1[i, 0]), (fkpts0[i, 1], fkpts1[i, 1]), zorder=1,
            transform=self.fig1.transFigure, c=color[i], linewidth=lw)
                    for i in range(len(kpts0))]
        self.ax[0].scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=ps)
        self.ax[1].scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=ps)
        self.ax[0].text(10,20,'#'+str(self.frame-1),color=(0,1,0),fontsize=15)
        self.ax[1].text(10,20,'#'+str(self.frame),color=(0,1,0),fontsize=15)   
        self.ax[0].set_xlabel('Keypoints detected in the previous frame.', fontsize=7)
        self.ax[1].set_xlabel('Keypoints detected in the current frame.', fontsize=7)
        self.ax[2].set_xlabel(r'Visualisation of the difference $\mathbf{p}_{%s}-\mathbf{p}_{%s}$.'%(self.frame,self.frame-1), fontsize=7)

    def plot_delta(self,prepos,delta):
        for i in range(prepos.shape[0]):
            self.ax[2].arrow(prepos.cpu().numpy()[i,:][0],prepos.cpu().numpy()[i,:][1], delta.cpu().numpy()[i,:][0],delta.cpu().numpy()[i,:][1], color='r',head_width = 2) 
        self.ax[2].text(10,20,'#'+str(self.frame),color=(0,1,0),fontsize=15)

    def __init__(self, img, use_cuda,visualizationSEKD):
        weights_path=os.path.join(env_settings().network_path, 'sekd.pth')
        self.frame=0
        max_height=480
        max_width=640
        confidence_threshold=0.55
        nms_radius=4
        max_keypoints=500
        multi_scale=False
        sub_pixel_location=False
        self.cuda=use_cuda
        self.visualization=visualizationSEKD
        self.net = SEKD(weights_path, confidence_threshold, nms_radius,
                         max_keypoints, self.cuda, multi_scale, sub_pixel_location) 
        
        self.height=img.shape[0]
        self.width=img.shape[1]       
        if self.height > max_height or self.width > max_width:
            self.resize_ratio = min(max_height / self.height,
                max_width / self.width)
        else:
            self.resize_ratio=1
            self.realwidth=self.width
            self.realheight=self.height
        self.preimage=img 
        img=self.preprocess(img)
        self.prekeypoints, self.predescriptors=self.net.detectAndCompute(img)
        self.prekeypoints=self.prekeypoints[0:2].T
        
    def __call__(self, image):
        self.frame+=1
        img=self.preprocess(image)
        self.keypoints, self.descriptors=self.net.detectAndCompute(img)
        if self.keypoints is None:
            return np.array([0,0])
        self.keypoints=self.keypoints[0:2].T
        matches,scores=similarity_matcher(self.predescriptors.T,self.descriptors.T)
        
        if len(matches)<10:
            self.prekeypoints=self.keypoints.clone()
            self.predescriptors=self.descriptors.clone()
            return np.array([0,0])
        
        prepos = self.prekeypoints[matches[:, 0],:2].clone()
        pos = self.keypoints[matches[:, 1],:2].clone()
        delta=pos-prepos
        
        if self.visualization:
            kp1=self.prekeypoints.cpu().numpy()
            kp2=self.keypoints.cpu().numpy()
            mkp1=prepos.cpu().numpy()
            mkp2=pos.cpu().numpy()
            self.plot_image_pair([self.preimage,image,image])
            self.plot_keypoints(kp1, kp2, color='w', ps=2)
            nmsocre=scores.cpu().numpy()
            self.plot_matches(mkp1,mkp2,cm.jet(np.interp(nmsocre, (nmsocre.min(), nmsocre.max()), (0, 1))))
            self.plot_delta(prepos,delta)
            plt.pause(0.00001)
            #plt.savefig('keypoints'+str(self.frame)+'.pdf',dpi=300,bbox_inches='tight')
        out=delta.mean(dim=0).cpu().numpy()
        self.prekeypoints=self.keypoints.clone()
        self.predescriptors=self.descriptors.clone()
        if any(np.isnan(out)):
            return np.array([0,0])
        return out/self.resize_ratio
            
        
      
        
        