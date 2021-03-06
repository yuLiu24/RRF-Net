#!/usr/bin/env python

# Make sure that caffe is on the python path:
caffe_root = '/home/titanx/RRF-Net/'  
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe
import numpy as np
import scipy.io as scio
import math
from caffe.proto import caffe_pb2

caffe.set_mode_gpu()

def main(argv):
	# set net config
	net = caffe.Net(caffe_root + 'models/Flickr30K/RRF-Net_t=3_deploy.prototxt',
		               caffe_root + 'models/Flickr30K/Flickr30K_RRF-Net_t=3_iter_6500.caffemodel', caffe.TEST)

        #transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        #transformer.set_transpose('data', (2,0,1))

        ## load image CNN feature
        img_features = scio.loadmat('./data/Flickr30K_image_CNN_feature_test.mat')
        img_features = img_features['features']
        ## load text HGLMM feature
        text_features = scio.loadmat('./data/Flickr30K_text_HGLMM_feature_test.mat')
        text_features = text_features['features']  

	counters = 0    
        
        img_embed = open('./models/Flickr30K_image_embedding_features.txt','w')
        text_embed = open('./models/Flickr30K_text_embedding_features.txt','w')

        ntest = 1000*5
        # main directory 
	for i in range(0,ntest):

            net.blobs['img_data'].data[0,:,0,0] = img_features[int(math.floor(i/5)),:] 
            net.blobs['text_data'].data[0,:,0,0] = text_features[i,:] 
            #transformer.preprocess('data', caffe.io.load_image(path))
	    scores = net.forward()

	    # get embedded features
            img_l2norm = net.blobs['img_l2norm'].data[0] 
            text_l2norm = net.blobs['text_l2norm'].data[0] 
            #print img_l2norm.shape

            print counters, counters%5

            if (counters%5 == 0) :
                # write image to txt file
	        img_l2norm = ' '.join(str(num1) for num1 in img_l2norm[:,0,0])
	        img_embed.write(img_l2norm + '\n') 

            # write text to txt file
	    text_l2norm = ' '.join(str(num2) for num2 in text_l2norm[:,0,0])
	    text_embed.write(text_l2norm + '\n') 

            counters = counters + 1

        text_embed.close()
        img_embed.close()

if __name__ == '__main__':
    main(sys.argv)

