#!/usr/bin/env python
# coding: utf-8

# In[8]:


from lrcb.representations.finite_representations import LinearRepresentation, normalize_param, is_hls, rank, hls_rank, is_cmb, cmb_rank, hls_lambda
import numpy as np
TOL = 1e-8


# In[2]:


ids = np.load('../problem_data/openml/ids.npy')
len(ids)


# In[5]:


mode = 'regression'
neurons = (32,32)


# In[6]:


data = ['../problem_data/openml/openml_%s_id%d_dim%d_hid%d_seed0.npz'%(mode, 
                                                                     int(i), 
                                                                     neurons[1]+1,
                                                                     neurons[0]) 
        for i in ids]


# In[ ]:


count_hls = 0
count_weakhls = 0
count_nr = 0
count_cmb = 0
count_weakcmb = 0
count_intersection = 0
test_scores = []
train_scores = []
lambdas = []
for i, id in enumerate(ids):
    f = np.load(data[i])
    phi = f['features']
    theta = f['theta']
    test_score = f['test_score'].item()
    train_score = f['train_score'].item()
    test_scores.append(test_score)
    train_scores.append(train_score)
    rep = LinearRepresentation(phi, theta)
    rep = normalize_param(rep)
    
    #_rank = rank(rep, TOL)
    _hls_rank = hls_rank(rep, TOL)
    #_cmb_rank = cmb_rank(rep, TOL)
    
    #count_nr += _rank==rep.dim
    #count_hls += _hls_rank==rep.dim
    #count_cmb += _cmb_rank==rep.dim
    #count_weakhls += _rank==_hls_rank
    #count_weakcmb += _cmb_rank==_rank
    #count_intersection += (_cmb_rank==_rank and _hls_rank==_rank)
    
    lam = hls_lambda(rep)
    print('%d : %f' % (_hls_rank==rep.dim, lam))
    #print('%d: score=%f, hls=%d, weak_hls=%d' % (id, score, is_hls(rep), rank(rep)==hls_rank(rep)))

