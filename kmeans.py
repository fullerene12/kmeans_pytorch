import torch
import numpy as np
from kmeans_pytorch.pairwise import pairwise_distance

def forgy(X, n_clusters):
	_len = len(X)
	indices = np.random.choice(_len, n_clusters)
	initial_state = X[indices]
	return initial_state


def lloyd(X, min_clusters =2, max_clusters = 20, device=0, epoch=1):
	X = torch.from_numpy(X).float().cuda(device)

	initial_state = forgy(X, n_clusters)
    
    for n_clusters in range(min_clusters,max_clusters):
        for i in range(epoch):
            dis = pairwise_distance(X, initial_state)

            choice_cluster = torch.argmin(dis, dim=1)

            initial_state_pre = initial_state.clone()

            for index in range(n_clusters):
                selected = torch.nonzero(choice_cluster==index).squeeze()

                selected = torch.index_select(X, 0, selected)
                initial_state[index] = selected.mean(dim=0)
		
		center_shift = torch.sum(torch.sqrt(torch.sum((initial_state - initial_state_pre) ** 2, dim=1)))

	return choice_cluster.cpu().numpy(), initial_state.cpu().numpy()

def lloyd_batch(X, n_clusters=3, device=0, epoch=1, batch_size=100):
    n_tail = X.shape[0] % batch_size
    if n_tail==0:
        n_tail = batch_size
        n_batches = X.shape[0]//batch_size
    else:
        n_batches = X.shape[0]//batch_size + 1
    n_pad = batch_size - n_tail
    X_padded = np.pad(X, [(0,n_pad),(0,0)],mode='constant')
    
    BX = X_padded.reshape(n_batches,batch_size,-1)
    
    initial_state = torch.from_numpy(forgy(X,n_clusters)).float().cuda(device)
        
    for i in range(epoch):
        initial_state_new = torch.zeros(initial_state.shape).cuda(device)
        for n in range(n_batches):
            if n==n_batches-1:
                x_batch = torch.from_numpy(BX[n,:n_tail,:]).squeeze().float().cuda(device)
            else:
                x_batch = torch.from_numpy(BX[n,:,:]).squeeze().float().cuda(device)
            dis = pairwise_distance(x_batch, initial_state)
            choice_cluster = torch.argmin(dis,dim=1)
            
            
            for index in range(n_clusters):
                selected = torch.nonzero(choice_cluster==index).squeeze()
                selected = torch.index_select(x_batch, 0, selected)
                if len(selected) != 0:
                    initial_state_new[index] += selected.mean(dim=0)/batch_size/n_batches*x_batch.shape[0]
                
        center_shift = torch.sum(torch.sqrt(torch.sum((initial_state_new - initial_state) ** 2, dim=1)))
        initial_state = initial_state_new
    
    return choice_cluster.cpu().numpy(), initial_state.cpu().numpy()
