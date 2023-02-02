#-*- coding:UTF-8 -*-
import dgl
import torch
import numpy as np
import scipy.sparse as sp

def idx(mask):
    idx = []
    for i in range(len(mask)):
        if mask[i]:
            idx.append(i)
    return idx

def load_data(g_data='cora',device=0):
    gpu = lambda x: x
    if torch.cuda.is_available() and device >= 0:
        dev = torch.device('cuda:%d' % device)
        gpu = lambda x: x.to(dev)

    raw_dir = '../data/' + g_data
    graph = (
        dgl.data.CoraGraphDataset(raw_dir=raw_dir) if g_data == 'cora'
        else dgl.data.CiteseerGraphDataset(raw_dir=raw_dir) if g_data == 'citeseer'
        else dgl.data.PubmedGraphDataset(raw_dir=raw_dir) if g_data == 'pubmed'
        else dgl.data.CoraFullDataset(raw_dir=raw_dir) if g_data == 'corafull'
        else dgl.data.CoauthorCSDataset(raw_dir=raw_dir) if g_data == 'coauthor-cs'
        else dgl.data.CoauthorPhysicsDataset(raw_dir=raw_dir) if g_data == 'coauthor-phy'
        else dgl.data.RedditDataset(raw_dir=raw_dir) if g_data == 'reddit'
        else dgl.data.AmazonCoBuyComputerDataset(raw_dir=raw_dir)
        if g_data == 'amazon-com'
        else dgl.data.AmazonCoBuyPhotoDataset(raw_dir=raw_dir) if g_data == 'amazon-photo'
        else None
    )[0]

    if g_data=='cora' or g_data=='citeseer' or g_data=='pubmed':
        train_mask = graph.ndata.pop('train_mask')
        val_mask = graph.ndata.pop('val_mask')
        test_mask = graph.ndata.pop('test_mask')
        idx_train = idx(train_mask)
        idx_val = idx(val_mask)
        idx_test = idx(test_mask)

    else:
        idx_train = range(140)
        idx_val = range(200, 500)
        idx_test = range(500, 1500)

    X = node_features = gpu(graph.ndata['feat'])
    Y = node_labels = gpu(graph.ndata['label'])
    n_nodes = node_features.shape[0]
    nrange = torch.arange(n_nodes)
    n_features = node_features.shape[1]
    n_labels = int(Y.max().item() + 1)
    src, dst = graph.edges()
    n_edges = src.shape[0]
    is_bidir = ((dst == src[0]) & (src == dst[0])).any().item()
    degree = n_edges * (2 - is_bidir) / n_nodes
    return X, Y, src, dst,idx_train,idx_val,idx_test


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def normalize(mx):
    """Row-normalize sparse matrix"""
    if "torch" in str(type(mx)):
        rowsum = np.array(mx.cpu().sum(1))
    else:
        rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    if "torch" in str(type(mx)):
        mx = r_mat_inv.dot(mx.cpu())
    else:
        mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def process_dataset(dataset='cora',device = 0):

    gpu = lambda x: x
    if torch.cuda.is_available() and device >= 0:
        dev = torch.device('cuda:%d' % device)
        gpu = lambda x: x.to(dev)

    X, Y, src, dst, idx_train,idx_val,idx_test = load_data(dataset,device)

    X = normalize(X)
    features = torch.FloatTensor(X)
    labels = torch.LongTensor(Y.cpu())

    adj = sp.coo_matrix((np.ones(src.shape), (src, dst)),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    return gpu(adj), gpu(features), gpu(labels), gpu(idx_train), gpu(idx_val), gpu(idx_test)