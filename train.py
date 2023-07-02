import networkx as nx
import numpy as np
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold
import sys
import scipy
import sklearn
import json
from collections import defaultdict
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import argparse
import math
import pickle as pkl

def EuclideanDistances(a,b):
    sq_a = a**2
    sum_sq_a = torch.sum(sq_a,dim=1).unsqueeze(1)  # m->[m, 1]
    sq_b = b**2
    sum_sq_b = torch.sum(sq_b,dim=1).unsqueeze(0)  # n->[1, n]
    bt = b.t()
    return torch.sqrt(sum_sq_a+sum_sq_b-2*a.mm(bt))

class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        '''

        super(MLP, self).__init__()

        self.linear_or_not = True #default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            #Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            #Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            #If linear model
            return self.linear(x)
        else:
            #If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
            return self.linears[self.num_layers - 1](h)




class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return 0.5*x*(1+torch.tanh(np.sqrt(2/np.pi)*(x+0.044715*torch.pow(x,3))))


class GCN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GCN, self).__init__()
        self.fc=nn.Linear(in_dim,out_dim)
        self.gelu=GELU()


    def normalize(self, A , symmetric=True):
        # A = A+I
        A = A + torch.eye(A.size(0))
        # 所有节点的度
        d = A.sum(1)
        if symmetric:
            #D = D^-1/2
            D = torch.diag(torch.pow(d , -0.5))
            return D.mm(A).mm(D)
        else :
            # D=D^-1
            D =torch.diag(torch.pow(d,-1))
            return D.mm(A)


    def forward(self,A,X,relu=True):

        # or use softmax?
        #A_norm=self.normalize(A)

        A_norm=A-1e9*torch.less_equal(A,0.8)
        A_norm=A_norm.softmax(-1)



        return F.leaky_relu(A_norm.mm(self.fc(X))) if relu else A_norm.mm(self.fc(X))
        #return F.leaky_relu(A_norm.mm(self.fc(X)),negative_slope=0.01) if relu else A_norm.mm(self.fc(X))




class GraphCNN(nn.Module):
    def __init__(self, num_layers=5, num_mlp_layers=2, input_dim=200, hidden_dim=128, output_dim=200,
                 final_dropout=0.5, learn_eps=True, graph_pooling_type='sum', neighbor_pooling_type='sum',use_select_sim=False):
        '''
            num_layers: number of layers in the neural networks (INCLUDING the input layer)
            num_mlp_layers: number of layers in mlps (EXCLUDING the input layer)
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            final_dropout: dropout ratio on the final linear layer
            learn_eps: If True, learn epsilon to distinguish center nodes from neighboring nodes. If False, aggregate neighbors and center nodes altogether.
            neighbor_pooling_type: how to aggregate neighbors (mean, average, or max)
            graph_pooling_type: how to aggregate entire nodes in a graph (mean, average)
            device: which device to use
        '''

        super(GraphCNN, self).__init__()


        self.final_dropout = final_dropout
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        self.num_layers = num_layers
        self.graph_pooling_type = graph_pooling_type
        self.neighbor_pooling_type = neighbor_pooling_type
        self.use_select_sim=use_select_sim


        self.learn_eps = learn_eps
        self.eps = nn.Parameter(torch.zeros(self.num_layers-1))

        ###List of MLPs
        self.mlps = torch.nn.ModuleList()

        ###List of batchnorms applied to the output of MLP (input of the final prediction linear layer)
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers-1):
            if layer == 0:
                self.mlps.append(MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim))
            else:
                self.mlps.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))

            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))


    def __preprocess_neighbors_maxpool(self, batch_graph):
        ###create padded_neighbor_list in concatenated graph

        #compute the maximum number of neighbors within the graphs in the current minibatch
        max_deg = max([graph.max_neighbor for graph in batch_graph])

        padded_neighbor_list = []
        start_idx = [0]


        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))
            padded_neighbors = []
            for j in range(len(graph.neighbors)):
                #add off-set values to the neighbor indices
                pad = [n + start_idx[i] for n in graph.neighbors[j]]
                #padding, dummy data is assumed to be stored in -1
                pad.extend([-1]*(max_deg - len(pad)))

                #Add center nodes in the maxpooling if learn_eps is False, i.e., aggregate center nodes and neighbor nodes altogether.
                if not self.learn_eps:
                    pad.append(j + start_idx[i])

                padded_neighbors.append(pad)
            padded_neighbor_list.extend(padded_neighbors)

        return torch.LongTensor(padded_neighbor_list)


    def __preprocess_neighbors_sumavepool(self, batch_graph):
        ###create block diagonal sparse matrix

        edge_mat_list = []
        start_idx = [0]
        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))
            edge_mat_list.append(graph.edge_mat + start_idx[i])
        Adj_block_idx = torch.cat(edge_mat_list, 1)
        Adj_block_elem = torch.ones(Adj_block_idx.shape[1])

        #Add self-loops in the adjacency matrix if learn_eps is False, i.e., aggregate center nodes and neighbor nodes altogether.

        if not self.learn_eps:
            num_node = start_idx[-1]
            self_loop_edge = torch.LongTensor([range(num_node), range(num_node)])
            elem = torch.ones(num_node)
            Adj_block_idx = torch.cat([Adj_block_idx, self_loop_edge], 1)
            Adj_block_elem = torch.cat([Adj_block_elem, elem], 0)

        Adj_block = torch.sparse.FloatTensor(Adj_block_idx, Adj_block_elem, torch.Size([start_idx[-1],start_idx[-1]]))

        return Adj_block.to(self.device), Adj_block_idx


    def __preprocess_graphpool(self, batch_graph):
        ###create sum or average pooling sparse matrix over entire nodes in each graph (num graphs x num nodes)

        start_idx = [0]

        #compute the padded neighbor list
        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))

        idx = []
        elem = []
        for i, graph in enumerate(batch_graph):
            ###average pooling
            if self.graph_pooling_type == "average":
                elem.extend([1./len(graph.g)]*len(graph.g))

            else:
                ###sum pooling
                elem.extend([1]*len(graph.g))

            idx.extend([[i, j] for j in range(start_idx[i], start_idx[i+1], 1)])
        elem = torch.FloatTensor(elem)
        idx = torch.LongTensor(idx).transpose(0,1)
        graph_pool = torch.sparse.FloatTensor(idx, elem, torch.Size([len(batch_graph), start_idx[-1]]))

        return graph_pool.to(self.device)

    def maxpool(self, h, padded_neighbor_list):
        ###Element-wise minimum will never affect max-pooling

        dummy = torch.min(h, dim = 0)[0]
        h_with_dummy = torch.cat([h, dummy.reshape((1, -1)).to(self.device)])
        pooled_rep = torch.max(h_with_dummy[padded_neighbor_list], dim = 1)[0]
        return pooled_rep


    def next_layer_eps(self, h, layer, padded_neighbor_list = None, Adj_block = None):
        ###pooling neighboring nodes and center nodes separately by epsilon reweighting.

        if self.neighbor_pooling_type == "max":
            ##If max pooling
            pooled = self.maxpool(h, padded_neighbor_list)
        else:
            #If sum or average pooling
            pooled = torch.spmm(Adj_block, h)
            if self.neighbor_pooling_type == "average":
                #If average pooling
                degree = torch.spmm(Adj_block, torch.ones((Adj_block.shape[0], 1)).to(self.device))
                pooled = pooled/degree

        #Reweights the center node representation when aggregating it with its neighbors
        pooled = pooled + (1 + self.eps[layer])*h
        pooled_rep = self.mlps[layer](pooled)
        h = self.batch_norms[layer](pooled_rep)

        #non-linearity
        h = F.relu(h)
        return h


    def next_layer(self, h, layer, padded_neighbor_list = None, Adj_block = None):
        ###pooling neighboring nodes and center nodes altogether

        if self.neighbor_pooling_type == "max":
            ##If max pooling
            pooled = self.maxpool(h, padded_neighbor_list)
        else:
            #If sum or average pooling
            pooled = torch.spmm(Adj_block, h)
            if self.neighbor_pooling_type == "average":
                #If average pooling
                degree = torch.spmm(Adj_block, torch.ones((Adj_block.shape[0], 1)).to(self.device))
                pooled = pooled/degree

        #representation of neighboring and center nodes
        pooled_rep = self.mlps[layer](pooled)

        h = self.batch_norms[layer](pooled_rep)

        #non-linearity
        h = F.relu(h)
        return h


    def forward(self, batch_graph):


        X_concat = torch.cat([graph.node_features for graph in batch_graph], 0).to(self.device)

        graph_pool = self.__preprocess_graphpool(batch_graph)

        if self.neighbor_pooling_type == "max":
            padded_neighbor_list = self.__preprocess_neighbors_maxpool(batch_graph)
        else:
            Adj_block, Adj_block_idx = self.__preprocess_neighbors_sumavepool(batch_graph)

        #list of hidden representation at each layer (including input)
        hidden_rep = [X_concat]
        h = X_concat

        for layer in range(self.num_layers-1):
            if self.neighbor_pooling_type == "max" and self.learn_eps:
                h = self.next_layer_eps(h, layer, padded_neighbor_list = padded_neighbor_list)
            elif not self.neighbor_pooling_type == "max" and self.learn_eps:
                h = self.next_layer_eps(h, layer, Adj_block = Adj_block)
            elif self.neighbor_pooling_type == "max" and not self.learn_eps:
                h = self.next_layer(h, layer, padded_neighbor_list = padded_neighbor_list)
            elif not self.neighbor_pooling_type == "max" and not self.learn_eps:
                h = self.next_layer(h, layer, Adj_block = Adj_block)

            hidden_rep.append(h)



        final_hidd= []
        if self.use_select_sim:
            for layer in range(self.num_layers):
                start_idx = 0
                select_hidden=[]
                for i, graph in enumerate(batch_graph):

                    g=graph.g
                    adj=torch.tensor(nx.adj_matrix(g).toarray(),dtype=torch.float).cuda()
                    d_inv=torch.diag(torch.tensor([1/one[1] for one in list(g.degree)],dtype=torch.float)).cuda()
                    hidden_all_nodes=hidden_rep[layer][start_idx:start_idx+len(g)]
                    importance= torch.norm((torch.eye(len(g)).cuda()-d_inv.matmul(adj)).matmul(hidden_all_nodes),p=1,dim=-1)

                    select_hidden.append(hidden_all_nodes[torch.argmax(importance)].unsqueeze(0))
                    start_idx+= len(g)

                final_hidd.append(torch.cat(select_hidden,0))


        pooled_h_layers = []

        #perform pooling over all nodes in each graph in every layer
        for layer, h in enumerate(hidden_rep):
            pooled_h = torch.spmm(graph_pool, h)
            pooled_h_layers.append(pooled_h)
            #pooled_h_layers.append(F.dropout(pooled_h))



        #return pooled_h_layers, h, Adj_block_idx, hidden_rep,final_hidd
        return pooled_h_layers, h, Adj_block_idx, hidden_rep,final_hidd


class Graph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
            edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor #[2, Number of edges]
            neighbors: list of neighbors (without self-loop)
        '''
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors = []
        self.node_features = 0
        self.edge_mat = 0

        self.max_neighbor = 0


def load_data(dataset, degree_as_tag):
    '''
        dataset: name of dataset
        test_proportion: ratio of test train split
        seed: random seed for random splitting of dataset
    '''

    print('loading data')

    if dataset != 'ogbg-ppa':
        g_list = []
        label_dict = {}
        feat_dict = {}

        with open('./datasets/%s/%s.txt' % (dataset, dataset), 'r') as f:
            n_g = int(f.readline().strip())
            for i in range(n_g):
                row = f.readline().strip().split()
                n, l = [int(w) for w in row]
                if not l in label_dict:
                    mapped = len(label_dict)
                    label_dict[l] = mapped
                g = nx.Graph()
                node_tags = []
                node_features = []
                n_edges = 0
                for j in range(n):
                    g.add_node(j)
                    row = f.readline().strip().split()
                    tmp = int(row[1]) + 2
                    if tmp == len(row):
                        # no node attributes
                        row = [int(w) for w in row]
                        attr = None
                    else:
                        row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])
                    if not row[0] in feat_dict:
                        mapped = len(feat_dict)
                        feat_dict[row[0]] = mapped
                    node_tags.append(feat_dict[row[0]])

                    if tmp > len(row):
                        node_features.append(attr)

                    n_edges += row[1]
                    for k in range(2, len(row)):
                        g.add_edge(j, row[k])

                if node_features != []:
                    node_features = np.stack(node_features)
                    node_feature_flag = True
                else:
                    node_features = None
                    node_feature_flag = False

                assert len(g) == n

                g_list.append(Graph(g, l, node_tags))

        # add labels and edge_mat
        for g in g_list:
            g.neighbors = [[] for i in range(len(g.g))]
            for i, j in g.g.edges():
                g.neighbors[i].append(j)
                g.neighbors[j].append(i)
            degree_list = []
            for i in range(len(g.g)):
                g.neighbors[i] = g.neighbors[i]
                degree_list.append(len(g.neighbors[i]))
            g.max_neighbor = max(degree_list)

            # g.label = label_dict[g.label]

            edges = [list(pair) for pair in g.g.edges()]
            edges.extend([[i, j] for j, i in edges])

            deg_list = list(dict(g.g.degree(range(len(g.g)))).values())
            g.edge_mat = torch.LongTensor(edges).transpose(0, 1)

        if degree_as_tag:
            for g in g_list:
                g.node_tags = list(dict(g.g.degree).values())
                if np.sum(np.array(g.node_tags)==0):print(g.node_tags)

        # Extracting unique tag labels
        tagset = set([])
        for g in g_list:
            tagset = tagset.union(set(g.node_tags))

        tagset = list(tagset)
        tag2index = {tagset[i]: i for i in range(len(tagset))}

        for g in g_list:
            g.node_features = torch.zeros(len(g.node_tags), len(tagset))
            g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1

        print('# classes: %d' % len(label_dict))
        print('# maximum node tag: %d' % len(tagset))

        print("# data: %d" % len(g_list), "\n")

        return g_list, label_dict, tagset

    else:
        from ogb.graphproppred import GraphPropPredDataset

        dataset = GraphPropPredDataset(name=dataset)

        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]

        g_list = []
        ### set i as an arbitrary index
        for i in range(len(dataset)):

            graph, label = dataset[i]  # graph: library-agnostic graph object

            nx_graph = nx.Graph()
            for j in range(graph['num_nodes']):
                nx_graph.add_node(j)
            for j in range(graph['edge_index'].shape[1]):
                nx_graph.add_edge(graph['edge_index'][j, 0], graph['edge_index'][j, 1])

            g = Graph(nx_graph, label)
            g.edge_mat = torch.LongTensor(graph['edge_index'])
            g.node_features = torch.FloatTensor(graph['node_feat'])

            g_list.append(g)

        return g_list, {i: i for i in range(37)}, tagset


class Dataset:
    def __init__(self, name, args):
        self.dataset_name = name
        self.args=args
        self.train_graphs = []
        self.test_graphs = []

        all_graphs, label_dict,tagset = load_data(self.dataset_name, True)
        all_classes = list(label_dict.keys())

        self.tagset=tagset

        with open("./split/{}/main_splits.json".format(args.dataset_name), "r") as f:
            all_class_splits = json.load(f)
            self.train_classes = all_class_splits["train"]
            self.test_classes = all_class_splits["test"]

        train_classes_mapping = {}
        for cl in self.train_classes:
            train_classes_mapping[cl] = len(train_classes_mapping)
        self.train_classes_num = len(train_classes_mapping)

        test_classes_mapping = {}
        for cl in self.test_classes:
            test_classes_mapping[cl] = len(test_classes_mapping)
        self.test_classes_num = len(test_classes_mapping)

        for i in range(len(all_graphs)):
            if all_graphs[i].label in self.train_classes:
                self.train_graphs.append(all_graphs[i])

            if all_graphs[i].label in self.test_classes:
                self.test_graphs.append(all_graphs[i])

        for graph in self.train_graphs:
            graph.label = train_classes_mapping[int(graph.label)]
        for i, graph in enumerate(self.test_graphs):
            graph.label = test_classes_mapping[int(graph.label)]

        #
        num_validation_graphs = math.floor(0.2 * len(self.train_graphs))
        #num_validation_graphs = math.floor(0 * len(self.train_graphs))

        np.random.seed(seed_value)
        np.random.shuffle(self.train_graphs)

        self.train_graphs = self.train_graphs[: len(self.train_graphs) - num_validation_graphs]
        self.validation_graphs = self.train_graphs[len(self.train_graphs) - num_validation_graphs:]

        self.train_tasks = defaultdict(list)
        for graph in self.train_graphs:
            self.train_tasks[graph.label].append(graph)

        self.valid_tasks = defaultdict(list)
        for graph in self.validation_graphs:
            self.valid_tasks[graph.label].append(graph)


        #np.random.seed(2)
        np.random.seed(seed_value)
        np.random.shuffle(self.test_graphs)

        self.test_graphs=self.test_graphs[:self.args.K_shot+self.args.N_way*(self.args.query_size)*200]

        nx.Graph().number_of_nodes()

        self.test_tasks = defaultdict(list)
        for graph in self.test_graphs:
            self.test_tasks[graph.label].append(graph)

        self.total_test_g_list=[]
        for index in range(self.test_classes_num):
            self.total_test_g_list.extend(list(self.test_tasks[index])[self.args.K_shot:])

        from numpy.random import RandomState
        rd=RandomState(0)


        rd.shuffle(self.total_test_g_list)

    def sample_P_tasks(self, task_source, P_num_task, sample_rate, N_way, K_shot, query_size):

        tasks = []
        support_classes = []
        for i in range(P_num_task):
            chosen_class = np.random.choice(list(range(sample_rate.shape[0])), N_way,p=sample_rate, replace=False)
            support_classes.append(chosen_class)
            tasks.append(self.sample_one_task(task_source, chosen_class, K_shot=K_shot, query_size=query_size))

        return tasks, support_classes

    def sample_one_task(self, task_source, class_index, K_shot, query_size,test_start_idx=None):

        support_set = []
        query_set = []
        for index in class_index:
            g_list = list(task_source[index])
            if task_source == self.train_tasks or test_start_idx==None:
                np.random.shuffle(g_list)

            support_set.append(g_list[:K_shot])
            if task_source==self.train_tasks or test_start_idx==None:
                query_set.append(g_list[K_shot:K_shot + query_size])

        #during test, sample from all test samples
        append_count=0
        if task_source==self.test_tasks and test_start_idx!=None:
            for i in range(len(class_index)):
                query_set.append(self.total_test_g_list[min(test_start_idx+i*query_size,len(self.total_test_g_list)):min(test_start_idx+(i+1)*query_size,len(self.total_test_g_list))])
                while len(query_set[-1])<query_size:
                    query_set[-1].append(query_set[0][-1])
                    append_count+=1



        return {'support_set':support_set,'query_set':query_set,'append_count':append_count}




class Model(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.P=args.P_num
        self.N=args.N_way
        self.K=args.K_shot
        self.Q=args.query_size
        #self.dropout=args.dropout

        self.sample_input_emb_size=args.sample_input_size
        self.proto_input_emb_size=self.sample_out_emb_size=300
        self.task_input_emb_size=self.proto_out_emb_size=300
        self.task_out_emb_size=300

        self.gin=GraphCNN(input_dim=args.node_fea_size,use_select_sim=args.use_select_sim,num_layers=args.gin_layer,hidden_dim=args.gin_hid).cuda()

        self.sample_soft_assign_linear=nn.Linear(self.sample_input_emb_size,self.sample_input_emb_size)
        self.sample_GCN_fea=GCN(self.sample_input_emb_size,self.proto_input_emb_size)
        self.sample_GCN_agg=GCN(self.sample_input_emb_size,1)

        self.proto_soft_assign_linear=nn.Linear(self.proto_input_emb_size,self.proto_input_emb_size)
        self.proto_GCN_fea=GCN(self.proto_input_emb_size,self.task_input_emb_size)
        self.proto_GCN_agg=GCN(self.proto_input_emb_size,1)

        self.task_soft_assign_linear=nn.Linear(self.task_input_emb_size,self.task_input_emb_size)
        self.task_GCN_fea=GCN(self.task_input_emb_size,self.task_out_emb_size)


        self.classify_linear=nn.Linear(self.sample_out_emb_size,self.proto_out_emb_size)
        self.base_classifier=nn.Linear(self.sample_input_emb_size,args.train_classes_num)

        if args.baseline_mode=='relation':
            self.rel_classifier=nn.Linear(self.sample_input_emb_size*2,args.train_classes_num)

        self.dropout=nn.Dropout(args.dropout)



    def sample_input_GNN(self, tasks):
        embs=[]
        final_hidds=[]
        for task in tasks:
            graphs=[]

            for i in range(len(task['support_set'])):
                graphs.extend(task['support_set'][i]+task['query_set'][i])

            pooled_h_layers, node_embeds, Adj_block_idx, hidden_rep,final_hidd =self.gin(graphs)  #[N(K+Q), emb_size]
            embs.append(torch.cat(pooled_h_layers[1:],-1))
            final_hidds.append(final_hidd)
        return torch.cat(embs,0), [torch.cat([one[layer] for one in final_hidds],0) for layer in range(self.gin.num_layers)] if self.args.use_select_sim else []

    def construct_sample_graph(self, sample_embs, support_classes,sample_embs_selected):


        # --calculate soft adj
        if not self.args.use_select_sim:
            soft_adj=torch.matmul(sample_embs/(sample_embs.norm(p=2,dim=-1,keepdim=True)+1e-9),(sample_embs/(sample_embs.norm(p=2,dim=-1,keepdim=True)+1e-9)).t())  #[(P+1)NK, (P+1)NK]
        else:
            sims=[]
            for emb_per_layer in sample_embs_selected:
                sims.append(torch.matmul(emb_per_layer/emb_per_layer.norm(p=2,dim=-1,keepdim=True),(emb_per_layer/emb_per_layer.norm(p=2,dim=-1,keepdim=True)).t()))
            soft_adj=torch.stack(sims,2).max(-1)[0]

        # --calculate hard adj
        hard_adj=torch.zeros(soft_adj.shape).cuda()
        for j in range((self.P+1)*self.N):
            hard_adj[j*self.K:(j+1)*self.K,j*self.K:(j+1)*self.K]=1

        # --combine them
        final_adj=soft_adj+hard_adj


        # input into GCN
        sample_output_embs=self.sample_GCN_fea(final_adj,sample_embs) #[(P+1)N(K+Q), sample_out_emb_size]
        sample_output_agg=self.sample_GCN_agg(final_adj,sample_embs,relu=False)  #[(P+1)N(K+Q), 1]
        #split sample embs
        sample_output_embs_reshape=sample_output_embs.reshape(((self.P+1)*self.N,self.K+self.Q,self.sample_out_emb_size))
        sample_output_embs_support=sample_output_embs_reshape[:,:self.K,:].reshape(((self.P+1)*self.N*self.K,self.sample_out_emb_size)) #[(P+1)NK, smaple_output_emb_size]
        sample_output_embs_query=sample_output_embs_reshape[:,self.K:,:].reshape(((self.P+1)*self.N*self.Q,self.sample_out_emb_size))#[(P+1)NQ, smaple_output_emb_size]

        sample_output_agg_reshape=sample_output_agg.reshape(((self.P+1)*self.N,self.K+self.Q,1))
        sample_output_agg_support=sample_output_agg_reshape[:,:self.K,:].reshape(((self.P+1)*self.N*self.K,1)) #[(P+1)NK, smaple_output_emb_size]

        # output embs for proto layer
        agg_matrix=sample_output_agg_support.reshape(((self.P+1)*self.N,1,self.K)).softmax(-1) #[(P+1)N, 1, K]
        sample_emb_reshape=sample_output_embs_support.reshape(((self.P+1)*self.N,self.K,-1))  #[(P+1)N, K, sample_out_emb_size]
        proto_input_embs=torch.matmul(agg_matrix,sample_emb_reshape).squeeze() #[(P+1)N, sample_out_emb_size]


        return sample_output_embs_query,proto_input_embs

    def construct_proto_graph(self, proto_input_embs, support_classes):


        # --calculate soft adj
        soft_adj=torch.matmul(proto_input_embs/(proto_input_embs.norm(p=2,dim=-1,keepdim=True)+1e-9),(proto_input_embs/(proto_input_embs.norm(p=2,dim=-1,keepdim=True)+1e-9)).t())  #[(P+1)N, (P+1)N]

        # --calculate hard adj
        hard_adj=0
        # --combine them
        final_adj=soft_adj+hard_adj
        # input into GCN
        proto_output_embs=self.proto_GCN_fea(final_adj,proto_input_embs) #[(P+1)N, proto_out_emb_size]
        proto_output_agg=self.proto_GCN_agg(final_adj,proto_input_embs,relu=False)  #[(P+1)N, 1]
        # output embs for task layer
        agg_matrix=proto_output_agg.reshape(((self.P+1),1,self.N)).softmax(-1) #[(P+1), 1, N]
        proto_emb_reshape=proto_output_embs.reshape(((self.P+1),self.N,-1))  #[(P+1), N, proto_out_emb_size]
        task_input_embs=torch.matmul(agg_matrix,proto_emb_reshape).squeeze() #[(P+1), proto_out_emb_size]


        return proto_output_embs, self.dropout(task_input_embs)

    def construct_task_graph(self, task_input_embs, support_classes):
        # --calculate soft adj
        soft_adj=torch.matmul(task_input_embs/(task_input_embs.norm(p=2,dim=-1,keepdim=True)+1e-9),(task_input_embs/(task_input_embs.norm(p=2,dim=-1,keepdim=True)+1e-9)).t())  #[(P+1), (P+1)]

        # --calculate hard adj
        hard_adj=0
        # --combine them
        final_adj=soft_adj+hard_adj
        # input into GCN
        task_output_embs=self.task_GCN_fea(final_adj,task_input_embs) #[(P+1), proto_out_emb_size]


        return self.dropout(task_output_embs)

    def classify_tasks(self,sample_emb,proto_emb,task_emb):
        sample_emb_reshape=sample_emb.reshape((self.P+1,self.N,self.Q,-1)) #[P+1, N, Q, sample_out_emb_size]
        proto_emb_reshape_=proto_emb.reshape((self.P+1,self.N,-1)) #[P+1, N, proto_out_emb_size)
        proto_emb_reshape=proto_emb_reshape_.unsqueeze(1).repeat([1,self.N,1,1]) #[P+1, N, N, proto_out_emb_size]

        task_emb_reshape=task_emb.unsqueeze(1).unsqueeze(1).repeat([1,self.N,self.N,1]) #[P+1, N, N, task_out_emb_size]


        use_dis=True
        if not use_dis:
            result=self.classify_linear(sample_emb_reshape).matmul((proto_emb_reshape*task_emb_reshape).transpose(-1,-2)) #[P+1, N, Q, N]
        else:
            def compute_l2(sample,proto,task):
                diff=torch.square(proto.unsqueeze(0)-sample.unsqueeze(1)) #[Q, N, emb_size]
                weighted_diff=diff.matmul(task.unsqueeze(1)).squeeze() #[Q,N]

                return weighted_diff

            temp=[]
            #sample_emb_reshape=self.classify_linear(sample_emb_reshape)
            for i in range(self.P+1):
                proto_emb_=proto_emb_reshape_[i,:,:]
                task_emb_=torch.sigmoid(task_emb[i,:])
                for j in range(self.N):
                    sample_emb_=sample_emb_reshape[i,j,:,:]
                    distance=compute_l2(sample_emb_,proto_emb_,task_emb_)
                    temp.append(distance)
            result=-torch.cat(temp,0)



            return result



        return self.dropout(result.reshape((-1,self.N)))


class Trainer:
    def __init__(self, args):
        self.args = args
        self.epoch_num = args.epoch_num
        self.P_num = args.P_num
        #self.N_way = args.N_way
        self.K_shot = args.K_shot
        self.query_size = args.query_size
        self.eval_interval=args.eval_interval
        self.test_task_num=args.test_task_num


        self.dataset = Dataset(args.dataset_name, args)
        args.train_classes_num=self.dataset.train_classes_num
        args.node_fea_size=self.dataset.train_graphs[0].node_features.shape[1]
        args.sample_input_size=(args.gin_layer-1)*args.gin_hid

        args.N_way=self.dataset.test_classes_num
        self.N_way=self.dataset.test_classes_num


        self.baseline_mode=args.baseline_mode


        self.model = Model(args).cuda()

        self.N_sample_prob = np.ones([self.dataset.train_classes_num]) / self.dataset.train_classes_num

        self.use_loss_based_prob=args.use_loss_based_prob
        self.loss_based_prob=torch.ones([100,self.dataset.train_classes_num]).cuda()*10

        self.optimizer= optim.Adam(self.model.parameters(), lr=args.lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=args.weight_decay)

        self.criterion = nn.CrossEntropyLoss()


    def train(self):
        best_test_acc=0
        best_valid_acc=0

        train_accs=[]
        for i in range(self.epoch_num):
            loss,acc,class_loss=self.train_one_step(mode='train',epoch=i)

            train_accs.append(np.mean(acc))

            if loss==None:continue

            if i%50==0:
                self.scheduler.step()
                print('Epoch {} Train Acc {:.4f} Loss {:.4f} Class Loss {:.4f}'.format(i,np.mean(train_accs),loss,class_loss))
                f.write('Epoch {} Train Acc {:.4f} Loss {:.4f} Class Loss {:.4f}'.format(i,np.mean(train_accs),loss,class_loss)+'\n')


            if i%self.eval_interval==0:
                with torch.no_grad():
                    test_accs=[]
                    start_test_idx=0
                    while start_test_idx<len(self.dataset.test_graphs)-self.K_shot*self.dataset.test_classes_num:
                        loss,test_acc,class_loss=self.train_one_step(mode='test',epoch=i,test_idx=start_test_idx)
                        if loss==None:continue
                        test_accs.extend(test_acc.tolist())
                        start_test_idx+=self.N_way*self.query_size

                    print('test task num',len(test_accs))
                    mean_acc=sum(test_accs)/len(test_accs)
                    if mean_acc>best_test_acc:
                        best_test_acc=mean_acc

                    print('Mean Test Acc {:.4f}  Best Test Acc {:.4f}'.format(mean_acc,best_test_acc))
                    f.write('Mean Test Acc {:.4f}  Best Test Acc {:.4f}'.format(mean_acc,best_test_acc)+'\n')

                    test_accs=[]
                    start_test_idx=0
                    while start_test_idx<len(self.dataset.validation_graphs)-self.K_shot*self.dataset.train_classes_num:
                        loss,test_acc,class_loss=self.train_one_step(mode='valid',epoch=i,test_idx=start_test_idx)
                        if loss==None:continue
                        test_accs.extend(test_acc.tolist())
                        start_test_idx+=self.N_way*self.query_size

                    print('test task num',len(test_accs))
                    mean_acc=sum(test_accs)/len(test_accs)
                    if mean_acc>best_valid_acc:
                        best_valid_acc=mean_acc

                    print('Mean Valid Acc {:.4f}  Best Valid Acc {:.4f}'.format(mean_acc,best_valid_acc))
                    f.write('Mean Valid Acc {:.4f}  Best Valid Acc {:.4f}'.format(mean_acc,best_valid_acc)+'\n')

        return best_test_acc




    def train_one_step(self,mode,epoch,test_idx=None, baseline_mode=None):
        if mode=='train':
            self.model.train()
            if self.use_loss_based_prob:
                p=(self.loss_based_prob-(self.loss_based_prob-20).relu()).mean(0).softmax(-1).cpu().detach().numpy()
                if epoch%50==0:
                    print(self.loss_based_prob.mean(0))
                if np.isnan(p).sum()>0:
                    print(self.loss_based_prob)
                    return None, None, None
            else:
                p=self.N_sample_prob



            first_N_class_sample = np.random.choice(list(range(self.dataset.train_classes_num)), self.N_way,
                                                    p=p , replace=False)
            current_task = self.dataset.sample_one_task(self.dataset.train_tasks, first_N_class_sample, K_shot=self.K_shot,
                                                        query_size=self.query_size)
        elif mode=='test':
            self.model.eval()
            first_N_class_sample = np.array(list(range(self.dataset.test_classes_num)))
            current_task = self.dataset.sample_one_task(self.dataset.test_tasks, first_N_class_sample, K_shot=self.K_shot,
                                                        query_size=self.query_size,test_start_idx=test_idx)
        elif mode=='valid':
            self.model.eval()
            first_N_class_sample = np.array(list(range(self.dataset.test_classes_num)))
            current_task = self.dataset.sample_one_task(self.dataset.test_tasks, first_N_class_sample, K_shot=self.K_shot,
                                                        query_size=self.query_size,test_start_idx=test_idx)





        if self.baseline_mode=='proto' or self.baseline_mode=='relation':


            current_sample_input_embs,current_sample_input_embs_selected = self.model.sample_input_GNN([current_task]) #[N(K+Q), emb_size]

            input_embs=current_sample_input_embs.reshape([self.N_way,self.K_shot+self.query_size,-1])
            support_embs=input_embs[:,:self.K_shot,:]
            query_embs=input_embs[:,self.K_shot:,:] #[N, q, emb_size]

            support_protos=support_embs.mean(1)  #[N, emb_size]

            if self.baseline_mode=='proto':
                scores=-EuclideanDistances(query_embs.reshape([self.N_way*self.query_size,-1]),support_protos)
            elif self.baseline_mode=='relation':
                scores=self.model.rel_classifier(torch.cat([support_protos.unsqueeze(1).repeat([1,self.query_size,1]).reshape(self.N_way*self.query_size,-1),query_embs.reshape([self.N_way*self.query_size,-1])],dim=-1))


            if mode=='train':
                label=torch.tensor(np.array(list(range(self.N_way)))).cuda()
                label=label.unsqueeze(0).repeat([self.query_size,1]).t()
                label=label.reshape([self.N_way*self.query_size])

            else:
                labels=[]
                for graphs in current_task['query_set']:
                    labels.append(torch.tensor(np.array([graph.label for graph in graphs])))
                label=torch.cat(labels,-1).cuda()

            y_preds=torch.argmax(scores,dim=1)

            if current_task['append_count']!=0:
                scores=scores[:label.shape[0]-current_task['append_count'],:]
                y_preds=y_preds[:label.shape[0]-current_task['append_count']]
                label=label[:label.shape[0]-current_task['append_count']]




            acc=(y_preds==label).float().cpu().numpy()
            loss=self.criterion(scores,label)

            if mode=='train':

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            return loss,acc,0

        # --calculte similarities (conduct base classification)
        current_sample_input_embs,current_sample_input_embs_selected = self.model.sample_input_GNN([current_task]) #[N(K+Q), emb_size]

        classifiy_result=self.model.base_classifier(current_sample_input_embs.reshape([self.N_way,self.K_shot+self.query_size,self.model.sample_input_emb_size]).mean(1)) #[N, N]

        loss_type=nn.CrossEntropyLoss(reduction='none')

        class_loss=loss_type(classifiy_result,torch.tensor(first_N_class_sample).cuda())


        if torch.isnan(class_loss).sum()>0:
            print(current_sample_input_embs)
            print(class_loss)
            print(classifiy_result)
            print(first_N_class_sample)
            print(1/0)
            return None,None,None

        if self.use_loss_based_prob and mode=='train':
            self.loss_based_prob[epoch%100,first_N_class_sample]=class_loss
            if torch.isnan(self.loss_based_prob).sum()>0:
                return None, None, None

        sim_matrix=classifiy_result.softmax(-1)

        sample_rate=sim_matrix.sum(0).softmax(-1).cpu().detach().numpy()


        exclude_self=False
        if exclude_self and mode=='train':
            sample_rate[first_N_class_sample]=0


        P_tasks, support_classes = self.dataset.sample_P_tasks(self.dataset.train_tasks, self.P_num, (sample_rate/sample_rate.sum()),
                                                               N_way=self.N_way, K_shot=self.K_shot,
                                                               query_size=self.query_size)

        test_sample_test=False
        if test_sample_test and mode=='test':

            P_tasks, support_classes = self.dataset.sample_P_tasks(self.dataset.test_tasks, self.P_num, np.ones([self.dataset.test_classes_num]) / self.dataset.test_classes_num,
                                                                   N_way=self.N_way, K_shot=self.K_shot,
                                                                   query_size=self.query_size)



        support_classes=[first_N_class_sample]+support_classes

        total_tasks = [current_task] + P_tasks
        sample_input_embs,sample_input_embs_selected = self.model.sample_input_GNN(total_tasks) #[(P+1)NK, emb_size]

        sample_output_embs_query,proto_input_embs = self.model.construct_sample_graph(sample_input_embs, support_classes,sample_input_embs_selected)
        #split the sample_output as support and query


        proto_output_embs,task_input_embs = self.model.construct_proto_graph(proto_input_embs , support_classes)
        task_output_embs = self.model.construct_task_graph(task_input_embs, support_classes)

        # --final classification

        #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)



        scores=self.model.classify_tasks(sample_output_embs_query,proto_input_embs,task_output_embs) #[(P+1)NQ, N]



        if mode=='train':
            label=torch.tensor(np.array(list(range(self.N_way)))).cuda()
            label=label.unsqueeze(0).repeat([self.query_size,1]).t()
            label=label.unsqueeze(0).repeat([(self.P_num+1),1,1]).reshape([(self.P_num+1)*self.N_way*self.query_size])

            current_only=False
            if current_only:
                label=label.reshape([self.P_num+1,self.N_way,self.query_size])[0,:,:].reshape(self.N_way*self.query_size)
                scores=scores.reshape([self.P_num+1,self.N_way,self.query_size,self.N_way])[0,:,:,:].reshape(self.N_way*self.query_size,self.N_way)

        else:
            labels=[]
            for graphs in current_task['query_set']:
                labels.append(torch.tensor(np.array([graph.label for graph in graphs])))
            label=torch.cat(labels,-1).cuda()
            scores=scores.reshape([self.P_num+1,self.N_way,self.query_size,self.N_way])[0,:,:,:].reshape(self.N_way*self.query_size,self.N_way)


        y_preds=torch.argmax(scores,dim=1)

        if current_task['append_count']!=0:
            scores=scores[:label.shape[0]-current_task['append_count'],:]
            y_preds=y_preds[:label.shape[0]-current_task['append_count']]
            label=label[:label.shape[0]-current_task['append_count']]

        acc=(y_preds==label).float().cpu().numpy()
        loss=self.criterion(scores,label)+class_loss.mean()*0.001


        if mode=='train':

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss,acc,class_loss.mean()

def parse_arguments():
    parser = argparse.ArgumentParser()

    # GIN parameters
    parser.add_argument('--dataset_name', type=str, default="TRIANGLES",
                        help='name of dataset')

    parser.add_argument('--baseline_mode', type=str, default=None,
                        help='baseline')

    parser.add_argument('--N_way', type=int, default=3)
    parser.add_argument('--K_shot', type=int, default=5)
    parser.add_argument('--query_size', type=int, default=10)
    parser.add_argument('--P_num', type=int, default=10)

    parser.add_argument('--test_task_num', type=int, default=10)

    parser.add_argument('--dropout', type=float, default=0.5)

    parser.add_argument('--gin_layer', type=int, default=5)
    parser.add_argument('--gin_hid', type=int, default=128)


    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.2)
    #parser.add_argument('--weight_decay', type=float, default=0)
    #parser.add_argument('--weight_decay', type=float, default=0)

    parser.add_argument('--eval_interval', type=int, default=25)
    parser.add_argument('--epoch_num', type=int, default=5000*5)

    parser.add_argument('--use_select_sim',type=bool,default=False)
    parser.add_argument('--use_loss_based_prob',type=bool,default=True)

    parser.add_argument('--save_test_emb',type=bool,default=True)

    args = parser.parse_args()
    return args

args = parse_arguments()

seed_value_start=31


datasets=['TRIANGLES']
result={dataset:defaultdict(list) for dataset in datasets}
for dataset in datasets:

    for k in [5]: #10
        args.K_shot=k

        for repeat in range(41,42):
            seed_value=seed_value_start+repeat
            import os
            os.environ['PYTHONHASHSEED']=str(seed_value)
            import random
            random.seed(seed_value)
            np.random.seed(seed_value)
            torch.manual_seed(seed_value)
            torch.cuda.manual_seed(seed_value)

            text_file_name='./our_results/{}-{}-shot.txt'.format(dataset, k)
            f=open(text_file_name, 'w')
            file_name='./our_results/save_test'

            print(file_name)
            args.dataset_name=dataset
            trainer=Trainer(args)

            test_acc=trainer.train()
            result[dataset][k].append(test_acc)

            del trainer
            del test_acc

            json.dump(result,open(file_name,'w'))

