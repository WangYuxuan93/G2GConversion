# coding:utf-8
# @Time     : 2021/10/5 10:27 AM
# @Author   : jeffrey

import numpy as np
from queue import Queue
import torch

class LCADAG(object):

    def __init__(self,G):
        # self.G = G.to(torch.device('cuda', 0))
        self.G = G
        self.batch,self.q_len,_ = G.shape

    def get_uG(self,g):
        undireted_G = g
        for i in range(0, self.q_len):
            for j in range(i + 1, self.q_len):
                if undireted_G[i][j] or undireted_G[j][i]:
                    undireted_G[i][j] = 1
                    undireted_G[j][i] = 1
                else:
                    undireted_G[i][j] = 0
                    undireted_G[j][i] = 0
        return undireted_G

    def get_next(self,g,i):
        """
        返回父节点

        :param i:
        :return:
        """
        next_value = torch.where(g[i]>0)
        return next_value[0]

    def get_child(self,g,i):
        """
        返回孩子节点
        :param i:
        :return:
        """
        child_value = torch.where(g[:,i]>0)
        return child_value[0]

    def get_ancestor(self):
        par = torch.zeros((self.batch,self.q_len,self.q_len,self.q_len),dtype=torch.long)
        for bt in range(self.batch):
            g = self.G[bt]
            for i in range(0,self.q_len):
                # if i==0:
                #     par[bt,i,:,0]=1
                par[bt,i,i,i] = 1  # (i,i)的最短公共节点是自身
                for j in range(i+1,self.q_len):
                    for k in self.get_nearest_par(g,i,j):
                        par[bt][i][j][k]=1
        return par


    def get_ancestor_by_bfs(self,g,i):
        res = []
        if i == 0:
            return [0]
        queue = Queue()
        nodeSet = set()
        queue.put(i)
        nodeSet.add(i)
        while not queue.empty():
            cur = queue.get() # 弹出元素
            res.append(cur)
            for next in self.get_next(g,cur):   #cur.nexts方法需要实现
                if next.item() not in nodeSet:
                    nodeSet.add(next.item())
                    queue.put(next.item())
        return res

    def get_nearest_par(self,g,i,j):
        res_i = self.get_ancestor_by_bfs(g,i)
        res_j = self.get_ancestor_by_bfs(g,j)
        candi = list(set(res_i).intersection(set(res_j)))
        fu = self.without_in_degree(g,candi)
        return fu

    def without_in_degree(self,g,par):
        """
        选择最近的公共父节点！
        :param par:
        :return:
        """
        del_node = set()
        for x in par:
            del_par = self.get_next(g,x)
            if len(del_par)>0:
                temp = set(list(del_par.numpy()))&(set(par))
                del_node = del_node.union(temp)
        return list(set(par)-del_node)

    def get_pattern(self,g,g2,uG3,uG4,uG5,uG6,i,j):  #i <- j
        if i==j:
            return 0
        par_reverse = self.get_next(g,j)
        if g[i,j]: # consistent
            return 1
        if g2[i,j]>0: # grand
            return 2
        if torch.any((g[i]==g[j])*g[i]>0): # sibling
            return 3
        if g[j,i]: # reverse consistent
            return 4
        if g2[j,i]>0: # reverse grand
            return 5
        # reverse sibling
        if torch.any((g[:,i]==g[:,j])*g[:,i]>0):
            return 6
        # 计算两个节点直接的距离

        if uG3[i][j]>0:
            return 7
        if uG4[i][j]>0 or uG5[i][j]>0:
            return 8
        if uG6[i][j]>0:
            return 9
        return 10

    def get_all_pattern(self):
        patterns = torch.zeros_like(self.G)
        for bt in range(self.batch):
            g = self.G[bt]
            g2 = torch.mm(g,g)
            ug = g | g.transpose(0, 1)
            uG3 = torch.mm(torch.mm(ug,ug),ug)
            uG4 = torch.mm(uG3,ug)
            uG5 = torch.mm(uG4,ug)
            uG6 = torch.mm(uG5,ug)
            for i in range(self.q_len):
                for j in range(self.q_len):
                    patterns[bt,i,j] = self.get_pattern(g,g2,uG3,uG4,uG5,uG6,i,j)
        return patterns


# G = np.array([[0,0,0,0,0,0,0,0,0],
#                   [1,0,0,0,0,0,0,0,0],
#                   [1,0,0,0,0,0,0,0,0],
#                   [0,0,1,0,0,0,0,0,0],
#                   [0,1,1,0,0,0,0,0,0],
#                   [0,0,0,0,0,0,1,0,0],
#                   [0,1,1,1,0,0,0,0,0],
#                   [0,0,0,0,0,0,1,0,0],
#                   [0,0,0,0,0,0,0,1,0]])
# lca = LCADAG(G)
# print(lca.get_ancestor())