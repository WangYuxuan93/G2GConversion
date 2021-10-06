# coding:utf-8
# @Time     : 2021/10/5 10:27 AM
# @Author   : jeffrey

import numpy as np
from queue import Queue


class LCADAG(object):

    def __init__(self,G):
        self.G = G
        self.q_len = len(G)
        self.undirected_G = self.get_uG()
        self.uG3 = np.dot(self.undireted_G,np.dot(self.undireted_G,self.undireted_G))
        self.uG4 = np.dot(self.undireted_G,self.uG3)
        self.uG5 = np.dot(self.undirected_G,self.uG4)
        self.uG6 = np.dot(self.undirected_G,self.uG5)

    def get_uG(self):
        undireted_G = self.G
        for i in range(0, self.q_len):
            for j in range(i + 1, self.q_len):
                if undireted_G[i, j] or undireted_G[j, i]:
                    undireted_G[i, j] = 1
                    undireted_G[j, i] = 1
                else:
                    undireted_G[i, j] = 0
                    undireted_G[j, i] = 0
        return undireted_G

    def get_next(self,i):
        """
        返回父节点

        :param i:
        :return:
        """
        next_value = np.nonzero(self.G[i])
        return list(next_value[0])

    def get_child(self,i):
        """
        返回孩子节点
        :param i:
        :return:
        """
        child_value = np.nonzero(self.G[:,i])
        return list(child_value[0])

    def get_ancestor(self):
        par = np.zeros((self.q_len,self.q_len,self.q_len),dtype=int)
        for i in range(0,self.q_len):
            if i==0:
                par[i,:,0]=1
            for j in range(i+1,self.q_len):
                for k in self.get_nearest_par(i,j):
                    par[i][j][k]=1
        return par


    def get_ancestor_by_bfs(self,i):
        res = []
        if i == 0:
            return res
        queue = Queue()
        nodeSet = set()
        queue.put(i)
        nodeSet.add(i)
        while not queue.empty():
            cur = queue.get() # 弹出元素

            res.append(cur)
            for next in self.get_next(cur):   #cur.nexts方法需要实现
                if next not in nodeSet:
                    nodeSet.add(next)
                    queue.put(next)
        return res

    def get_nearest_par(self,i,j):
        res_i = self.get_ancestor_by_bfs(i)
        res_j = self.get_ancestor_by_bfs(j)
        candi = list(set(res_i).intersection(set(res_j)))
        return self.without_in_degree(candi)

    def without_in_degree(self,par):
        """
        选择最近的公共父节点！
        :param par:
        :return:
        """
        del_node = set()
        for x in par:
            del_par = self.get_next(x)
            if len(del_par)>0:
                temp = set(del_par)&(set(par))
                del_node = del_node.union(temp)
        return list(set(par)-del_node)

    def get_pattern(self,i,j):  #i <- j
        par = self.get_next(i)
        par_reverse = self.get_next(j)
        if j in par: # consistent
            return 1
        for k in par: # grand
            grand_par = self.get_next(k)
            if j in grand_par:
                return 2
        if len(list(set(par) & set(par_reverse)))>0: # sibling
            return 3
        if i in par_reverse: # reverse
            return 4
        for k in par_reverse: # reverse grand
            reverse_grand_par = self.get_next(k)
            if i in reverse_grand_par:
                return 5
        # reverse sibling
        child_i = self.get_child(i)
        child_j = self.get_child(j)
        if len(list(set(child_i)&set(child_j)))>0:
            return 6
        # 计算两个节点直接的距离
        if self.uG3[i][j]>0:
            return 7
        if self.uG4[i][j]>0 or self.uG5[i][j]>0:
            return 8
        if self.uG6[i][j]>0:
            return 9
        return 10

    def get_all_pattern(self):
        patterns = np.zeros_like(self.G)
        for i in range(self.q_len):
            for j in range(self.q_len):
                patterns[i,j] = self.get_pattern(i,j)
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