# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


from typing import List
import queue


class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:

        q = queue.Queue()
        q.put([root, int(0)])
        list1 = []

        while not q.empty():
            root = q.get()
            print(root)
            if root[1] > len(list1):
                list1.append([])
            list1[root[1]].append(root[0].val)
            if root[0].left:
                q.put([root[0].left, root[1] + 1])
            if root[0].right:
                q.put([root[0].right, root[1] + 1])
        return list1

