import math
import random
import typing
from typing import Optional


class BTree:
    class BNode:
        def __init__(self, keys: list, data: list, *children):
            self.keys = keys
            self.values = data
            self.children: list[typing.Self] = list(children)

        @property
        def is_leaf(self):
            return len(self.children) == 0

        @property
        def keys_count(self):
            return len(self.keys)

        def __str__(self):
            keys = f'|{"|".join(map(str, self.keys))}|'
            return f'BNode {keys}'

    def __init__(self, root: BNode, order: int):
        #  math.ceil the t?
        self.order = order  # released for odd orders
        self.root = root
        self.t = math.ceil(order / 2)

    def traverse_subtree(self, node):
        print(node)
        for child in node.children:
            self.traverse_subtree(child)

    def get_value(self, key):
        node = self.search(key)
        if not node:
            return
        i = node.keys.index(key)
        return node.values[i]

    def node_is_full(self, node: BNode):
        return node.keys_count == self.order

    def search(self, key) -> Optional[BNode]:
        return self._search(key, self.root)

    def _search(self, key, node: Optional[BNode]) -> typing.Any | None:
        if node is None:
            return None
        i = 0
        # TODO to replace with binary search
        while i < node.keys_count and node.keys[i] < key:
            i += 1
        if i < node.keys_count and node.keys[i] == key:
            return node.values[i]
        if node.is_leaf:
            return None
        return self._search(key, node.children[i])

    # node insertion
    def insert(self, key, data) -> bool:  # inserted? or updated.
        if self.root is None:
            self.root = self.BNode([key], [data])
            return True
        root = self.root
        if self.node_is_full(self.root):
            tmp_node = self.BNode([], [])
            self.root = tmp_node
            tmp_node.children.insert(0, root)
            self._split_child(tmp_node, 0)
            return self._insert_not_full(tmp_node, key, data)
        else:
            return self._insert_not_full(root, key, data)

    def _insert_not_full(self, node: BNode, key, data):
        i = 0
        while i < node.keys_count and node.keys[i] < key:
            i += 1
        if i < node.keys_count and node.keys[i] == key:
            node.values[i] = data
            return False
        if node.is_leaf:
            node.values.insert(i, data)
            node.keys.insert(i, key)
            return True
        if self.node_is_full(node.children[i]):
            self._split_child(node, i)
            if key > node.keys[i]:
                i += 1
        return self._insert_not_full(node.children[i], key, data)

    def _split_child(self, node: BNode, i: int):  # full node
        split_point = self.order // 2
        ch1 = node.children[i]
        new_node = self.BNode(
            ch1.keys[split_point + 1:],
            ch1.values[split_point + 1:],
        )
        node.children.insert(i + 1, new_node)
        node.keys.insert(i, ch1.keys[split_point])
        node.values.insert(i, ch1.values[split_point])
        ch1.keys = ch1.keys[:split_point]
        ch1.values = ch1.values[:split_point]
        if not ch1.is_leaf:
            new_node.children = ch1.children[split_point + 1:]
            ch1.children = ch1.children[:split_point + 1]

    # key delteion
    def delete(self, key, node: BNode) -> typing.Any | None:
        t = self.order // 2 + 1
        i = 0
        while i < node.keys_count and key > node.keys[i]:
            i += 1
        if node.is_leaf:
            if i < node.keys_count and node.keys[i] == key:
                node.keys.remove(i)
                return node.values.pop(i)
            return
        if i < node.keys_count and node.keys[i] == key:
            return self.delete_internal(node, key, i)

    def delete_internal(self, node: BNode, key, i):
        t = math.ceil(self.order / 2)
        if node.children[i].keys_count >= t:
            node.keys[i], node.values[i] =\
                self.delete_predecessor(node.children[i])

    def delete_predecessor(self, node: BNode):
        if node.is_leaf:
            return ...
        n = node.keys_count - 1
        if node.children[n].keys >= self.t:
            self.delete_sibling(node, n + 1, n)
        else:
            self.delete_merge(node, n, n + 1)
        self.delete_predecessor(node.children[n])

    @staticmethod
    def delete_sibling(node: BNode, i: int, j: int):
        ch_node = node.children[i]
        if i < j:
            right_side_node = node.children[j]
            ch_node.keys.append(node.keys[i])
            ch_node.values.append(node.values[i])
            node.keys[i] = right_side_node.keys.pop(0)
            node.values[i] = right_side_node.values.pop(0)
            if not right_side_node.is_leaf:
                ch_node.children.append(right_side_node.children.pop(0))
        else:
            left_side_node = node.children[j]
            ch_node.keys.insert(0, node.keys[i - 1])
            ch_node.values.insert(0, node.values[i - 1])
            node.keys[i - 1] = left_side_node.keys.pop()
            node.values[i - 1] = left_side_node.values.pop()
            if not left_side_node.is_leaf:
                ch_node.children.insert(0, left_side_node.children.pop())

    def delete_merge(self, node: BNode, i, j):
        ch_node = node.children[i]
        if j > i:
            right_side_node = node.children[j]
            potential_root = self._merge_delete_r_l(
                node, ch_node, right_side_node, i, j)
            node.children.remove(j)
        else:
            left_side_node = node.children[j]
            potential_root = self._merge_delete_r_l(
                node, left_side_node, ch_node, j, i)
        if self.root == node and node.is_leaf:
            self.root = potential_root

    @staticmethod
    def _merge_delete_r_l(parent: BNode, left: BNode, right: BNode, l_i: int, r_i: int):
        left.keys.append(parent.keys.pop(l_i))
        left.values.append(parent.values.pop(l_i))
        for i, j in enumerate(right.keys):
            left.keys.append(j)
            left.values.append(right.values[i])
            if not left.is_leaf:
                left.children.append(right.children[i])
        if not left.is_leaf:
            left.children.append(right.children.pop())
        parent.children.pop(r_i)
        return left


def main():
    b_tree = BTree(BTree.BNode([47, 92], [11, 456]), 5)
    for i in range(45):
    #     b_tree.insert(i, 2 * i)
    # b_tree.traverse_subtree(b_tree.root)
        # print(b_tree.root.children)
        key = random.randint(0, 255)
        data = random.randint(-65536, 65536)
        print(key, data)
        b_tree.insert(key, data)
        if i % 3 == 1:
            print('-' * 45)
            b_tree.traverse_subtree(b_tree.root)
            print('-' * 45)
        if i == 2:
            print(b_tree.search(46))
        if i == 5:
            print('SEARCH EXISTING:')
            print(b_tree.search(47))
            print()
        if i == 11:
            print(b_tree.search(228))
        if i == 20:
            b_tree.insert(92, 133)
    return


if __name__ == '__main__':
    main()
