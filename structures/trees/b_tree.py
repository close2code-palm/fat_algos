import collections
import math
import random
import typing
from contextlib import suppress
from typing import Optional

K = typing.TypeVar('K', bound=str | bytes | int | float)
V = typing.TypeVar('V', bound=typing.Any)

# TODO fully test


# Some amount of code is coming from https://github.com/jhomswk/B_Tree/blob/master/b_tree.py,
# debuged and tested by myself. You can get implementation from there, as it is not included



class BTreeMap(collections.abc.MutableMapping, typing.Generic[K, V]):
    """
    Unique keys with data structure
    optimization of disk space storages
    # >>> BTreeMap(BTreeMap.BNode([2][3]), 2)
    """

    class BNode:
        def __init__(self, keys: list[K], data: list[V], *children):
            self.keys = keys
            self.values = data
            self.children: list[typing.Self] = list(children)

        @property
        def is_leaf(self) -> bool:
            """
            >>> b_node = BTreeMap.BNode([2],[3])
            >>> b_node.is_leaf
            True
            >>> a_node = BTreeMap.BNode([4],[5])
            >>> c_node = BTreeMap.BNode([3],[2], b_node, a_node)
            >>> c_node.is_leaf
            False

            :return: if the node has no children
            """
            return len(self.children) == 0

        @property
        def keys_count(self) -> int:
            """
            >>> b_node = BTreeMap.BNode([2],[3])
            >>> b_node.keys_count
            1
            >>> dummy_node = BTreeMap.BNode([],[])
            >>> dummy_node.keys_count
            0
            """
            return len(self.keys)

        def b_search(self, key: K) -> int:
            """
            >>> node = BTreeMap.BNode([3,4], [24,56], [7, 13])
            >>> node.b_search(3)
            0
            >>> node.b_search(7)
            2

            :param key:
            :return: index
            """
            left = 0
            right = self.keys_count
            while right > left:
                mid = (left + right) // 2
                if self.keys[mid] >= key:
                    right = mid
                else:
                    left = mid + 1
            return left

        def get_value(self, key: K) -> Optional[V]:
            """
            >>> node = BTreeMap.BNode([3,4], [24,56])
            >>> node.get_value(4)
            56
            >>> node.get_value(5)

            :param key:
            :return:
            """
            with suppress(ValueError):
                index = self.keys.index(key)
                return self.values[index]

        def grow_child(self, index: int, min_num_keys: int) -> typing.Self:
            """
            Returns self's index-th child after increasing its number of
            keys by either:

                - Transferring a key from a direct sibling that contains
                  more than min_num_keys keys or,

                - Merging with a sibling that contains at most min_num_keys
                keys.
            """
            child = self.children[index]
            left_sibling = index > 0 and self.children[index - 1]
            right_sibling = index < self.keys_count and self.children[index + 1]

            if left_sibling and left_sibling.keys_count > min_num_keys:
                self.transfer_key_clockwise(index - 1)

            elif right_sibling and right_sibling.keys_count > min_num_keys:
                self.transfer_key_counter_clockwise(index)

            else:
                shared_key_index = (index - 1) if left_sibling else index
                child = self.merge_children(shared_key_index)

            return child

        def merge_children(self, index: int) -> typing.Self:
            """
            Merges self's index-th key and its left
            and right children into a single node.
            """
            median_key = self.keys[index]
            left, right = self.children[index: index + 2]

            left.keys.append(median_key)
            left.values.append(self.values[index])
            left.keys.extend(right.keys)
            left.values.extend(right.values)

            if not right.is_leaf:
                left.children.extend(right.children)

            del self.keys[index]
            del self.children[index + 1]

            merged = left

            if self.keys_count == 0:
                self.keys = left.keys
                self.values = left.values
                self.children = left.children
                merged = self

            return merged

        def contains_key_at(self, key: K, index: int) -> bool:
            """
            Checks whether index is the index of key in self.
            """
            return index < self.keys_count and self.keys[index] == key

        def linear_serach(self, key: K) -> int:
            index = 0
            while index < self.keys_count and self.keys[index] < key:
                index += 1
            return index

        def __str__(self) -> str:
            return f'|{"|".join(map(str, self.keys))}|'

        def subs_max(self) -> tuple[K, V] | None:
            """
            Returns the largest key in self's subtree.
            """
            node = self
            while not node.is_leaf:
                node = node.children[-1]
            return (node.keys[-1], node.values[-1]) if node.keys else None

        def subs_min(self) -> tuple[K, V] | None:
            """
            Returns the smallest key in self's subtree.
            """
            node = self
            while not node.is_leaf:
                node = node.children[0]
            if node.keys:
                return node.keys[0], node.values[0]

        def predecessor(self, index: int) -> typing.Self:
            """
            Returns the key, in self's subtree, that
            precedes the index-th key in self.

            Note: Assumes that self is not a leaf.
            """
            return self.children[index].subs_max()

        def successor(self, index) -> typing.Self:
            """
            Returns the key, in self's subtree, that
            succeeds the index-th key in self.

            Note: Assumes that self is not a leaf.
            """
            return self.children[index + 1].subs_min()

        def delete(self, key) -> None:
            """
            >>> node = BTreeMap.BNode([2,3],[4,7])
            >>> node.delete(2)
            >>> node.contains_key_at(2, node.b_search(2))
            False

            Deletes key from self.
            """
            index = self.b_search(key)
            if self.contains_key_at(key, index):
                del self.keys[index]
                del self.values[index]

        def transfer_key_clockwise(self, index: int) -> None:
            """
            Let child be self's index-th child and let sibling be child's left sibling.
            This method transfers the largest key of sibling to self, replacing its
            index-th key. Then the replaced key and the rightmost child of sibling are
            transferred to child.
            """
            left, right = self.children[index: index + 2]
            right.keys.insert(0, self.keys[index])
            right.values.insert(0, self.values[index])

            if left.children:
                right.children.insert(0, left.children.pop())

            self.keys[index] = left.keys.pop()
            self.values[index] = left.values.pop()

        def transfer_key_counter_clockwise(self, index: int):
            """
            Let child be self's index-th child and let sibling be the right sibling of
            child. This method transfers the smallest key of sibling to self, replacing
            its index-th key. Then the replaced key and the leftmost child of sibling
            are transferred to child.
            """
            left, right = self.children[index: index + 2]
            left.keys.append(self.keys[index])
            left.values.append(self.values[index])

            if not right.is_leaf:
                left.children.append(right.children.pop(0))

            self.keys[index] = right.keys.pop(0)
            self.values[index] = right.values.pop(0)

    def __init__(self, root: BNode, order: int):
        #  math.ceil the t?
        self.order = order  # released for odd orders only!
        self.root = root
        self.t = math.ceil(order / 2)
        self.min_keys = self.t - 1
        self._iter_keys: list[K] = []
        self._iter_ptr = 0

    def __getitem__(self, item: K) -> Optional[BNode]:
        """
        >>> btm = BTreeMap(BTreeMap.BNode([2],[3]), 5)
        >>> btm[2]
        3
        >>> btm[4]

        :param item:
        :return:
        """

        return self.get_value(item)

    def __contains__(self, item: K) -> bool:
        """
        >>> btm = BTreeMap(BTreeMap.BNode([2],[3]), 7)
        >>> 2 in btm
        True
        >>> 3 in btm
        False
        """
        return item in self.list_keys()

    def __len__(self) -> int:
        return len(list(self.list_keys()))

    def __setitem__(self, key: K, value: V) -> None:
        """
        >>> btm = BTreeMap(BTreeMap.BNode([2,5],[5,6]), 7)
        >>> btm[4] = 23
        >>> btm[4]
        23
        >>> btm.list_keys()
        [2, 4, 5]

        :param key:
        :param value:
        :return:
        """
        self.insert(key, value)

    def __delitem__(self, key: K) -> None:
        self.delete(key)

    def __iter__(self):
        self._iter_keys = self.list_keys()
        return self

    def __next__(self):
        try:
            self._iter_ptr += 1
            return self._iter_keys[self._iter_ptr - 1]
        except IndexError as e:
            raise StopIteration from e

    def list_keys(self) -> list[K]:
        # return list(self._get_keys(self.root))
        return list(self.ordered_iter(self.root))

    def ordered_iter(self, node):
        if node.is_leaf:
            for k in node.keys:
                yield k
            return
        for i, child in enumerate(node.children[:-1]):
            yield from self.ordered_iter(child)
            yield node.keys[i]
        yield from self.ordered_iter(node.children[-1])

    def _get_keys(self, node: BNode) -> typing.Iterator[K]:
        for k in node.keys:
            yield k
        for child in node.children:
            yield from self._get_keys(child)

    def tree_map(self) -> None:  # Saw much better implementations
        tree_state = collections.defaultdict(list)
        self._build_map(0, tree_state, self.root)
        tree_image = {}
        for k in tree_state:
            tree_image[k] = '   '.join(tree_state[k])
        n = len(tree_image[k])
        print('-' * n)
        for k in tree_image:
            print(tree_image[k].center(n))
        print('-' * n)

    def _build_map(self, level, tree_state, node: BNode):
        tree_state[level].append(str(node))
        for child in node.children:
            self._build_map(level + 1, tree_state, child)

    def get_value(self, key: K):
        node = self.search(key)
        if not node:
            return
        return node.get_value(key)

    def node_is_full(self, node: BNode) -> bool:
        return node.keys_count == self.order

    def search(self, key) -> Optional[BNode]:
        return self._search(key, self.root)

    def _search(self, key, node: Optional[BNode]) -> typing.Any | None:
        if node is None:
            return None
        index = node.b_search(key)
        if node.contains_key_at(key, index):
            return node
        if node.is_leaf:
            return None
        return self._search(key, node.children[index])

    # node insertion
    def insert(self, key: K, data: V) -> bool:  # inserted? or updated.
        if self.root is None:
            self.root = self.BNode([key], [data])
            return True
        key_node = self.search(key)
        if key_node:
            key_pos = key_node.b_search(key)
            key_node.values[key_pos] = data
            return False
        root = self.root
        if self.node_is_full(self.root):
            tmp_node = self.BNode([], [])
            self.root = tmp_node
            tmp_node.children.insert(0, root)
            self._split_child(tmp_node, 0)
            return self._insert_not_full(tmp_node, key, data)
        else:
            return self._insert_not_full(root, key, data)

    def _insert_not_full(self, node: BNode, key: K, data: V):
        i = node.b_search(key)
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
        ch1 = node.children[i]
        new_node = self.BNode(
            ch1.keys[self.t:],
            ch1.values[self.t:],
        )
        node.children.insert(i + 1, new_node)
        node.keys.insert(i, ch1.keys[self.t - 1])
        node.values.insert(i, ch1.values[self.t - 1])
        ch1.keys = ch1.keys[:self.t - 1]
        ch1.values = ch1.values[:self.t - 1]
        if not ch1.is_leaf:
            new_node.children = ch1.children[self.t:]
            ch1.children = ch1.children[:self.t]

    def delete(self, key: K) -> None:
        """
        Deletes key from the b-tree.
        """
        node = self.root
        while not node.is_leaf:
            index = node.b_search(key)
            if node.contains_key_at(key, index):
                left, right = node.children[index: index + 2]
                if left.keys_count > self.min_keys:
                    node.keys[index], node.values[index] = node.predecessor(index)
                    node, key = left, node.keys[index]
                elif right.keys_count > self.min_keys:
                    node.keys[index], node.values[index] = node.successor(index)
                    node, key = right, node.keys[index]
                else:
                    node.merge_children(index)
            else:
                child = node.children[index]
                if child.keys_count <= self.min_keys:
                    child = node.grow_child(index, self.min_keys)
                node = child
        node.delete(key)


def fuzz():
    for j in range(100):
        b_tree = BTreeMap(BTreeMap.BNode([47, 92], [11, 456]), 5)
        for i in range(1200):
            #     b_tree.insert(i, 2 * i)
            # b_tree.traverse_subtree(b_tree.root)
            # print(b_tree.root.children)
            key = random.randint(0, 1023)
            data = random.randint(-65536, 65535)
            print(key, data)
            b_tree.insert(key, data)
            # if i == 2:
            #     print(b_tree.search(46))
            # if i == 5:
            #     print('SEARCH EXISTING:')
            #     print(b_tree.search(47))
            #     print()
            # if i == 11:
            #     print(b_tree.search(228))
            # if i == 20:
            #     b_tree.insert(92, 133)
        # b_tree.delete(92)
        # b_tree.delete(428)
        # b_tree.delete(47)
        for i in range(700):
            b_tree.tree_map()
            keys = list(b_tree.list_keys())
            ktd = random.choice(keys)
            print(ktd)
            b_tree.delete(ktd)
        # b_tree.tree_map()
        # t_d = random.randint(0, 255)
        # print(t_d)
        # b_tree.delete(t_d)
    b_tree.tree_map()
    # b_tree.traverse_subtree(b_tree.root)
    return


def deletion_bug():
    l_ch = BTreeMap.BNode([0, 1, 2], [123, 1231, 6543])
    r_ch = BTreeMap.BNode([4, 5], [124123, 1231241])
    b_tree = BTreeMap(BTreeMap.BNode([3], [3141], l_ch, r_ch), 5)
    # keys = iter(b_tree)
    for ks in b_tree:
        print(ks, ' ', b_tree[ks])
    # for i in range(2):
    #     print(next(keys))
    # for node in b_tree:
    #     print(node)
    # b_tree.tree_map()
    # b_tree.delete(6)
    # b_tree.tree_map()
    # b_tree.delete(6)
    # b_tree.tree_map()


def fuzz1():
    for _ in range(100):
        b_tree = BTreeMap(BTreeMap.BNode([47, 92], [11, 456]), 15)

        for i in range(1200):
            key = random.randint(0, 1023)
            data = random.randint(-65536, 65535)
            # print(key, data)
            b_tree.insert(key, data)
            if i % 2 == 0 or i % 3 == 0:
                # b_tree.tree_map()
                keys = list(b_tree.list_keys())
                ktd = random.choice(keys)
                # print(ktd)
                b_tree.delete(ktd)
        b_tree.tree_map()
        assert len(list(b_tree.list_keys())) == len(set(b_tree.list_keys()))

    # for i in range(7000):


if __name__ == '__main__':
    # fuzz()
    doctest.testmode()  # type: ignore
    # BTreeMap.BNode
    # deletion_bug()
    # print('ok')
