"""
Consistent hashing ring for URL-to-worker assignment
======================================================
URLs from the same domain always map to the same worker.
Uses virtual nodes for even distribution.

PDC concept: consistent hashing is used in distributed
systems to minimise data movement when nodes join/leave.
(Used by Amazon Dynamo, Apache Cassandra, CDNs)
"""

import hashlib
from urllib.parse import urlparse


class HashRing:
    def __init__(self, nodes, virtual_nodes=150):
        """
        nodes         : list of worker ranks (e.g. [1,2,3,4,5,6])
        virtual_nodes : number of virtual positions per real node
        """
        self.virtual_nodes = virtual_nodes
        self.ring          = {}   # position -> rank
        self.sorted_keys   = []

        for node in nodes:
            self._add_node(node)

    def _hash(self, key):
        """MD5 hash of key, returned as integer."""
        return int(hashlib.md5(key.encode()).hexdigest(), 16)

    def _add_node(self, node):
        for i in range(self.virtual_nodes):
            vkey = f"{node}-vnode-{i}"
            pos  = self._hash(vkey)
            self.ring[pos] = node
        self.sorted_keys = sorted(self.ring.keys())

    def get_node(self, key):
        """
        Return the worker rank responsible for this key.
        Finds the first virtual node clockwise from key's hash position.
        """
        if not self.ring:
            return None
        h = self._hash(key)
        for pos in self.sorted_keys:
            if h <= pos:
                return self.ring[pos]
        # wrap around
        return self.ring[self.sorted_keys[0]]

    def get_node_for_url(self, url):
        """Route by domain — all URLs from same domain go to same worker."""
        domain = urlparse(url).netloc
        return self.get_node(domain)