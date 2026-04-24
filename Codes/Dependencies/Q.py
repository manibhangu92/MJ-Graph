# -*- coding: utf-8 -*-
"""
Created on Sat Jan  3 06:58:00 2026

@author: DR. MANPREET SINGH
"""

import heapq

class Q:
    def __init__(self, is_min=True):
        """
        is_min = True  -> Min Priority Queue
        is_min = False -> Max Priority Queue
        """
        self.heap = [];
        self.sign = 1 if is_min else -1;

    def push(self, key: int, value: int):
        """Insert a (key, value) pair."""
        heapq.heappush(self.heap, (self.sign * key, value));

    def pop(self):
        """Remove and return the highest-priority (key, value) pair."""
        if self.is_empty():
            raise IndexError("Priority queue is empty")
        key, value = heapq.heappop(self.heap);
        return self.sign * key, value;

    def peek(self):
        """Return the highest-priority (key, value) without removing it."""
        if self.is_empty():
            raise IndexError("Priority queue is empty")
        key, value = self.heap[0];
        return self.sign * key, value;

    def is_empty(self):
        return len(self.heap) == 0;

    def size(self):
        return len(self.heap);

