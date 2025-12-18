import pandas as pd
import numpy as np
import nltk,os,re
from collections import defaultdict,Counter

class Node():
    def __init__(self,data):
        self.data = data
        self.next = None

    def get_data(self):
        return self.data

    def get_next(self):
        return self.next

    def set_data(self,data):
        self.data = data

    def set_next(self,next):
        self.next = next


class LinkedList():
    def __init__(self):
        self.head = None

    def is_empty(self):
        return self.head == None

    def add_node(self,data):
        new_node = Node(data)
        new_node.set_next(self.head)
        self.head = new_node

    def get_data(self,data):
        current = self.head
        found = False
        result = None
        while current != None and not found:
            if current.get_data() == data:
                result = current.get_data()
                found = True
            else:
                current = current.get_next()
        return result


class Stack():
    def __init__(self):
        self.items = []

    def push(self,data):
        self.items.append(data)

    def is_empty(self):
        return self.items == []

    def pop(self):
        return self.items.pop()

    def peek(self):
        return self.items[-1]

    def size(self):
        return len(self.items)

    def clear(self):
        self.items = []

class Queue():
    def __init__(self):
        self.items = []

    def is_empty(self):
        return self.items == []

    def push(self,data):
        self.items.append(data)

    def pop(self):
        return self.items.pop(0)

    def peek(self):
        return self.items[-1]

    def size(self):
        return len(self.items)

    def return_queue(self):
        return self.items


