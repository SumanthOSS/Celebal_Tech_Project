#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Let's create a simple Node class to represent each element in our linked list
class Node:
    def __init__(self, data):
        self.data = data  # Store the data
        self.next = None  # Pointer to the next node

# Now, the LinkedList class to manage our nodes
class LinkedList:
    def __init__(self):
        self.head = None  # Start with an empty list

    # Add a node to the end of the list
    def append(self, data):
        new_node = Node(data)
        # If the list is empty, make this the head
        if not self.head:
            self.head = new_node
            return
        # Traverse to the last node
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node

    # Print the entire list
    def print_list(self):
        if not self.head:
            print("The list is empty!")
            return
        current = self.head
        elements = []
        while current:
            elements.append(str(current.data))
            current = current.next
        print(" -> ".join(elements))

    # Delete the nth node (1-based index)
    def delete_nth(self, n):
        try:
            # Check if the list is empty
            if not self.head:
                raise ValueError("Cannot delete from an empty list")

            # If n is less than 1, it's invalid
            if n < 1:
                raise ValueError("Index must be at least 1")

            # Special case: deleting the head
            if n == 1:
                self.head = self.head.next
                return

            # Traverse to find the node before the one to delete
            current = self.head
            count = 1
            while current and count < n - 1:
                current = current.next
                count += 1

            # Check if n is out of range
            if not current or not current.next:
                raise IndexError("Index out of range")

            # Skip the nth node
            current.next = current.next.next

        except ValueError as e:
            print(f"Error: {e}")
        except IndexError as e:
            print(f"Error: {e}")

# Let's test our linked list implementation
def test_linked_list():
    print("Creating a new linked list...")
    ll = LinkedList()

    # Test 1: Adding some elements
    print("\nAdding elements 10, 20, 30, 40...")
    ll.append(10)
    ll.append(20)
    ll.append(30)
    ll.append(40)
    print("Current list:")
    ll.print_list()

    # Test 2: Deleting the 2nd node
    print("\nDeleting the 2nd node...")
    ll.delete_nth(2)
    print("Current list:")
    ll.print_list()

    # Test 3: Try deleting from an empty list
    print("\nCreating an empty list and trying to delete...")
    empty_ll = LinkedList()
    empty_ll.delete_nth(1)

    # Test 4: Try deleting an out-of-range index
    print("\nTrying to delete the 10th node...")
    ll.delete_nth(10)

# Run the test
test_linked_list()


# In[ ]:




