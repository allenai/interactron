import torch


class Node:
    def __init__(self, cost=None, action=None):
        self.cost = cost
        self.action = action
        self.edges = []

    def get_edges(self):
        return {e.value: e.b for e in self.edges}

    def add_edge(self, e):
        if e not in [x.value for x in self.edges]:
            self.edges.append(e)


class Edge:
    def __init__(self, a, b, x):
        self.value = x
        self.a = a
        self.b = b


class PathStorage:

    def __init__(self):
        self.root = Node(float('inf'))

    def add_path(self, path, ifga):
        curr = self.root
        for a in path:
            a = a.item()
            if a is None:
                print("wow")
            if ifga < curr.cost:
                curr.cost = ifga
                curr.action = a
            if a not in curr.get_edges():
                curr.add_edge(Edge(curr, Node(float('inf')), a))
            curr = curr.get_edges()[a]

    def get_label(self, path):
        actions = []
        curr = self.root
        for a in path:
            a = a.item()
            actions.append(curr.action)
            curr = curr.get_edges()[a]
        return actions


def collate_fn(batch):
    collated_batch = {
        'frames': torch.stack([torch.stack(b['frames']) for b in batch]),
        "masks": torch.stack([torch.stack(b['masks']) for b in batch]),
        "actions": torch.stack([torch.tensor(b['actions'], dtype=torch.long) for b in batch]),
        "object_ids": [[torch.tensor(inst, dtype=torch.long) for inst in b['object_ids']] for b in batch],
        "category_ids": [[inst for inst in b['category_ids']] for b in batch],
        "boxes": [[inst for inst in b['boxes']] for b in batch],
        "episode_ids": torch.stack([torch.tensor(b['episode_ids'], dtype=torch.long) for b in batch]),
        "initial_image_path": [b['initial_image_path'] for b in batch],
    }
    return collated_batch
