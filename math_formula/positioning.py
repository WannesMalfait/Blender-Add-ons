import math
from bpy.types import Node


class PositionNode():
    def __init__(self, node: Node, parent=None, children=None, left_sibling=None, right_sibling=None, has_dimensions=False):
        self.node = node
        self.parent: PositionNode = parent
        self.children: list[PositionNode] = children
        self.first_child: PositionNode = children[0] if children else None
        self.left_sibling: PositionNode = left_sibling
        self.right_sibling: PositionNode = right_sibling
        # TODO: make update work so this is correct and doesn't require such a hack
        if has_dimensions:
            self.width = self.node.dimensions.x
            self.height = self.node.dimensions.y
        else:
            inputs = 0
            linked_sockets = 0
            for socket in node.inputs:
                if socket.enabled:
                    inputs += 1
                if socket.is_linked:
                    linked_sockets += 1
            if 'NodeMath' in node.bl_idname:
                self.height = 120
                self.height += inputs*22
            elif 'NodeVectorMath' in node.bl_idname:
                self.height = 96
                self.height += inputs*88
                self.height -= linked_sockets*66
            self.width = 153.611
        self.prelim_y = 0
        self.modifier = 0
        self.left_neighbour: PositionNode = None

    def set_x(self, x):
        self.node.location.x = x

    def set_y(self, y):
        self.node.location.y = y

    def get_x(self) -> int:
        return self.node.location.x

    def get_y(self) -> int:
        return self.node.location.y

    def get_width(self) -> float:
        return self.width

    def get_height(self) -> float:
        return self.height

    def is_leaf(self) -> bool:
        return self.first_child is None

    def has_right(self) -> bool:
        return self.right_sibling is not None

    def has_left(self) -> bool:
        return self.left_sibling is not None

    def __str__(self) -> str:
        parent = self.parent.node.operation if self.parent else ""
        # first_child = self.first_child.node.operation if self.first_child else ""
        left = self.left_sibling.node.operation if self.left_sibling else ""
        right = self.right_sibling.node.operation if self.right_sibling else ""
        neighbour = self.left_neighbour.node.operation if self.left_neighbour else ""
        return f"{self.node.operation}:\n \
        parent: {parent}, \n \
        children: {self.children}, \n \
        left sibling: {left}, \n \
        right sibling: {right}, \n \
        left neighbour: {neighbour}"

    def __repr__(self) -> str:
        return f"{self.node.operation}"


class TreePositioner():
    """
    Class to position nodes in a node tree
    Algorithm: https://www.cs.unc.edu/techreports/89-034.pdf
    """

    def __init__(self, context):
        prefs = context.preferences.addons[__name__].preferences
        self.level_separation: int = prefs.node_distance
        self.sibling_separation: int = prefs.sibling_distance
        self.subtree_separation: int = prefs.subtree_distance
        self.x_top_adjustment: int = 0
        self.y_top_adjustment: int = 0
        self.max_width_per_level: list[float] = [0 for _ in range(100)]
        self.prev_node_per_level = [None for _ in range(100)]
        self.min_x_loc = +math.inf
        self.max_x_loc = -math.inf
        self.min_y_loc = +math.inf
        self.max_y_loc = -math.inf
        self.visited_nodes: list[PositionNode] = []

    def place_nodes(self, root_node: PositionNode, cursor_loc: tuple[float] = None) -> tuple[float]:
        """
        Aranges the nodes connected to `root_node` so that the top
        left corner lines up with `cursor_loc`. If `cursor_loc` is `None`,
        the tree is aligned such that `root_node` stays in the same place.

        The returned value is the top right corner, i.e the place where
        you would want to place the next nodes, if `cursor_loc` is not `None`.
        Otherwise `None` is returned.
        """
        old_root_node_pos_x, old_root_node_pos_y = root_node.node.location
        self.first_walk(root_node, 0)
        self.x_top_adjustment = root_node.get_x()
        self.y_top_adjustment = root_node.get_y() - root_node.prelim_y
        self.second_walk(root_node, 0, 0, 0)
        offset_x = 0
        offset_y = 0
        if cursor_loc is not None:
            offset_x = cursor_loc[0] - self.min_x_loc
            offset_y = cursor_loc[1] - self.max_y_loc
        else:
            offset_x = old_root_node_pos_x-root_node.get_x()
            offset_y = old_root_node_pos_y-root_node.get_y()
        for pnode in self.visited_nodes:
            pnode.set_x(pnode.get_x() + offset_x)
            pnode.set_y(pnode.get_y() + offset_y)
        if cursor_loc is not None:
            return (cursor_loc[0]+self.max_x_loc-self.min_x_loc, cursor_loc[1])

    def get_leftmost(self, node: PositionNode, level: int, depth: int) -> PositionNode:
        if level >= depth:
            return node
        if node.is_leaf():
            return None
        rightmost = node.first_child
        leftmost = self.get_leftmost(rightmost, level + 1, depth)
        while leftmost is None and rightmost.has_right():
            rightmost = rightmost.right_sibling
            leftmost = self.get_leftmost(rightmost, level + 1, depth)
        return leftmost

    def get_prev_node_at_level(self, level: int) -> PositionNode:
        return self.prev_node_per_level[level]

    def set_prev_node_at_level(self, level: int, node: PositionNode):
        self.prev_node_per_level[level] = node

    def apportion(self, node: PositionNode):
        leftmost = node.first_child
        neighbour = leftmost.left_neighbour
        compare_depth = 1
        while leftmost is not None and neighbour is not None:
            # Compute the location of leftmost and where it
            # should be with respect to neighbour
            left_mod_sum = right_mod_sum = 0
            ancestor_leftmost = leftmost
            ancestor_neighbour = neighbour

            for _ in range(compare_depth):
                ancestor_leftmost = ancestor_leftmost.parent
                ancestor_neighbour = ancestor_neighbour.parent
                right_mod_sum += ancestor_leftmost.modifier

                left_mod_sum += ancestor_neighbour.modifier

            # Find the move_distance and apply it to the node's subtree
            # Add appropriate portions to smaller interior subtrees
            move_distance = neighbour.prelim_y +\
                left_mod_sum + \
                self.subtree_separation + \
                neighbour.get_height() - \
                (leftmost.prelim_y + right_mod_sum)
            if move_distance > 0:
                tmp = node
                left_siblings = 0
                # Count the interior sibling subtrees
                while tmp is not None and tmp != ancestor_neighbour:
                    left_siblings += 1
                    tmp = tmp.left_sibling
                if tmp is not None:
                    # Apply posrtions to appropriate left sibling
                    # subtrees
                    portion = move_distance/left_siblings
                    tmp = node
                    while tmp != ancestor_neighbour:
                        tmp.prelim_y += move_distance
                        tmp.modifier += move_distance
                        move_distance -= portion
                        tmp = tmp.left_sibling
                else:
                    # In this case ancestor_neighbour and ancestor_leftmost
                    # aren't siblings, so the job to move should be done by
                    # an ancestor instead
                    return

            # Determine the leftmost descendant of Node at the next lower level
            # to compare its positioning against that of its neighbour.
            compare_depth += 1
            if leftmost.is_leaf():
                leftmost = self.get_leftmost(node, 0, compare_depth)
            else:
                leftmost = leftmost.first_child
            if leftmost is not None:
                neighbour = leftmost.left_neighbour
            else:
                return

    def first_walk(self, node: PositionNode, level: int):
        node.left_neighbour = self.get_prev_node_at_level(level)
        self.set_prev_node_at_level(level, node)
        node.modifier = 0
        if node.is_leaf():
            if node.has_left():
                node.prelim_y = node.left_sibling.prelim_y + \
                    self.sibling_separation + \
                    node.left_sibling.get_height()
            else:
                node.prelim_y = 0
        else:
            # It's not a leaf, so recursivly call for children
            leftmost = rightmost = node.first_child
            self.first_walk(leftmost, level+1)
            while rightmost.has_right():
                rightmost = rightmost.right_sibling
                self.first_walk(rightmost, level+1)
            mid = (leftmost.prelim_y + rightmost.prelim_y)/2
            if node.has_left():
                node.prelim_y = node.left_sibling.prelim_y + \
                    self.sibling_separation + \
                    node.left_sibling.get_height()
                node.modifier = node.prelim_y - mid
                self.apportion(node)
            else:
                node.prelim_y = mid
        self.max_width_per_level[level] = max(
            node.width, self.max_width_per_level[level])

    def second_walk(self, node: PositionNode, level: int, width_sum_x: float, mod_sum_y: float):
        x = self.x_top_adjustment - width_sum_x
        y = self.y_top_adjustment - node.prelim_y - mod_sum_y
        self.min_x_loc = min(x, self.min_x_loc)
        self.min_y_loc = min(y+node.get_height(), self.min_y_loc)
        self.max_x_loc = max(x+node.get_width(), self.max_x_loc)
        self.max_y_loc = max(y, self.max_y_loc)
        node.set_x(x)
        node.set_y(y)
        self.visited_nodes.append(node)
        if not node.is_leaf():
            self.second_walk(node.first_child, level + 1,
                             width_sum_x +
                             self.max_width_per_level[level+1] +
                             self.level_separation,
                             mod_sum_y + node.modifier)
        if node.has_right():
            self.second_walk(node.right_sibling, level, width_sum_x, mod_sum_y)
