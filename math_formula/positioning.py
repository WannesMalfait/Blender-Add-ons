import math
from bpy.types import Node, NodeLinks


class PositionNode():
    def __init__(self, node: Node, parent=None, children=None, left_sibling=None, right_sibling=None, depth: int = 0):
        self.node = node
        self.parent: PositionNode = parent
        self.children: list[PositionNode] = children
        self.first_child: PositionNode = children[0] if children else None
        self.left_sibling: PositionNode = left_sibling
        self.right_sibling: PositionNode = right_sibling
        self.width = self.node.dimensions.x
        self.height = self.node.dimensions.y
        self.prelim_y = 0
        self.modifier = 0
        self.depth = depth
        self.left_neighbour: PositionNode = None

    def set_children(self, children: list) -> None:
        self.children = children
        if children != []:
            self.first_child = children[0]

    def update_parent(self, new_parent) -> None:
        """Update relations so that the this node
        is not a child of the old parent, or a sibling of
        its old siblings. Does not take into account
        possible new siblings from the new parent."""
        prev_parent = self.parent
        if prev_parent is not None and prev_parent.children is not None:
            prev_parent.set_children([
                child for child in prev_parent.children if child is not self])
        self.parent = new_parent
        ls = self.left_sibling
        rs = self.right_sibling
        if ls:
            ls.right_sibling = rs
        if rs:
            rs.left_sibling = ls
        self.left_sibling = None
        self.right_sibling = None
        self.depth = new_parent.depth + 1

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

    def good_name(self, node) -> str:
        if node is None:
            return None
        if 'Math' in node.node.bl_idname:
            return node.node.operation
        elif node.node.label != "":
            return node.node.label
        else:
            return node.node.bl_idname

    def __str__(self) -> str:
        parent = self.good_name(self.parent)
        # first_child = self.good_name(self.first_child.node) if self.first_child else ""
        left = self.good_name(self.left_sibling)
        right = self.good_name(self.right_sibling)
        neighbour = self.good_name(self.left_neighbour)
        return f"{self.good_name(self)}:\n \
        parent: {parent}, \n \
        children: {self.children}, \n \
        left sibling: {left}, \n \
        right sibling: {right}, \n \
        left neighbour: {neighbour}"

    def __repr__(self) -> str:
        return f"{self.good_name(self)}"


class TreePositioner():
    """
    Class to position nodes in a node tree
    Algorithm: https://www.cs.unc.edu/techreports/89-034.pdf
    """

    def __init__(self, context):
        prefs = context.preferences.addons['math_formula'].preferences
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

    # Test formula:
    # coords.xyz; r = length({x,y,0}); theta = atan2(y,x); {r,theta,z}
    def build_relations(self, pnode: PositionNode, links: NodeLinks, depth: int = 0) -> None:
        # Get all links connected to the input sockets of the node
        input_links = []
        for link in links:
            # It's possible that nodes have multiple parents. In that case the
            # algorithm doesn't work, so we only allow one parent per node.
            if link.to_node == pnode.node:
                add_link = True
                child = None
                from_node = link.from_node
                for ilink, _ in input_links:
                    if ilink.from_node == from_node:
                        add_link = False
                        break
                if not add_link:
                    continue
                for vnode in self.visited_nodes:
                    if from_node == vnode.node:
                        if depth > vnode.depth:
                            vnode.update_parent(pnode)
                            add_link = True
                            child = vnode
                            break
                        add_link = False
                        break
                if add_link:
                    input_links.append((link, child))

        if input_links == []:
            # It's a leaf node
            return None

        # Sort the links in order of the sockets
        sorted_children: list[tuple[PositionNode, bool]] = []
        for socket in pnode.node.inputs:
            for link, node in input_links:
                if socket == link.to_socket:
                    if node is not None:
                        sorted_children.append((node, False))
                        continue
                    new_node = link.from_node
                    new_node.select = True
                    child = PositionNode(new_node, depth=depth+1)
                    self.visited_nodes.append(child)
                    sorted_children.append((child, True))

        # In the recursive sense, this is now the root node. The parent of this
        # node is set during backtracking.
        children_only = [child for child, _ in sorted_children]
        pnode.set_children(children_only)
        root_node = pnode
        for i, child in enumerate(children_only):
            if i < len(children_only)-1:
                child.right_sibling = children_only[i+1]
            if i > 0:
                child.left_sibling = children_only[i-1]
            child.parent = root_node
        for child, needs_building in sorted_children:
            if needs_building:
                self.build_relations(child, links, depth=depth+1)

    def place_nodes(self, root_node: PositionNode, links: NodeLinks, cursor_loc: tuple[float] = None) -> tuple[float]:
        """
        Aranges the nodes connected to `root_node` so that the top
        left corner lines up with `cursor_loc`. If `cursor_loc` is `None`,
        the tree is aligned such that `root_node` stays in the same place.

        The returned value is the bottom right corner, i.e the place where
        you would want to place the next nodes, if `cursor_loc` is not `None`.
        Otherwise `None` is returned.
        """
        root_node = PositionNode(root_node, depth=0)
        self.visited_nodes = [root_node]
        self.build_relations(root_node, links)
        self.visited_nodes = []
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
                    # Apply portions to appropriate left sibling
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
            # It's not a leaf, so recursively call for children
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
