from sys import maxsize as INF
from typing import cast, Optional
from bpy.types import Node, NodeLinks
from mathutils import Vector


class DummyVec2():
    def __init__(self) -> None:
        self.x = 0
        self.y = 0


class DummyNode():
    def __init__(self) -> None:
        self.dimensions = DummyVec2()
        self.location = DummyVec2()
        self.bl_idname = "DummyNode"
        self.bl_width_default = self.bl_height_default = 0


class PositionNode():
    def __init__(self,
                 node: Node | DummyNode,
                 parent: Optional['PositionNode'] = None,
                 children: Optional[list['PositionNode']] = None,
                 left_sibling: Optional['PositionNode'] = None,
                 right_sibling: Optional['PositionNode'] = None,
                 depth: int = 0):
        self.node = node
        self.parent = parent
        self.children = children
        self.first_child = children[0] if children else None
        self.left_sibling = left_sibling
        self.right_sibling = right_sibling
        dimensions = cast(Vector, self.node.dimensions)
        self.width = int(dimensions.x)
        if self.width == 0:
            # Not always great, but better than nothing
            self.width = int(self.node.bl_width_default) + 10
        self.height = int(dimensions.y)
        if self.height == 0:
            self.height = int(self.node.bl_height_default) + 10
        self.prelim_y = 0
        self.modifier = 0
        self.depth = depth
        self.left_neighbour: 'PositionNode' | None = None

    def set_children(self, children: list) -> None:
        if children != []:
            self.children = children
            self.first_child = children[0]
        else:
            self.children = None
            self.first_child = None

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
        self.update_depth(new_parent.depth + 1)

    def update_depth(self, new_depth: int):
        self.depth = new_depth
        if self.children is not None:
            for child in self.children:
                child.update_depth(new_depth+1)

    def set_x(self, x: int):
        self.node.location.x = x  # type: ignore

    def set_y(self, y: int):
        self.node.location.y = y  # type: ignore

    def get_x(self) -> int:
        return self.node.location.x  # type: ignore

    def get_y(self) -> int:
        return self.node.location.y  # type: ignore

    def get_width(self) -> int:
        return self.width

    def get_height(self) -> int:
        return self.height

    def is_leaf(self) -> bool:
        return self.first_child is None

    def has_right(self) -> bool:
        return self.right_sibling is not None

    def has_left(self) -> bool:
        return self.left_sibling is not None

    def good_name(self, node) -> str | None:
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
        return f"{self.good_name(self)} (depth= {self.depth}):\n \
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
    Algorithm: https://techreports.cs.unc.edu/papers/89-034.pdf
    """

    def __init__(self, context):
        prefs = context.preferences.addons['math_formula'].preferences
        self.level_separation: int = prefs.node_distance
        self.sibling_separation: int = prefs.sibling_distance
        self.subtree_separation: int = prefs.subtree_distance
        self.x_top_adjustment: int = 0
        self.y_top_adjustment: int = 0
        self.max_width_per_level: list[int] = [0 for _ in range(100)]
        self.prev_node_per_level: list['PositionNode' | None] = [
            None for _ in range(100)]
        self.min_x_loc: int = +INF
        self.max_x_loc: int = -INF
        self.min_y_loc: int = +INF
        self.max_y_loc: int = -INF
        self.visited_nodes: list[PositionNode] = []

    # Test formula:
    # p.xyz; r = length(p); theta = acos(z/r); phi = atan2(y,x); {r, theta, phi}
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
                        # TODO: make sure this is correct.
                        # Due to the DFS it's possible that a node's depth is
                        # increased after checking if it should update the parent.
                        # When this happens a parent update might be missed. This
                        # is not a drastic problem but should be tackled in the
                        # future.
                        if depth >= vnode.depth:
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
        for socket in cast(Node, pnode.node).inputs:
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

    def place_nodes(self, root_nodes: list[Node] | Node, links: NodeLinks, cursor_loc: tuple[int, int] | None = None) -> tuple[float, float] | None:
        """
        Aranges the nodes connected to `root_node` so that the top
        left corner lines up with `cursor_loc`. If `cursor_loc` is `None`,
        the tree is aligned such that `root_node` stays in the same place.
        If a list of root_nodes are supplied, a fake parent is created for
        these nodes to improve positioning.

        The returned value is the bottom right corner, i.e the place where
        you would want to place the next nodes, if `cursor_loc` is not `None`.
        Otherwise `None` is returned.
        """
        root_node = None
        if isinstance(root_nodes, list):
            # Use a dummy node as the parent of all the root nodes
            dummy = DummyNode()
            root_node = PositionNode(dummy, depth=0)
            for root in root_nodes:
                root_pnode = PositionNode(root, depth=1)
                self.visited_nodes.append(root_pnode)
            r_nodes = self.visited_nodes.copy()
            root_node.set_children(r_nodes)
            for i, child in enumerate(r_nodes):
                if i < len(r_nodes)-1:
                    child.right_sibling = r_nodes[i+1]
                if i > 0:
                    child.left_sibling = r_nodes[i-1]
                child.parent = root_node
            for pnode in r_nodes:
                self.build_relations(pnode, links, depth=1)
        else:
            root_node = PositionNode(root_nodes)
            self.build_relations(root_node, links, depth=0)
        self.visited_nodes = []
        old_root_node_pos_x: int = root_node.node.location.x  # type: ignore
        old_root_node_pos_y: int = root_node.node.location.y  # type: ignore
        self.first_walk(root_node, 0)
        self.x_top_adjustment = root_node.get_x()
        self.y_top_adjustment = root_node.get_y() - root_node.prelim_y
        self.second_walk(root_node, 0, 0, 0)
        offset_x = 0
        offset_y = 0
        if cursor_loc is not None:
            offset_x = - self.min_x_loc
            offset_y = cursor_loc[1] - self.max_y_loc
        else:
            offset_x = old_root_node_pos_x-root_node.get_x()
            offset_y = old_root_node_pos_y-root_node.get_y()
        for pnode in self.visited_nodes:

            pnode.set_x(pnode.get_x() + offset_x)
            pnode.set_y(pnode.get_y() + offset_y)
            if 'NodeReroute' in pnode.node.bl_idname:
                # It looks weird if it is placed at the top. This makes it a bit
                # more centrally placed, near the sockets.
                pnode.set_y(pnode.get_y()-30)
        if cursor_loc is not None:
            return (cursor_loc[0]+self.max_x_loc-self.min_x_loc, cursor_loc[1])

    def get_leftmost(self, node: PositionNode, level: int, depth: int) -> PositionNode | None:
        if level >= depth:
            return node
        if node.is_leaf():
            return None
        rightmost = cast(PositionNode, node.first_child)
        leftmost = cast(PositionNode, self.get_leftmost(
            rightmost, level + 1, depth))
        while leftmost is None and rightmost.has_right():
            rightmost = cast(PositionNode, rightmost.right_sibling)
            leftmost = self.get_leftmost(rightmost, level + 1, depth)
        return leftmost

    def get_prev_node_at_level(self, level: int) -> Optional[PositionNode]:
        return self.prev_node_per_level[level]

    def set_prev_node_at_level(self, level: int, node: PositionNode):
        self.prev_node_per_level[level] = node

    def apportion(self, node: PositionNode):
        leftmost = cast(PositionNode, node.first_child)
        neighbour = cast(PositionNode, leftmost.left_neighbour)
        compare_depth = 1
        while leftmost is not None and neighbour is not None:
            # Compute the location of leftmost and where it
            # should be with respect to neighbour
            left_mod_sum = right_mod_sum = 0
            ancestor_leftmost = leftmost
            ancestor_neighbour = neighbour

            for _ in range(compare_depth):
                ancestor_leftmost = cast(
                    PositionNode, ancestor_leftmost.parent)
                ancestor_neighbour = cast(
                    PositionNode,  ancestor_neighbour.parent)
                right_mod_sum += ancestor_leftmost.modifier

                left_mod_sum += ancestor_neighbour.modifier

            # Find the move_distance and apply it to the node's subtree
            # Add appropriate portions to smaller interior subtrees
            move_distance: int = neighbour.prelim_y +\
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
                    portion = move_distance//left_siblings
                    tmp = node
                    while tmp != ancestor_neighbour:
                        tmp.prelim_y += move_distance
                        tmp.modifier += move_distance
                        move_distance -= portion
                        tmp = cast(PositionNode, tmp.left_sibling)
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
                node.prelim_y = cast(PositionNode, node.left_sibling).prelim_y + \
                    self.sibling_separation + \
                    cast(PositionNode, node.left_sibling).get_height()
            else:
                node.prelim_y = 0
        else:
            # It's not a leaf, so recursively call for children
            leftmost = rightmost = cast(PositionNode, node.first_child)
            self.first_walk(leftmost, level+1)
            while rightmost.has_right():
                rightmost = cast(PositionNode, rightmost.right_sibling)
                self.first_walk(rightmost, level+1)
            mid = (leftmost.prelim_y + rightmost.prelim_y)//2
            if node.has_left():
                node.prelim_y = cast(PositionNode, node.left_sibling).prelim_y + \
                    self.sibling_separation + \
                    cast(PositionNode, node.left_sibling).get_height()
                node.modifier = node.prelim_y - mid
                self.apportion(node)
            else:
                node.prelim_y = mid
        self.max_width_per_level[level] = max(
            node.width, self.max_width_per_level[level])

    def second_walk(self, node: PositionNode, level: int, width_sum_x: int, mod_sum_y: int):
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
            self.second_walk(cast(PositionNode, node.first_child), level + 1,
                             width_sum_x +
                             self.max_width_per_level[level+1] +
                             self.level_separation,
                             mod_sum_y + node.modifier)
        if node.has_right():
            self.second_walk(cast(PositionNode, node.right_sibling),
                             level, width_sum_x, mod_sum_y)
