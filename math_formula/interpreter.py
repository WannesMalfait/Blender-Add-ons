import bpy
from bpy.types import Node, NodeSocket
from math_formula.backends.type_defs import Operation, OpType, ValueType, NodeInstance, CompiledFunction


class Interpreter():

    def __init__(self, tree: bpy.types.NodeTree) -> None:
        self.tree = tree
        self.stack: list[ValueType] = []
        # The nodes that we added
        self.nodes: list[Node] = []
        # Variables in the form of output sockets
        self.variables: dict[str, NodeSocket] = {}
        self.function_outputs: list = []

    def operation(self, operation: Operation):
        op_type = operation.op_type
        op_data = operation.data
        assert OpType.END_OF_STATEMENT.value == 11, 'Exhaustive handling of Operation types.'
        if op_type == OpType.PUSH_VALUE:
            self.stack.append(op_data)
        elif op_type == OpType.CREATE_VAR:
            assert isinstance(
                op_data, str), 'Variable name should be a string.'
            socket = self.stack.pop()
            assert isinstance(
                socket, (NodeSocket, list)), 'Create var expects a node socket or struct.'
            self.variables[op_data] = socket
        elif op_type == OpType.GET_VAR:
            assert isinstance(
                op_data, str), 'Variable name should be a string.'
            self.stack.append(self.variables[op_data])
        elif op_type == OpType.GET_OUTPUT:
            assert isinstance(
                op_data, int), 'Bug in type checker, index should be int.'
            index = op_data
            struct = self.stack.pop()
            assert isinstance(
                struct, list), 'Bug in type checker, GET_OUTPUT only works on structs.'
            # Index order is reversed
            self.stack.append(struct[-index-1])
        elif op_type == OpType.SET_OUTPUT:
            assert isinstance(
                op_data, tuple), 'Data should be tuple of index and value'
            index, value = op_data
            self.nodes[-1].outputs[index].default_value = value
        elif op_type == OpType.SET_FUNCTION_OUT:
            assert isinstance(op_data, int), 'Data should be an index'
            self.function_outputs[op_data] = self.stack.pop()
        elif op_type == OpType.SPLIT_STRUCT:
            struct = self.stack.pop()
            assert isinstance(
                struct, list), 'Bug in type checker, GET_OUTPUT only works on structs.'
            self.stack += struct
        elif op_type == OpType.CALL_FUNCTION:
            assert isinstance(
                op_data, CompiledFunction), 'Bug in type checker.'
            args = self.get_args(self.stack, len(op_data.inputs))
            # Store state outside function, and prepare state in function
            outer_vars = self.variables
            self.variables = {}
            for name, arg in zip(op_data.inputs, args):
                self.variables[name] = arg
            outer_function_outputs = self.function_outputs
            self.function_outputs = [None for _ in range(op_data.num_outputs)]
            outer_stack = self.stack
            self.stack = []
            # Execute function
            for operation in op_data.body:
                self.operation(operation)
            # Restore state outside function
            self.stack = outer_stack
            if len(self.function_outputs) == 1:
                self.stack.append(self.function_outputs[0])
            elif len(self.function_outputs) > 1:
                self.stack.append(list(reversed(self.function_outputs)))
            self.function_outputs = outer_function_outputs
            self.variables = outer_vars
        elif op_type == OpType.CALL_NODEGROUP:
            raise NotImplementedError
        elif op_type == OpType.CALL_BUILTIN:
            assert isinstance(op_data, NodeInstance), 'Bug in compiler.'
            args = self.get_args(self.stack, len(op_data.inputs))
            node = self.add_builtin(op_data, args,)
            outputs = op_data.outputs
            if len(outputs) == 1:
                self.stack.append(node.outputs[outputs[0]])
            elif len(outputs) > 1:
                self.stack.append([node.outputs[o] for o in reversed(outputs)])
            self.nodes.append(node)
        elif op_type == OpType.RENAME_NODE:
            self.nodes[-1].label = op_data
        elif op_type == OpType.END_OF_STATEMENT:
            self.stack = []
        else:
            print(f'Need implementation of {op_type}')
            raise NotImplementedError

    def get_args(self, stack: list, num_args: int) -> list[ValueType]:
        if num_args == 0:
            return []
        args = stack[-num_args:]
        stack[:] = stack[:-num_args]
        return args

    def add_builtin(self, node_info: NodeInstance, args: list[ValueType]) -> bpy.types.Node:
        tree = self.tree
        node = tree.nodes.new(type=node_info.key)
        for name, value in node_info.props:
            setattr(node, name, value)
        for i, input_index in enumerate(node_info.inputs):
            arg = args[i]
            if isinstance(arg, bpy.types.NodeSocket):
                tree.links.new(arg, node.inputs[input_index])
            elif not (arg is None):
                node.inputs[input_index].default_value = arg
        return node