import numpy as np
import numbers
import ast
from .neuron import Neuron, WeightedNeuron
from .ensemble import Ensemble, DataEnsemble
import latte.util as util
import astor
from itertools import product
import inspect
from .util import sgemm

class Connection:
    def __init__(self, source_ens, sink_ens, mapping):
        self.source = source_ens
        self.sink = sink_ens
        self.mapping = mapping

class Task:
    def __init__(self, fn, args):
        self.fn = fn
        self.args = args

    def __call__(self):
        self.fn(*self.args)

# class Task:
#     def __init__(self, neuron):
#         self.neuron = neuron
#         self.inputs = []

# class TaskGroup:
#     def __init__(self, tasks):
#         self.tasks = tasks

#     def __iter__(self):
#         return np.ndenumerate(self.tasks)
    
#     def __getitem__(self, index):
#         return self.tasks[index]

class NeuronTransformer(ast.NodeTransformer):
    def __init__(self, ensemble, connections):
        super().__init__()
        self.ensemble = ensemble
        self.seen_vars = set()
        self.connections = connections

    def visit_Attribute(self, node):
        if node.value.id == "self":
            name = "{}{}".format(self.ensemble.name, node.attr)
            self.seen_vars.add(name)
            ndims = self.ensemble.ndim
            incr = 0
            if node.attr in ["value", "grad"]:
                ndims += 1
            elif node.attr in ["inputs", "grad_inputs"]:
                ndims = 1
            else:
                incr = 1
            args = [ast.Name("_neuron_index_{}".format(i + incr), ast.Load()) for i in range(ndims)]
            # if node.attr in ["value", "inputs"]:
            #     args.append(C.Dot(C.SymbolRef(next_rdom), C.SymbolRef("x")))
            node = ast.Subscript(ast.Name(name, ast.Load()), ast.Index(ast.Tuple(args, ast.Load())), node.ctx)
            return node
        else:
            raise Exception("Unsupported Attribute node")

    def visit_Subscript(self, node):
        node.value = self.visit(node.value)
        # assert isinstance(node.value, ast.Call)
        # if isinstance(node.value, ast.Name) and "inputs" in node.value.id:
        #     assert False
        #     value = node.value
        #     assert isinstance(node.slice, ast.Index)
        #     value.id = "{}{}".format(value.id, node.slice.value.n)
        #     return value
        if isinstance(node.value, ast.Subscript):
            value = node.value
            value.ctx = node.ctx
            if isinstance(node.slice.value, (ast.Name, ast.Num)):
                value.slice.value.elts.append(node.slice.value)
            else:
                raise NotImplementedError(node.slice.value)
            return value
        else:
            raise NotImplementedError()
        return node

    def visit_For(self, node):
        _range = node.iter
        if isinstance(_range, ast.Call) and _range.func.id == "enumerate_mapping":
            closure_vars = inspect.getclosurevars(self.connections[0].mapping)
            mapping_func = util.get_ast(self.connections[0].mapping).body[0]
            for var, value in closure_vars.nonlocals.items():
                mapping_func = util.InlineVariable(var, value).visit(mapping_func)
            val = mapping_func.body[0].value
            val = eval(astor.to_source(val))
            if isinstance(val, int):
                node.iter = ast.Call(ast.Name("range", ast.Load()), [ast.Num(val)], [])
                node.body = [util.replace_name(node.target.elts[0], node.target.elts[1], s) 
                             for s in node.body]
                node.target = node.target.elts[1]
                node.body = [self.visit(s) for s in node.body]
                return node
            else:
                raise NotImplementedError(ast.dump(node))
                # val = ast.Call(ast.Name("range", ast.Load()), [ast.Num(eval(astor.to_source(val)))], [])
                return ast.Call(ast.Name("enumerate", ast.Load()), [val], [])
        else:
            raise NotImplementedError(ast.dump(node))

def extend_or_append(_list, value):
    if isinstance(value, list):
        _list.extend(value)
    else:
        _list.append(value)

class Unpack(ast.NodeTransformer):
    def __init__(self):
        super().__init__()
        self.counter = 0
        self.curr_tmp_var = None

    def _gen_next_tmp_var(self):
        self.counter += 1
        self.curr_tmp_var = "v{}".format(self.counter)
        return self.curr_tmp_var

    def visit_FunctionDef(self, node):
        new_body = []
        for statement in node.body:
            result = self.visit(statement)
            extend_or_append(new_body, result)
        node.body = new_body
        return node

    def visit_Assign(self, node):
        block = []
        if not isinstance(node.value, ast.Name):
            result = self.visit(node.value)
            extend_or_append(block, result[:-1])
            node.value = result[-1]
        if len(block) > 0:
            return block + [node]

    def visit_BinOp(self, node):
        block = []
        if not isinstance(node.left, ast.Name):
            var = self._gen_next_tmp_var()
            result = self.visit(node.left)
            node.left = ast.Name(var, ast.Load())
            extend_or_append(block, result)
            # block[-1] = ast.Assign([C.SymbolRef(var, ctree.c.HalideFunc())], block[-1])
        if not isinstance(node.right, ast.Name):
            var = self._gen_next_tmp_var()
            result = self.visit(node.right)
            extend_or_append(block, result)
            node.right = ast.Name(var, ast.Load())
            # block[-1] = ast.Assign([C.SymbolRef(var, ctree.c.HalideFunc())], block[-1])
        if len(block) > 0:
            return block + [node]

    def visit_Call(self, node):
        block = []
        new_args = []
        for arg in node.args:
            if not isinstance(arg, ast.Name):
                var = self._gen_next_tmp_var()
                result = self.visit(arg)
                extend_or_append(block, result)
                # block[-1] = ast.Assign([C.SymbolRef(var, ctree.c.HalideFunc())], block[-1])
                new_args.append(ast.Name(var, ast.Load()))
        node.args = new_args
        if len(block) > 0:
            return block + [node]
class Net:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.ensembles = []
        self.connections = []
        self.buffers = {}
        self.forward_tasks = []
        self.backward_tasks = []
        self.connections_map = {}

    def add_ensemble(self, ensemble):
        self.ensembles.append(ensemble)

    def add_connections(self, source_ens, sink_ens, mapping):
        self.connections.append(Connection(source_ens, sink_ens, mapping))

    def compile(self):
        task_groups = {}
        connections_map = {ensemble: [] for ensemble in self.ensembles}
        self.connections_map = connections_map
        for connection in self.connections:
            connections_map[connection.sink].append(connection)

        for ensemble in self.ensembles:
            neuron = ensemble.neurons.flat[0]
            # tasks = np.empty(ensemble.shape, dtype='object')
            # for i, neuron in np.ndenumerate(ensemble.neurons):
            #     tasks[i] = Task(ensemble.neurons[i])
            # task_groups[ensemble] = TaskGroup(tasks)

            for field in vars(neuron):
                if field == "value":
                    _shape = (self.batch_size, ) + ensemble.shape
                    self.buffers[(ensemble.name + "value")] = np.zeros(_shape, dtype=np.float32)
                elif field == "grad":
                    _shape = (self.batch_size, ) + ensemble.shape
                    self.buffers[(ensemble.name + "grad")] = np.zeros(_shape, dtype=np.float32)
                elif field == "inputs":
                    source_name = connections_map[ensemble][0].source.name
                    self.buffers[(ensemble.name + "inputs")] = self.buffers[(source_name + "value")]
                elif field == "grad_inputs":
                    source_name = connections_map[ensemble][0].source.name
                    self.buffers[(ensemble.name + "grad_inputs")] = self.buffers[(source_name + "grad")]
                else:
                    value = getattr(neuron, field)
                    if isinstance(value, numbers.Real):
                        buff = np.empty(ensemble.shape, dtype=type(value))
                        self.buffers[(ensemble.name + field)] = buff
                        for i, v in ensemble:
                            buff[i] = getattr(v, field)
                    elif isinstance(value, np.ndarray):
                        _shape = ensemble.shape
                        _shape += value.shape
                        buff = np.empty(_shape, dtype=value.dtype)
                        self.buffers[(ensemble.name + field)] = buff
                        for i, v in ensemble:
                            i += tuple(np.arange(0,d) for d in value.shape)
                            buff[i] = getattr(v, field)
                    else:
                        print(field)
            if isinstance(ensemble, DataEnsemble):
                self.forward_tasks.append(
                    Task(ensemble.forward, [self.buffers[ensemble.name + "value"]]))
                continue

            func, args = self._synthesize_ast(ensemble, neuron.forward)
            self.forward_tasks.append(Task(func, args))

            func, args = self._synthesize_ast(ensemble, neuron.backward)
            self.backward_tasks.insert(0, Task(func, args))

    def _synthesize_ast(self, ensemble, fn):
        fn_def = util.get_ast(fn).body[0]
        transformer = NeuronTransformer(ensemble, self.connections_map[ensemble])
        fn_def = transformer.visit(fn_def)
        loop_vars = ["_neuron_index_{}".format(i) for i in range(ensemble.ndim + 1)][::-1]
        loop_ranges = [self.batch_size] + [d for d in ensemble.shape]
        loop_ranges = loop_ranges[::-1]
        nests = [util.gen_loop_nest([s], loop_vars, loop_ranges) for s in fn_def.body]
        args = [ast.arg(arg, None) for arg in transformer.seen_vars]
        func_name = util.generate_unique_function_name()
        func_def = ast.FunctionDef(func_name,
                ast.arguments(args, None, [], [], None, []), nests,
                [], None)
        func_def = util.PatternMatchGemm().visit(func_def)
        func_def = ast.fix_missing_locations(func_def)
        # print(astor.to_source(func_def))
        exec(compile(ast.Module([func_def]), filename="<ast>", mode="exec"))
        func = eval(func_name)
        _args = [self.buffers[arg.arg] for arg in args]
        return func, _args

    def forward(self):
        for task in self.forward_tasks:
            task()

    def backward(self):
        for task in self.backward_tasks:
            task()

def prod(elts):
    result = 1
    for _range in elts:
        result *= (_range.stop - _range.start)
    return result
