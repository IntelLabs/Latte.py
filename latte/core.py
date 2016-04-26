import numpy as np
import numbers
import itertools
import ast
from .ensemble import Ensemble, DataEnsemble
import latte.util as util
import astor
from itertools import product
import inspect
from .util import sgemm
import ctree
from ctree.transformations import PyBasicConversions
import ctree.c.nodes as C
from ctree.templates.nodes import StringTemplate
import ctypes

class Connection:
    def __init__(self, source_ens, sink_ens, mapping):
        self.source = source_ens
        self.sink = sink_ens
        self.mapping = mapping
        self.mapping_inserted = False

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
    def __init__(self, ensemble, connections, buffer_dim_info):
        super().__init__()
        self.ensemble = ensemble
        self.seen_vars = set()
        self.connections = connections
        self.buffer_dim_info = buffer_dim_info

    def visit(self, node):
        node = super().visit(node)
        if hasattr(node, 'body'):
            node.body = util.flatten(node.body)
        return node

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
            args = []
            for i in range(ndims):
                if name not in self.buffer_dim_info or not self.buffer_dim_info[name][i]:
                    args.append(ast.Name("_neuron_index_{}".format(i + incr), ast.Load()))
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
            elif isinstance(node.slice.value, ast.Tuple):
                value.slice.value.elts.extend(node.slice.value.elts)
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
        elif isinstance(_range, ast.Call) and _range.func.id == "enumerate_dim":
            closure_vars = inspect.getclosurevars(self.connections[0].mapping)
            mapping_func = util.get_ast(self.connections[0].mapping).body[0]
            for var, value in closure_vars.nonlocals.items():
                mapping_func = util.InlineVariable(var, value).visit(mapping_func)
            for i, arg in enumerate(mapping_func.args.args):
                i += 1  # offset batch
                mapping_func = util.InlineVariable(arg.arg, "_neuron_index_{}".format(i)).visit(mapping_func)

            shape = mapping_func.body[-1].value
            node.iter = ast.Call(ast.Name("enumerate", ast.Load()), [shape.elts[_range.args[1].n]], [])
            if self.connections[0].mapping_inserted:
                pre_stmts = []
            else:
                pre_stmts = mapping_func.body[:-1]
                for stmt in pre_stmts:
                    assert isinstance(stmt, ast.Assign)
                    assert len(stmt.targets) == 1
                    stmt.targets[0] = C.SymbolRef(stmt.targets[0].id, ctypes.c_int())
                self.connections[0].mapping_inserted = True
            node.body = [self.visit(s) for s in node.body]
            node.body = util.flatten(node.body)
            return pre_stmts + [node]
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

class ConvertEnumerateRange(ast.NodeTransformer):
    def visit(self, node):
        node = super().visit(node)
        if hasattr(node, 'body'):
            node.body = util.flatten(node.body)
        return node

    def visit_For(self, node):
        node.body = util.flatten([self.visit(s) for s in node.body])
        if isinstance(node.iter, ast.Call) and node.iter.func.id == "enumerate":
            assert node.iter.args[0].func.id == "range"
            range_args = node.iter.args[0].args
            if len(range_args) == 1:
                init = [C.Assign(
                    C.SymbolRef(node.target.elts[0].id),
                    C.Constant(0)),
                    C.Assign(
                        C.SymbolRef(node.target.elts[1].id),
                        C.Constant(0))]
                end = node.iter.args[0].args[0]
            elif len(range_args) == 2:
                start = node.iter.args[0].args[0] 
                if isinstance(start, ast.Name):
                    start = C.SymbolRef(start.id)
                elif isinstance(start, ast.Num):
                    start = C.Constant(start.n)
                else:
                    raise NotImplementedError
                init = [C.Assign(
                    C.SymbolRef(node.target.elts[0].id),
                    C.Constant(0)),
                    C.Assign(
                        C.SymbolRef(node.target.elts[1].id),
                        C.Constant(start))]
                end = node.iter.args[0].args[1]
            else:
                raise NotImplementedError

            if isinstance(end, ast.Name):
                end = C.SymbolRef(end.id)
            elif isinstance(end, ast.Num):
                end = C.Constant(end.n)
            else:
                raise NotImplementedError
            pre_stmts = [C.SymbolRef(var.id, ctypes.c_int()) for var in node.target.elts]
            return pre_stmts + [C.For(
                    init,
                    C.Lt(C.SymbolRef(node.target.elts[1].id), end),
                    [C.PostInc(C.SymbolRef(var.id)) for var in node.target.elts],
                    node.body
                )]
        return node

class SimpleFusion(ast.NodeTransformer):
    def visit(self, node):
        node = super().visit(node)
        if hasattr(node, 'body') and len(node.body) > 1:
            new_body = [node.body[0]]
            for statement in node.body[1:]:
                if isinstance(new_body[-1], ast.For) and \
                        isinstance(statement, ast.For) and \
                        astor.to_source(statement.iter) == astor.to_source(new_body[-1].iter) and \
                        astor.to_source(statement.target) == astor.to_source(new_body[-1].target):
                    new_body[-1].body.extend(statement.body)
                else:
                    new_body.append(statement)
            node.body = [self.visit(s) for s in new_body]

        return node

class ConvertSGEMMCalls(ast.NodeTransformer):
    def visit_Call(self, node):
        if node.func.id == "sgemm":
            for i in range(2):
                node.args[i] = C.Constant(112) if node.args[i].id == "True" else C.Constant(111) 
            node.args.insert(0, C.Constant(101))
            node.func.id = "cblas_sgemm"
        return node

class FlattenSubscripts(ast.NodeTransformer):
    def __init__(self, buffers):
        self.buffers = buffers

    def visit_Subscript(self, node):
        assert node.value.id in self.buffers
        shape = self.buffers[node.value.id].shape
        if isinstance(node.slice.value, ast.Tuple):
            idxs = node.slice.value.elts
            flat_idx = idxs[0]
            for i in range(len(idxs[1:])):
                flat_idx = ast.BinOp(
                        ast.BinOp(
                            flat_idx,
                            ast.Mult(),
                            ast.Num(shape[i + 1])
                        ),
                        ast.Add(),
                        idxs[i + 1]
                    )
            node.slice.value = flat_idx
        return node

class Net:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.ensembles = []
        self.connections = []
        self.buffers = {}
        self.forward_tasks = []
        self.backward_tasks = []
        self.connections_map = {}
        self.buffer_dim_info = {}

    def add_ensemble(self, ensemble):
        self.ensembles.append(ensemble)

    def init_ensemble(self, neurons):
        ens = Ensemble(neurons)
        self.ensembles.append(ens)
        return ens

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
                        uniform_across_dim = [True for _ in range(len(_shape))]
                        first = getattr(ensemble.neurons.flat[0], field)
                        for d in range(len(_shape)):
                            for i in range(_shape[d]):
                                idx = tuple(i if x == d else 0 for x in range(len(_shape)))
                                if first.ctypes.data != getattr(ensemble.neurons[idx], field).ctypes.data:
                                    uniform_across_dim[d] = False
                                    break
                        _iter = []
                        shape = []
                        for i in range(len(_shape)):
                            if not uniform_across_dim[i]:
                                _iter.append(range(_shape[i]))
                                shape.append(_shape[i])
                            else:
                                _iter.append(range(1))
                        shape += first.shape

                        buff = np.empty(shape, dtype=value.dtype)
                        self.buffers[(ensemble.name + field)] = buff
                        self.buffer_dim_info[ensemble.name + field] = uniform_across_dim
                        for index in itertools.product(*_iter):
                            _index = []
                            for i in range(len(uniform_across_dim)):
                                if not uniform_across_dim[i]:
                                    _index.append(index[i])
                            buff[_index] = getattr(ensemble.neurons[index], field)
                    else:
                        raise NotImplementedError(field)
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
        transformer = NeuronTransformer(ensemble, self.connections_map[ensemble], self.buffer_dim_info)
        self.connections_map[ensemble][0].mapping_inserted = False
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
        func_def = SimpleFusion().visit(func_def)
        func_def = FlattenSubscripts(self.buffers).visit(func_def)
        func_def = ConvertEnumerateRange().visit(func_def)
        func_def = ConvertSGEMMCalls().visit(func_def)
        func_def = PyBasicConversions().visit(func_def)
        type_sig = []
        for arg in func_def.params:
            arg.type = ctypes.POINTER(ctypes.c_float)()
            buf = self.buffers[arg.name]
            type_sig.append(np.ctypeslib.ndpointer(buf.dtype, buf.ndim, buf.shape))
        type_sig = ctypes.CFUNCTYPE(None, *type_sig)
        include = StringTemplate("#include <math.h>\n#include <mkl.h>")
        print(func_def)
        module = ctree.nodes.Project([C.CFile(func_name, [include, func_def])]).codegen()
        fn = module.get_callable(func_name, type_sig)

        _args = [self.buffers[arg.arg] for arg in args]
        return fn, _args

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
