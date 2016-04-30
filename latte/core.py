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
import latte.transformers as transformers

def aligned(a, alignment=64):
    if (a.ctypes.data % alignment) == 0:
        return a

    extra = alignment // a.itemsize
    buf = np.empty(a.size + extra, dtype=a.dtype)
    ofs = (-buf.ctypes.data % alignment) // a.itemsize
    aa = buf[ofs:ofs+a.size].reshape(a.shape)
    np.copyto(aa, a)
    assert (aa.ctypes.data % alignment) == 0
    return aa

SIMDWIDTH = 8

class Connection:
    def __init__(self, source_ens, sink_ens, mapping, reshape):
        self.source = source_ens
        self.sink = sink_ens
        self.mapping = mapping
        self.mapping_inserted = False
        self.reshape = reshape

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
                # return ast.Name(node.attr, node.ctx)
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
            assert False
            # closure_vars = inspect.getclosurevars(self.connections[0].mapping)
            # mapping_func = util.get_ast(self.connections[0].mapping).body[0]
            # for var, value in closure_vars.nonlocals.items():
            #     mapping_func = util.InlineVariable(var, value).visit(mapping_func)
            # val = mapping_func.body[0].value
            # val = eval(astor.to_source(val))
            # if isinstance(val, int):
            #     node.iter = ast.Call(ast.Name("range", ast.Load()), [ast.Num(val)], [])
            #     node.body = [util.replace_name(node.target.elts[0], node.target.elts[1], s) 
            #                  for s in node.body]
            #     node.target = node.target.elts[1]
            #     node.body = [self.visit(s) for s in node.body]
            #     return node
            # else:
            #     raise NotImplementedError(ast.dump(node))
            #     # val = ast.Call(ast.Name("range", ast.Load()), [ast.Num(eval(astor.to_source(val)))], [])
            #     return ast.Call(ast.Name("enumerate", ast.Load()), [val], [])
        elif isinstance(_range, ast.Call) and _range.func.id in ["enumerate_dim", "range_dim"]:
            closure_vars = inspect.getclosurevars(self.connections[0].mapping)
            mapping_func = util.get_ast(self.connections[0].mapping).body[0]
            for var, value in closure_vars.nonlocals.items():
                mapping_func = util.InlineVariable(var, value).visit(mapping_func)
            for i, arg in enumerate(mapping_func.args.args):
                i += 1  # offset batch
                mapping_func = util.InlineVariable(arg.arg, "_neuron_index_{}".format(i)).visit(mapping_func)

            shape = mapping_func.body[-1].value
            node.iter = shape.elts[_range.args[1].n]
            if _range.func.id == "enumerate_dim":
                node.iter = ast.Call(ast.Name("enumerate", ast.Load()), [node.iter], [])
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

    def add_connections(self, source_ens, sink_ens, mapping, reshape=None):
        self.connections.append(Connection(source_ens, sink_ens, mapping, reshape))

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
                    self.buffers[(ensemble.name + "value")] = aligned(np.zeros(_shape, dtype=np.float32))
                elif field == "grad":
                    _shape = (self.batch_size, ) + ensemble.shape
                    self.buffers[(ensemble.name + "grad")] = aligned(np.zeros(_shape, dtype=np.float32))
                elif field == "inputs":
                    conn = connections_map[ensemble][0]
                    source_name = conn.source.name
                    buff = self.buffers[(source_name + "value")]
                    if conn.reshape is not None:
                        buff = buff.reshape((self.batch_size, ) + conn.reshape)
                    self.buffers[(ensemble.name + "inputs")] = buff
                elif field == "grad_inputs":
                    conn = connections_map[ensemble][0]
                    source_name = conn.source.name
                    buff = self.buffers[(source_name + "grad")]
                    if conn.reshape is not None:
                        buff = buff.reshape((self.batch_size, ) + conn.reshape)
                    self.buffers[(ensemble.name + "grad_inputs")] = buff
                else:
                    value = getattr(neuron, field)
                    if isinstance(value, numbers.Real):
                        buff = aligned(np.empty(ensemble.shape, dtype=type(value)))
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

                        buff = aligned(np.empty(shape, dtype=value.dtype))
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

            func, args = self._synthesize_ast(ensemble, neuron.forward, "forward")
            self.forward_tasks.append(Task(func, args))

            func, args = self._synthesize_ast(ensemble, neuron.backward, "backward")
            self.backward_tasks.insert(0, Task(func, args))

    def _synthesize_ast(self, ensemble, fn, direction):
        fn_def = util.get_ast(fn).body[0]
        transformer = NeuronTransformer(ensemble, self.connections_map[ensemble], self.buffer_dim_info)
        self.connections_map[ensemble][0].mapping_inserted = False
        fn_def = transformer.visit(fn_def)
        loop_vars = ["_neuron_index_{}".format(i) for i in range(ensemble.ndim + 1)][::-1]
        loop_ranges = [self.batch_size] + [d for d in ensemble.shape]
        loop_ranges = loop_ranges[::-1]
        body = fn_def.body
        # Add value to args
        # transformer.seen_vars.add(ensemble.name+to_load)

        nests = [util.gen_loop_nest([s], loop_vars, loop_ranges) for s in body]
        args = [ast.arg(arg, None) for arg in transformer.seen_vars]
        func_name = util.generate_unique_function_name()
        func_def = ast.FunctionDef(func_name,
                ast.arguments(args, None, [], [], None, []), nests,
                [], None)
        func_def = transformers.pattern_match_gemm(func_def)
        func_def = ast.fix_missing_locations(func_def)
        func_def = transformers.simple_fusion(func_def)
        func_def = transformers.register_promote_value_refs(func_def, ensemble,
                direction, self.batch_size)
        # func_def = transformers.flatten_subscripts(func_def, self.buffers)
        func_def = transformers.convert_tuple_subscripts(func_def)
        func_def = transformers.convert_enumerate_ranges(func_def)
        func_def = transformers.convert_sgemm_calls(func_def)
        func_def = PyBasicConversions().visit(func_def)
        func_def, vectorized_buffers = transformers.vectorize_outer_loop(func_def)
        # vectorized_buffers[ensemble.name+"inputs"] = ensemble.ndim - 1
        type_sig = []
        for arg in func_def.params:
            name = arg.name
            arg.type = ctypes.POINTER(ctypes.c_float)()
            arg.name = "_{}".format(name)
            buf = self.buffers[name]
            type_sig.append(np.ctypeslib.ndpointer(buf.dtype, buf.ndim, buf.shape))
            buf_shape = buf.shape
            if name.endswith("value") or name.endswith("grad"):
                buf_shape = (max(buf_shape[1] // SIMDWIDTH, 1), *buf_shape[2:], SIMDWIDTH)
            elif name in vectorized_buffers:
                dim_to_vectorize = buf.ndim - vectorized_buffers[name] - 1
                buf_shape = list(buf_shape)
                buf_shape[dim_to_vectorize] //= SIMDWIDTH
                buf_shape = tuple(buf_shape[1:]) + (SIMDWIDTH, )
            else:
                buf_shape = buf_shape[1:]
            shape = ""
            cast = ""
            for d in buf_shape:
                shape += "[{}]".format(d)
                cast += "[{}]".format(d)
            func_def.defn.insert(0, StringTemplate("""float (* __restrict $arg_name)$shape = (float (*)$cast) _$arg_name;""",
                {"arg_name": C.SymbolRef(name), "shape": C.SymbolRef(shape),
                "cast": C.SymbolRef(cast)}))
        type_sig = ctypes.CFUNCTYPE(None, *type_sig)
        include = StringTemplate("""
                #include <immintrin.h>
                #include <math.h>
                #include <mkl.h>
                #define SIMDWIDTH 8
            """)
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
