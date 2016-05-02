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

SIMDWIDTH = 8
TILE_SIZE = SIMDWIDTH

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

class NeuronTransformer(ast.NodeTransformer):
    def __init__(self, ensemble, connections, buffer_dim_info):
        super().__init__()
        self.ensemble = ensemble
        self.seen_vars = set()
        self.connections = connections
        self.buffer_dim_info = buffer_dim_info

    def visit(self, node):
        """
        Support replacing nodes with a list of nodes by flattening `body`
        fields.
        """
        node = super().visit(node)
        if hasattr(node, 'body'):
            node.body = util.flatten(node.body)
        return node

    def visit_Attribute(self, node):
        """
        A reference `self.field[...]` will be replaced with an array reference
        ensemble_namefield[...] to reflect SOA (struct of array) layout.
        """
        if node.value.id == "self":
            # name is ensemble name + attribute
            name = self.ensemble.name + node.attr

            # mark as seen
            self.seen_vars.add(name)

            ndim = self.ensemble.ndim
            offset = 0

            if node.attr in ["value", "grad"]:
                # increment ndim for fields that have a batch dimension
                ndim += 1
            elif node.attr in ["inputs", "grad_inputs"]:
                # only generate batch index for inputs/grad_inputs because
                # the user will provide rest of indices in expression
                ndim = 1
            else:
                # fields that don't have a batch dimension start at an offset 1
                # as 0 is the batch dimension
                offset = 1
            args = []
            for i in range(ndim):
                # only append this dimension if it is not fixed in self.buffer_dim_info
                # (used for shared values)
                if name not in self.buffer_dim_info or not self.buffer_dim_info[name][i]:
                    args.append(ast.Name("_neuron_index_{}".format(i + offset), ast.Load()))

            # return updated indedxing expression
            return ast.Subscript(ast.Name(name, ast.Load()), 
                                 ast.Index(ast.Tuple(args, ast.Load())), node.ctx)
        else:
            raise Exception("Unsupported Attribute node")

    def visit_Subscript(self, node):
        """
        If self.visit(node.value) returns a Subscript, flatten that expression
        into the current subscript node.
        """
        node.value = self.visit(node.value)
        if isinstance(node.value, ast.Subscript):
            value = node.value
            # append or extend the indexing expressions for *current* node to child node
            if isinstance(node.slice.value, (ast.Name, ast.Num)):
                value.slice.value.elts.append(node.slice.value)
            elif isinstance(node.slice.value, ast.Tuple):
                value.slice.value.elts.extend(node.slice.value.elts)
            else:
                raise NotImplementedError(node.slice.value)
            # return child node
            return value
        else:
            raise NotImplementedError()
        return node

    def visit_For(self, node):
        """
        Converts iteration expressions into enumerate and range calls
        """
        _range = node.iter
        if isinstance(_range, ast.Call) and _range.func.id in ["enumerate_dim", "range_dim"]:
            # grab closure variables and inline them into the mapping ast
            closure_vars = inspect.getclosurevars(self.connections[0].mapping)
            mapping_func = util.get_ast(self.connections[0].mapping).body[0]
            for var, value in closure_vars.nonlocals.items():
                mapping_func = util.inline_variable(var, value, mapping_func)

            # replace argument variables with loop variables corresponding to 
            # the current _neuron_index
            for i, arg in enumerate(mapping_func.args.args):
                i += 1  # offset batch
                mapping_func = util.inline_variable(arg.arg, "_neuron_index_{}".format(i), mapping_func)
            
            # the dimension is the second argument i.e. range_dim(self.inputs, dim)
            target_dim = _range.args[1].n
            shape = mapping_func.body[-1].value
            node.iter = shape.elts[target_dim]

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

    def _get_uniformity(self, ensemble, field):
        _shape = ensemble.shape
        uniform_across_dim = [True for _ in range(len(_shape))]
        first = getattr(ensemble.neurons.flat[0], field)
        for d in range(len(_shape)):
            for i in range(_shape[d]):
                idx = tuple(i if x == d else 0 for x in range(len(_shape)))
                if first.ctypes.data != getattr(ensemble.neurons[idx], field).ctypes.data:
                    uniform_across_dim[d] = False
                    break
        return uniform_across_dim

    def _init_buffers(self, ensemble):
        neuron = ensemble.neurons.flat[0]
        for field in vars(neuron):
            if field in ["value", "grad"]:
                _shape = (self.batch_size, ) + ensemble.shape
                self.buffers[ensemble.name + field] = util.zeros(_shape, np.float32)
            elif field in ["inputs", "grad_inputs"]:
                conn = self.connections_map[ensemble][0]
                source_name = conn.source.name
                source_target = "value" if field == "inputs" else "grad"
                buff = self.buffers[source_name + source_target]
                if conn.reshape is not None:
                    buff = buff.reshape((self.batch_size, ) + conn.reshape)
                self.buffers[ensemble.name + field] = buff
            else:
                value = getattr(neuron, field)

                if isinstance(value, numbers.Real):
                    buff = util.empty(ensemble.shape, type(value))
                    self.buffers[ensemble.name + field] = buff
                    for index, neuron in ensemble:
                        buff[index] = getattr(neuron, field)

                elif isinstance(value, np.ndarray):
                    _shape = ensemble.shape

                    uniform_across_dim = self._get_uniformity(ensemble, field)
                    shape = []
                    _iter = []
                    for i in range(len(_shape)):
                        if not uniform_across_dim[i]:
                            _iter.append(range(_shape[i]))
                            shape.append(_shape[i])
                        else:
                            _iter.append(range(1))
                    shape += value.shape

                    buff = util.empty(shape, value.dtype)
                    self.buffers[ensemble.name + field] = buff
                    self.buffer_dim_info[ensemble.name + field] = uniform_across_dim

                    for index in itertools.product(*_iter):
                        _index = []
                        for i in range(len(uniform_across_dim)):
                            if not uniform_across_dim[i]:
                                _index.append(index[i])
                        buff[_index] = getattr(ensemble.neurons[index], field)
                else:
                    raise NotImplementedError(field)

    def compile(self):
        task_groups = {}
        self.connections_map = {ensemble: [] for ensemble in self.ensembles}
        for connection in self.connections:
            self.connections_map[connection.sink].append(connection)

        for ensemble in self.ensembles:
            self._init_buffers(ensemble)
            if isinstance(ensemble, DataEnsemble):
                self.forward_tasks.append(
                    Task(ensemble.forward, [self.buffers[ensemble.name + "value"]]))
            else:
                neuron = ensemble.neurons.flat[0]

                func, args = self._synthesize_ast(ensemble, neuron.forward, "forward")
                self.forward_tasks.append(Task(func, args))

                # func, args = self._synthesize_ast(ensemble, neuron.backward, "backward")
                # self.backward_tasks.insert(0, Task(func, args))

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
        func_def = transformers.simple_fusion(func_def)

        func_def = transformers.register_promote_value_refs(func_def, ensemble,
                direction, self.batch_size)

        func_def = transformers.convert_tuple_subscripts(func_def)
        func_def, tiled_buffers = transformers.convert_enumerate_ranges(func_def)
        func_def = transformers.convert_sgemm_calls(func_def)
        func_def = PyBasicConversions().visit(func_def)
        func_def, vectorized_buffers = transformers.vectorize_outer_loop(func_def)

        for key in vectorized_buffers.keys():
            vectorized_buffers[key] = [(vectorized_buffers[key], SIMDWIDTH)]
        for key in tiled_buffers.keys():
            if key in vectorized_buffers:
                vectorized_buffers[key].insert(0, (tiled_buffers[key], TILE_SIZE))
            else:
                vectorized_buffers[key] = [(tiled_buffers[key], TILE_SIZE)]
        print(vectorized_buffers)

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
                buf_shape = list(buf_shape)
                for (dim, factor) in vectorized_buffers[name]:
                    dim_to_vectorize = len(buf_shape) - dim - 1
                    buf_shape[dim_to_vectorize] //= factor
                    buf_shape.append(factor)
                buf_shape = tuple(buf_shape[1:])
            else:
                buf_shape = buf_shape[1:]
            self._insert_cast(func_def.defn, buf_shape, name)
        type_sig = ctypes.CFUNCTYPE(None, *type_sig)
        include = StringTemplate("""
                #include <immintrin.h>
                #include <math.h>
                #include <mkl.h>
                #define SIMDWIDTH 8
                #define TILE_SIZE 8
                #define MIN(x, y) (((x) < (y)) ? (x) : (y))
            """)
        print(func_def)
        module = ctree.nodes.Project([C.CFile(func_name, [include, func_def])]).codegen()
        fn = module.get_callable(func_name, type_sig)

        _args = [self.buffers[arg.arg] for arg in args]
        return fn, _args

    def _insert_cast(self, body, shape, name):
        shape_str = "".join("[{}]".format(d) for d in shape)

        body.insert(0, StringTemplate(
            "float (* __restrict $arg_name)$shape = (float (*)$cast) _$arg_name;",
            {
                "arg_name": C.SymbolRef(name), 
                "shape": C.SymbolRef(shape_str),
                "cast": C.SymbolRef(shape_str)
            }))

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
