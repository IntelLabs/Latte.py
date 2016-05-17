import numpy as np
import numbers
import itertools
import ast
from .ensemble import Ensemble, DataEnsemble, ActivationEnsemble
import latte.util as util
import astor
from itertools import product
from .util import sgemm
import ctree
from ctree.transformations import PyBasicConversions
import ctree.c.nodes as C
from ctree.templates.nodes import StringTemplate, FileTemplate
import ctypes
import latte.transformers as transformers
import os
# import logging
# logging.basicConfig(level=20)
import multiprocessing
import inspect
from latte.mapping import Mapping, one_to_one
from latte.connection import Connection
from latte.task import Task

num_threads = int(os.getenv("OMP_NUM_THREADS", multiprocessing.cpu_count()))
os.environ["OMP_NUM_THREADS"] = str(num_threads)

os.environ["KMP_AFFINITY"] = "compact"

SIMDWIDTH = 8
TILE_SIZE = SIMDWIDTH
UNROLL_FACTOR = 12

include = FileTemplate(os.path.dirname(os.path.abspath(__file__)) + "/templates/includes.tmpl.c")

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
        self.reshaped_buffers = {}

    def add_ensemble(self, ensemble):
        self.ensembles.append(ensemble)

    def init_ensemble(self, neurons):
        ens = Ensemble(neurons)
        self.ensembles.append(ens)
        return ens

    def init_activation_ensemble(self, neurons):
        ens = ActivationEnsemble(neurons)
        self.ensembles.append(ens)
        return ens

    def add_connections(self, source_ens, sink_ens, mapping, reshape=None):
        self.connections.append(Connection(source_ens, sink_ens, mapping, reshape))

    def add_one_to_one_connections(self, source_ens, sink_ens):
        self.connections.append(Connection(source_ens, sink_ens, one_to_one, None))

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

    def _initialize_inputs(self, ensemble, source_target, buffer_name):
        conn = self.connections_map[ensemble][0]
        source_name = conn.source.name
        buff = self.buffers[source_name + source_target]
        if conn.reshape is not None:
            buff = buff.reshape((self.batch_size, ) + conn.reshape)
        self.buffers[buffer_name] = buff

    def _initialize_numeric_field(self, ensemble, field):
        neuron = ensemble.neurons.flat[0]
        value = getattr(neuron, field)
        buff = util.empty(ensemble.shape, type(value))
        self.buffers[ensemble.name + field] = buff
        for index, neuron in ensemble:
            buff[index] = getattr(neuron, field)

    def _initialize_ndarray_field(self, ensemble, field):
        neuron = ensemble.neurons.flat[0]
        value = getattr(neuron, field)
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

        if field in neuron.batch_fields:
            shape.insert(0, self.batch_size)

        buff = util.empty(shape, value.dtype)
        self.buffers[ensemble.name + field] = buff
        self.buffer_dim_info[ensemble.name + field] = uniform_across_dim
        if field in neuron.batch_fields:
            # Never uniform across batch dimension
            self.buffer_dim_info[ensemble.name + field].insert(0, False)

        if "grad_" in field:
            self.buffers[ensemble.name + field] = util.zeros((num_threads, ) + buff.shape, np.float32)
        else:
            for index in itertools.product(*_iter):
                _index = []
                if field in neuron.batch_fields:
                    # skip batch dimension
                    for i in range(len(uniform_across_dim[1:])):
                        if not uniform_across_dim[i + 1]:
                            _index.append(index[i])
                    for i in range(self.batch_size):
                        buff[i][tuple(_index)] = getattr(ensemble.neurons[index], field)
                else:
                    for i in range(len(uniform_across_dim)):
                        if not uniform_across_dim[i]:
                            _index.append(index[i])
                    buff[tuple(_index)] = getattr(ensemble.neurons[index], field)

    def _initialize_value_grad_activation(self, ensemble):
        for field, target in [("value", "inputs"), ("grad", "grad_inputs")]:
            target_buf = self.buffers[ensemble.name + target]
            self.buffers[ensemble.name + field] = target_buf

    def _initialize_value_grad(self, ensemble):
        for field in ["value", "grad"]:
            _shape = (self.batch_size, ) + ensemble.shape
            self.buffers[ensemble.name + field] = util.zeros(_shape, np.float32)

    def _init_buffers(self, ensemble):
        neuron = ensemble.neurons.flat[0]
        for field in vars(neuron):
            buffer_name = ensemble.name + field
            if field in ["value", "grad"]:
                # `value` and `grad` are initialized in the second pass, after
                # `inputs` and `grad_inputs` have been initialized (for in
                # place operations where `value` == `inputs`)
                pass
            elif field in ["inputs", "grad_inputs"]:
                source_target = "value" if field == "inputs" else "grad"
                self._initialize_inputs(ensemble, source_target, buffer_name)
            else:
                value = getattr(neuron, field)
                if isinstance(value, numbers.Real):
                    self._initialize_numeric_field(ensemble, field)
                elif isinstance(value, np.ndarray):
                    self._initialize_ndarray_field(ensemble, field)
                else:
                    raise NotImplementedError(field)

        if isinstance(ensemble, ActivationEnsemble):
            self._initialize_value_grad_activation(ensemble)
        else:
            self._initialize_value_grad(ensemble)

    def compile(self):
        task_groups = {}
        self.connections_map = {ensemble: [] for ensemble in self.ensembles}
        for connection in self.connections:
            self.connections_map[connection.sink].append(connection)

        forward_body = []
        forward_casts = []
        forward_args = set()

        backward_body = []
        backward_casts = []
        backward_args = set()

        print("Initializing ensembles...")
        for ensemble in self.ensembles:
            print("    {} [shape={}]".format(ensemble.name, ensemble.shape))
            self._init_buffers(ensemble)
            if isinstance(ensemble, DataEnsemble):
                self.forward_tasks.append(
                    Task(ensemble.forward, [self.buffers[ensemble.name + "value"]]))
            else:
                neuron = ensemble.neurons.flat[0]

                casts, body, args = self._synthesize_ast(ensemble, neuron.forward, "forward")
                forward_args = forward_args.union(args)
                forward_casts += casts
                forward_body += body

                casts, body, args = self._synthesize_ast(ensemble, neuron.backward, "backward")
                backward_args = backward_args.union(args)
                backward_casts += casts
                backward_body += body

        for args, direction, body, casts, tasks in zip([forward_args, backward_args], 
                                                       ["forward", "backward"],
                                                       [forward_body, backward_body],
                                                       [forward_casts, backward_casts],
                                                       [self.forward_tasks, self.backward_tasks]):
            args = list(args)
            arg_bufs = [self.buffers[arg.arg] for arg in args]
            type_sig = [np.ctypeslib.ndpointer(buf.dtype, buf.ndim, buf.shape) for buf in arg_bufs]
            params = [C.SymbolRef("_" + arg.arg, typ()) for arg, typ in zip(args, type_sig)]

            type_sig = ctypes.CFUNCTYPE(None, *type_sig)

            c_file = C.CFile(direction + self._uniqueid(), [
                include, 
                C.FunctionDecl(None, C.SymbolRef(direction), params, casts + body)
            ], path=".compiled")

            c_file._ext = "cpp"

            c_file = transformers.simple_fusion(c_file)
            c_file = transformers.remove_repeated_declarations(c_file)
            module = ctree.nodes.Project([c_file]).codegen()
            fn = module.get_callable(direction, type_sig)
            tasks.append(Task(fn, arg_bufs))

    unique_id = -1
    def _uniqueid(self):
        Net.unique_id += 1
        return str(self.unique_id)

    def _gen_untiled_neuron_index_1(self):
        """
        Generates the ast for the following expression:
        _neuron_index_1 = _neuron_index_1_outer * SIMDWIDTH + _neuron_index_1_inner

        This is used as an argument to the mapping function which expects an
        untiled _neuron_index_1
        """
        base_str = "_neuron_index_1 = _neuron_index_1_outer * {SIMDWIDTH} + _neuron_index_1_inner"
        return ast.parse(base_str.format(SIMDWIDTH=SIMDWIDTH)).body[0]

    def _parallelize_loops(self, func_def, ndim):
        for loop in func_def.defn:
            if ndim > 1:
                # count = 1
                # curr_loop = loop
                # while len(curr_loop.body) == 1 and isinstance(curr_loop.body[0], C.For):
                #     count += 1
                #     curr_loop = curr_loop.body[0]
                # loop.pragma = "omp parallel for collapse({})".format(min(count, 4))
                loop.pragma = "omp parallel for collapse(2)"
            else:
                loop.pragma = "omp parallel for collapse(2)"

    def _unroll(self, func_def, ensemble, unroll_target_loop_var):
        if unroll_target_loop_var == "_neuron_index_1_inner":
            unroll_factor = SIMDWIDTH
        else:
            unroll_factor = UNROLL_FACTOR
            unroll_dim = ensemble.shape[-1]
            if ensemble.ndim == 1:
                unroll_dim //= SIMDWIDTH
            while unroll_factor > unroll_dim or unroll_dim % unroll_factor != 0 :
                unroll_factor -= 1
                if unroll_factor == 0:
                    break
        if unroll_factor > 1:
            transformers.unroll_inner_neuron_loop(func_def, unroll_target_loop_var, unroll_factor)
            # func_def = transformers.promote_single_use_registers(func_def)
            # func_def = transformers.interleave_loads(func_def)

    def _reshape_buffer(self, args, ensemble, tiled_buffers):
        for arg in args:
            name = arg.arg
            buf = self.buffers[name]
            buf_shape = list(buf.shape)
            if buf.ctypes._data not in self.reshaped_buffers:
                if "grad_" in name and "grad_inputs" not in name or \
                        name.replace(ensemble.name, "") in ensemble.batch_fields or \
                        "inputs" in name:
                    buf_shape[1] //= SIMDWIDTH
                    buf_shape.append(SIMDWIDTH)
                elif "inputs" not in name:
                    buf_shape[0] //= SIMDWIDTH
                    buf_shape.append(SIMDWIDTH)
                if name in tiled_buffers and not name.endswith("inputs"):
                    dim = len(buf_shape) - tiled_buffers[name] - 1
                    buf_shape[dim] //= SIMDWIDTH
                    buf_shape.append(SIMDWIDTH)

                self.reshaped_buffers[buf.ctypes._data] = buf_shape
                self.buffers[name] = buf.reshape(buf_shape)
            elif "inputs" in name and self.connections_map[ensemble][0].reshape is not None:
                buf_shape = list((self.batch_size, ) + self.connections_map[ensemble][0].reshape)
                buf_shape[1] //= SIMDWIDTH
                buf_shape.append(SIMDWIDTH)
                self.buffers[name] = buf.reshape(buf_shape)
            else:
                buf_shape = self.reshaped_buffers[buf.ctypes._data]
                self.buffers[name] = buf.reshape(buf_shape)

    def _gen_transposes(self, body, transposed_buffers):
        for buffer_name, trans_dim in transposed_buffers.items():
            curr_body = []
            shape = self.buffers[buffer_name].shape

            node = util.gen_for("x0", 0, shape[0], curr_body)

            parfor_len = 2 if len(shape) - trans_dim - 1 > 1 else 1
            node.pragma = "omp parallel for collapse({})".format(parfor_len)

            idx = [C.SymbolRef("x0")]
            for i, d in enumerate(shape[1:-trans_dim-1]):
                i += 1  # offset range
                loopvar = "x{}".format(i)
                next_body = []
                curr_body.append(util.gen_for(loopvar, 0, d, next_body))
                idx.append(C.SymbolRef(loopvar))
                curr_body = next_body

            idx += [C.Constant(0), C.Constant(0)]

            if "grad_" in buffer_name:
                curr_body.append(C.FunctionCall(C.SymbolRef("transpose<SIMDWIDTH,SIMDWIDTH>"), 
                    [C.Ref(util.gen_index_expr(C.SymbolRef(buffer_name + "_transposed"), idx)),
                     C.Ref(util.gen_index_expr(C.SymbolRef(buffer_name), idx))]))
                body.append(node)
            else:
                curr_body.append(C.FunctionCall(C.SymbolRef("transpose<SIMDWIDTH,SIMDWIDTH>"), 
                    [C.Ref(util.gen_index_expr(C.SymbolRef(buffer_name), idx)), 
                     C.Ref(util.gen_index_expr(C.SymbolRef(buffer_name + "_transposed"), idx))]))
                body.insert(0, node)

    def _synthesize_ast(self, ensemble, fn, direction):
        # get_ast returns a ast.Module, grab the first item for the function
        fn_def = util.get_ast(fn).body[0]

        # transform domain constructs
        transformer = transformers.NeuronTransformer(ensemble,
                self.connections_map[ensemble], self.buffer_dim_info)
        fn_def = transformer.visit(fn_def)

        # Reverse iteration space for row major indexing
        loop_vars = ["_neuron_index_{}".format(i) for i in range(ensemble.ndim + 1)][::-1]
        loop_vars[-2] += "_outer"
        loop_vars.insert(0, "_neuron_index_1_inner")
        shape = list(ensemble.shape)
        shape[0] //= SIMDWIDTH
        shape.append(SIMDWIDTH)
        loop_ranges = ([self.batch_size] + [d for d in shape])[::-1]

        body = fn_def.body

        # Each statement in body is put into a separate nest (distributed loops)
        # This allows statements to be pattern matched independently
        # Fusion will merge the loops eventually (unless they are pattern matched)
        nests = [util.gen_loop_nest([s], loop_vars, loop_ranges) for s in body]


        args = [ast.arg(arg, None) for arg in transformer.seen_vars]

        # This placeholder function is used to wrap the body of statements for
        # convenience, after transformations have completed, the function body
        # is returned and the function node discarded
        func_def = ast.FunctionDef('func',
                ast.arguments(args, None, [], [], None, []), nests,
                [], None)

        func_def = transformers.pattern_match_gemm(func_def)
        func_def = transformers.simple_fusion(func_def)
        for loop in func_def.body:
            loop = loop.body[0]
            for dim in range(ensemble.ndim):
                loop = loop.body[0]

            mapping = self.connections_map[ensemble][0].mapping
            mapping_func = mapping.ast

            body = [self._gen_untiled_neuron_index_1()]

            for dim in range(1, ensemble.ndim):
                offset = mapping.get_offset(dim)
                input_offset = "_input_offset_{}".format(dim + 1)

                # If the offset is not a constant, we need to collect
                # dependent statements
                if not isinstance(offset, ast.Num):
                    body += util.get_dependent_statements(mapping_func.body[:-1], offset.id)

                # Store the offset value
                body.append(ast.Assign([ast.Name(input_offset, ast.Store())], offset))

            # Compute the tiled offsets from the result of the mapping function
            # For a one_to_one connection it is just the current neuron index
            # else its the remainder and floor division of the mapping result
            if mapping.is_one_to_one():
                body.append(ast.Assign([ast.Name("_input_offset_1", ast.Store())], 
                                       ast.Name("_neuron_index_1_outer", ast.Load())))
                body.append(ast.Assign([ast.Name("_input_offset_1_inner", ast.Store())], 
                                       ast.Name("_neuron_index_1_inner", ast.Load())))
            else:
                offset = mapping.get_offset(0)
                body.append(ast.Assign([ast.Name("_input_offset_1", ast.Store())], 
                                       ast.BinOp(offset, ast.Div(), ast.Num(SIMDWIDTH))))
                body.append(ast.Assign([ast.Name("_input_offset_1_inner", ast.Store())], 
                                       ast.BinOp(offset, ast.Mod(), ast.Num(SIMDWIDTH))))

            # Prepend to body
            loop.body = body + loop.body

        # convert [x, y, z] exprs into [x][y][z]
        func_def = transformers.convert_tuple_subscripts(func_def)
        # convert semantic (domain) ast nodes introduced by the neuron
        # transformer
        func_def, tiled_buffers = transformers.convert_enumerate_ranges(func_def, direction)

        func_def = transformers.convert_sgemm_calls(func_def)
        func_def = PyBasicConversions().visit(func_def)

        # convert loopvars from long to int. we do this as PyBasicConversion
        # defaults to using longs for range based for loops
        for loop in func_def.defn:
            loop.init.left.type = ctypes.c_int()
            for dim in range(ensemble.ndim + 1):
                loop = loop.body[0]
                loop.init.left.type = ctypes.c_int()

        # Seed the argument types as pointers for type inference
        for arg in func_def.params:
            buf = self.buffers[arg.name]
            arg.type = np.ctypeslib.ndpointer(buf.dtype, buf.ndim, buf.shape)()

        # Basic type inference and constant propogation
        func_def = transformers.BasicTypeInference().visit(func_def)
        func_def = transformers.SimpleConstantPropogation().visit(func_def)
        # print(func_def)
        # exit()

        vectorized_buffers = {key: [(value, TILE_SIZE)] for key, value in tiled_buffers.items()}
        # vectorized_buffers[ensemble.name+"inputs"] = [(2, SIMDWIDTH)]
        # vectorized_buffers[ensemble.name+"grad_inputs"] = [(2, SIMDWIDTH)]

        candidate = transformers.get_loop_to_vectorize(func_def)

        if candidate == "_neuron_index_1_inner":
            if ensemble.ndim > 1:
                unroll_target_loop_var = "_neuron_index_{}".format(ensemble.ndim)
            else:
                unroll_target_loop_var = "_neuron_index_1_outer".format(ensemble.ndim)
        else:
            unroll_target_loop_var = "_neuron_index_1_inner"
        # func_def = transformers.register_promote_value_refs(func_def, ensemble, direction, self.batch_size, target_loop_var)
        if candidate is not None:
            func_def, transposed_buffers = transformers.vectorize_loop(func_def, candidate)
            # func_def = transformers.push_inner_loop_down(func_def)
            # func_def = transformers.interchange_inner_loop(func_def)
            # func_def, _ = transformers.tile_outer_loop(func_def, ensemble.ndim)
            # func_def = transformers.interchange_tiled_loops(func_def)
            func_def = transformers.register_promote_vector_loads_stores(func_def)
            func_def = transformers.lift_invariant_load_stores(func_def)
            func_def = transformers.lift_loads(func_def)
            func_def = transformers.fma_replace(func_def)
            # func_def = transformers.move_inner_index(func_def)
            # func_def = transformers.unroll_constant_loops(func_def)

        self._parallelize_loops(func_def, ensemble.ndim)

        unroll = True
        if candidate is not None and unroll:
            self._unroll(func_def, ensemble, unroll_target_loop_var)
        else:
            func_def = transformers.insert_pragma_simd(func_def)

        type_sig = []
        casts = []
        self._reshape_buffer(args, ensemble, tiled_buffers)

        for arg in args:
            name = arg.arg
            buf = self.buffers[name]
            # casts.insert(0, StringTemplate("__assume_aligned({}, 64);\n".format(name)))
            self._insert_cast(casts, buf.shape[1:], name, buf.dtype)
            # casts.insert(0, StringTemplate("__assume_aligned(_{}, 64);\n".format(name)))

        if candidate is not None:
            self._gen_transposes(func_def.defn, transposed_buffers)

            for buffer_name, trans_dim in transposed_buffers.items():
                curr_body = []
                shape = self.buffers[buffer_name].shape
                shape_str = "".join("[{}]".format(d) for d in shape)

                args.append(ast.arg(buffer_name + "_transposed", None))
                self.buffers[buffer_name + "_transposed"] = util.zeros(shape, np.float32)
                self._insert_cast(casts, shape[1:], buffer_name + "_transposed", self.buffers[buffer_name + "_transposed"].dtype)

        return casts, func_def.defn, args

    def _insert_cast(self, body, shape, name, dtype):
        shape_str = "".join("[{}]".format(d) for d in shape)

        body.insert(0, StringTemplate(
            "$type (* __restrict $arg_name)$shape = ($type (*)$cast) _$arg_name;",
            {
                "arg_name": C.SymbolRef(name), 
                "shape": C.SymbolRef(shape_str),
                "cast": C.SymbolRef(shape_str),
                "type": C.SymbolRef(ctree.types.codegen_type(ctree.types.get_c_type_from_numpy_dtype(dtype)()))
            }))

    def forward(self):
        for task in self.forward_tasks:
            task()

    def backward(self):
        for task in self.backward_tasks:
            task()
