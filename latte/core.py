import numpy as np
import numbers
import itertools
import ast
from .ensemble import Ensemble, DataEnsemble, ActivationEnsemble, LossEnsemble, AccuracyEnsemble, EnsembleGroup
import latte.util as util
import astor
from itertools import product
# from .util import sgemm
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
import latte.transformers.vectorize as vectorizer
import latte.transformers.code_motion as code_motion
import latte.transformers.unroll as unroller
import latte.analysis as analyzer
import latte.optimizations as optimizer

# num_threads = int(os.getenv("OMP_NUM_THREADS", multiprocessing.cpu_count()))
# os.environ["OMP_NUM_THREADS"] = str(num_threads)
# 
os.environ["KMP_AFFINITY"] = "compact,granularity=fine,0,0"

latte_vec_config = os.getenv("LATTE_VEC_CONFIG", "AVX-2")
print("Latte running with vector instruction set {}".format(latte_vec_config))
vec_configs = {
    "AVX": 8,
    "AVX-2": 8,
    "AVX-512": 16
}
try:
    SIMDWIDTH = vec_configs[latte_vec_config]
except KeyError:
    raise Exception("ERROR: Invalid LATTE_VEC_CONFIG value = {}.  Supported values are {} ".format(latte_vec_config, vec_configs.keys()))

forward_unroll_factor = 8
backward_unroll_factor = 4
transpose_path = {
    "AVX": "/templates/transpose_256.tmp.c",
    "AVX-2": "/templates/transpose_256.tmp.c",
    "AVX-512": "/templates/transpose_512.tmp.c"
}[latte_vec_config]
package_path = os.path.dirname(os.path.abspath(__file__))

transpose = FileTemplate(package_path + transpose_path)

include = FileTemplate(package_path + "/templates/includes.tmpl.c",
        {"LATTE_PACKAGE_PATH": StringTemplate(package_path),
        "TRANSPOSE": transpose,
        "SIMDWIDTH": C.Constant(SIMDWIDTH)
        })

def compute_tiled_shape(buf_shape, field, ensemble):
    for dim, factor in ensemble.tiling_info[field]:
        if field in ensemble.batch_fields:
            dim += 1
        # elif "grad_" in field and field != "grad_inputs":
        elif field in ensemble.private_info:
            dim += 1  # offset for omp_get_thread_num()
        assert buf_shape[dim] % factor == 0, "Invalid tiling factor"
        buf_shape[dim] //= factor
        buf_shape.append(factor)
    return buf_shape

def gen_gen_clamp(_max):
    return lambda index: C.FunctionCall(C.SymbolRef("MIN"), 
            [C.FunctionCall(C.SymbolRef("MAX"), 
                [index,
                 C.Constant(0)]), 
             C.Constant(_max)])

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
        self.nowait = False
        self.force_backward = False
        self.loss = 0.0
        self.accuracy = 0.0
        self.value_buffers = []
        self.grad_buffers = []

    def add_ensemble(self, ensemble):
        self.ensembles.append(ensemble)

    def init_ensemble(self, neurons):
        ens = Ensemble(neurons)
        self.ensembles.append(ens)
        return ens

    def init_activation_ensemble(self, neurons, source_ens):
        ens = ActivationEnsemble(neurons, source_ens)
        self.ensembles.append(ens)
        self.add_one_to_one_connections(source_ens, ens)
        return ens

    def add_connections(self, source_ens, sink_ens, mapping, reshape=None, clamp=False):
        if isinstance(source_ens, EnsembleGroup):
            source_ens = source_ens.ensembles[-1]
        self.connections.append(Connection(source_ens, sink_ens, mapping, reshape, clamp))

    def add_loss_connection(self, source_ens, sink_ens):
        self.connections.append(Connection(source_ens, sink_ens, one_to_one, None))

    def add_one_to_one_connections(self, source_ens, sink_ens):
        if isinstance(source_ens, EnsembleGroup):
            source_ens = source_ens.ensembles[-1]
        self.connections.append(Connection(source_ens, sink_ens, one_to_one, None))

    def clear_grad(self):
        for arr in self.grad_buffers:
            arr.fill(0.0)

    def clear_values(self):
        for arr in self.value_buffers:
            arr.fill(0.0)

    def _get_uniformity(self, ensemble, field):
        """
        Could be done symbolically?
        """
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
            assert False, "Deprecated"
            buff = buff.reshape((self.batch_size, ) + conn.reshape)
        self.buffers[buffer_name] = buff

    def _initialize_numeric_field(self, ensemble, field):
        try:
            neuron = ensemble.neurons.flat[0]
            value = getattr(neuron, field)
            if field in ensemble.batch_fields:
                buff = util.empty((self.batch_size, ) + ensemble.shape, type(value))
                self.buffers[ensemble.name + field] = buff
                for index, neuron in ensemble:
                    buff[(slice(None),) + index] = getattr(neuron, field)   
            else:
                buff = util.empty(ensemble.shape, type(value))
                self.buffers[ensemble.name + field] = buff
                for index, neuron in ensemble:
                    buff[index] = getattr(neuron, field)
        except Exception as e:
            print("error initializing numeric field for " + str(type(ensemble.neurons.flat[0])) + ", field: " + str(field))
            raise e

    def _initialize_ndarray_field(self, ensemble, field):
        neuron = ensemble.neurons.flat[0]
        value = getattr(neuron, field)
        _shape = ensemble.shape

        uniform_across_dim = self._get_uniformity(ensemble, field)
        shape = []
        _iter = []
        for i in range(len(_shape)):
            if not uniform_across_dim[i]:
                _iter.append(_shape[i])
                shape.append(_shape[i])
            else:
                _iter.append(1)
        shape += value.shape

        if field in neuron.batch_fields or field in ensemble.private_info:
            shape.insert(0, self.batch_size)
            # Never uniform across batch dimension
            uniform_across_dim.insert(0, False)

        buff = util.zeros(shape, value.dtype)
        self.buffers[ensemble.name + field] = buff
        self.buffer_dim_info[ensemble.name + field] = uniform_across_dim

        if field not in neuron.zero_init_fields:
            if field in neuron.batch_fields or field in ensemble.private_info:
                # Skip batch dimension
                uniform_across_dim = uniform_across_dim[1:]
            for index in np.ndindex(*_iter):
                attr = getattr(ensemble.neurons[index], field)
                _index = ()
                for i in range(len(uniform_across_dim)):
                    if not uniform_across_dim[i]:
                        _index += (index[i], )
                if field in neuron.batch_fields or field in ensemble.private_info:
                    for i in range(self.batch_size):
                        buff[i][_index] = attr
                else:
                    buff[_index] = attr

    def _initialize_value_grad_activation(self, ensemble):
        for field, target in [("value", "inputs"), ("grad", "grad_inputs")]:
            target_buf = self.buffers[ensemble.name + target]
            self.buffers[ensemble.name + field] = target_buf

    def _initialize_value_grad(self, ensemble):
        for field in ["value", "grad"]:
            # p = (bottom_pad, top_pad)
            # d = size of a dimension
            shape = (self.batch_size, ) + \
                   tuple(p[0] + p[1] + d for p, d in zip(ensemble.pad, ensemble.shape))
            self.buffers[ensemble.name + field] = util.zeros(shape, np.float32)

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
                    ensemble.scalar_fields.append(field)
                    self._initialize_numeric_field(ensemble, field)
                elif isinstance(value, np.ndarray):
                    self._initialize_ndarray_field(ensemble, field)
                else:
                    raise NotImplementedError(field)

        if isinstance(ensemble, ActivationEnsemble):
            self._initialize_value_grad_activation(ensemble)
        else:
            self._initialize_value_grad(ensemble)

        for field in vars(neuron):
            buffer_name = ensemble.name + field
            ensemble.set_buffer(field, self.buffers[buffer_name])

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

        in_place_buffer_map = {}

        print("Initializing ensembles and synthesizing functions...")
        for ensemble in self.ensembles:
            print("    {} [shape={}]".format(ensemble.name, ensemble.shape))
            if isinstance(ensemble, (LossEnsemble, AccuracyEnsemble)):
                raise NotImplementedError("Ensemble type {} no longer supported".format(type(ensemble)))
            self._init_buffers(ensemble)
            if isinstance(ensemble, DataEnsemble):
                self.forward_tasks.append(
                    Task(ensemble.forward, [self.buffers[ensemble.name + "value"]]))
                for field in ["value", "grad"]:
                    buffer = self.buffers[ensemble.name + field]
                    if field in ensemble.tiling_info:
                        buf_shape = compute_tiled_shape(list(buffer.shape), field, ensemble)
                        buffer = buffer.reshape(buf_shape)
                        self.buffers[ensemble.name + field] = buffer
            else:
                neuron = ensemble.neurons.flat[0]

                casts, body, args = self._synthesize_ast(ensemble, neuron, "forward")
                forward_args = forward_args.union(args)
                forward_casts += casts
                forward_body += body

                casts, body, args = self._synthesize_ast(ensemble, neuron, "backward")
                backward_args = backward_args.union(args)
                backward_casts += casts
                backward_body = body + backward_body

            if isinstance(ensemble, ActivationEnsemble):
                source = self.connections_map[ensemble][0].source
                in_place_buffer_map[source.name + "value"] = [ensemble.name + "inputs"]

        print("Compiling functions...")
        for args, direction, body, casts, tasks, in zip([forward_args, backward_args], 
                                                        ["forward", "backward"],
                                                        [forward_body, backward_body],
                                                        [forward_casts, backward_casts],
                                                        [self.forward_tasks, self.backward_tasks],
                                                        ):
            args = list(args)
            arg_bufs = [self.buffers[arg.arg] for arg in args]
            type_sig = [np.ctypeslib.ndpointer(buf.dtype, buf.ndim, buf.shape) for buf in arg_bufs]
            params   = [C.SymbolRef("_" + arg.arg, typ()) for arg, typ in zip(args, type_sig)]

            type_sig = ctypes.CFUNCTYPE(None, *type_sig)

            _id = self._uniqueid()
            c_file = C.CFile(direction + _id, [
                include, 
                C.FunctionDecl(None, C.SymbolRef(direction + _id), params, body)
            ], path=".compiled")

            c_file._ext = "cpp"

            # print(c_file)
            c_file = transformers.simple_fusion(c_file)

            new_body = []
            incr = -1
            for stmt in c_file.body[1].defn:
                incr += 1
                if isinstance(stmt, C.For):
                    if hasattr(stmt, 'pre_trans') and stmt.pre_trans is not None:
                        new_body.extend(stmt.pre_trans)
                    # loopvar1 = C.SymbolRef(stmt.init.left.name)
                    # looplen1 = stmt.test.right
                    # body = stmt.body
                    # new_body.append(self._gen_graph_nodes_from_loop(stmt, incr))
                    stmt = transformers.convert_parallel_loops(stmt)
                    new_body.append(stmt)
                    # if incr > 0:
                    #     new_body.append(self._gen_graph_edges_for_loop(stmt, incr-1, incr))
                    # else:
                    #     execute_stmt = self._gen_execute_for_loop(stmt, incr)
                    if hasattr(stmt, 'reduce_vars') and len(stmt.reduce_vars) > 0:
                        for var in stmt.reduce_vars:
                            size = np.prod(self.buffers[var].shape[1:])
                            new_body.append(self._gen_reduce_for_loop(stmt, var, size, incr))
                else:
                    new_body.append(stmt)
            c_file.body[1].defn = new_body # + [execute_stmt]
            c_file.body[1].defn = casts + c_file.body[1].defn
            c_file = transformers.promote_in_place_load_stores(c_file, in_place_buffer_map)
            c_file.body[1].defn.insert(0, StringTemplate("static FlowGraph graph;"))
            c_file.body[1].defn.append(StringTemplate("graph.wait_for_all();"))
            # c_file = transformers.remove_repeated_declarations(c_file)
            module = util.mpi_compile(ctree.nodes.Project([c_file]))
            # get_callable(functions_handle, type_signature)
            fn = module.get_callable(direction + _id, type_sig)
            tasks.append(Task(fn, arg_bufs))

        self._collect_value_grad_bufs()
        print("Done")

    def _collect_value_grad_bufs(self):
        for key, buf in self.buffers.items():
            if "value" in key:
                self.value_buffers.append(buf)
            if "grad" in key:
                self.grad_buffers.append(buf)

    unique_id = -1
    def _uniqueid(self):
        Net.unique_id += 1
        return str(self.unique_id)

    def _parallelize_loops(self, func_def, loopvars):
        class Parallelizer(ast.NodeTransformer):
            def __init__(self, loopvars):
                self.loopvars = loopvars

            def visit_For(self, node):
                node.body = [self.visit(s) for s in node.body]
                if node.init.left.name in self.loopvars:
                    node.parallel = True
                    # return StringTemplate("""
                    # parallel_for(0,$looplen / $loopincr,
                    #   [=](int low, int high) {
                    #     for (int tmp_$loopvar = low; tmp_$loopvar < high; tmp_$loopvar++) {
                    #       int $loopvar = tmp_$loopvar * $loopincr;
                    #       $body;
                    #     }
                    #   });
                    # """, {
                    #     "looplen": node.test.right,
                    #     "loopvar": node.test.left,
                    #     "loopincr": node.incr.value,
                    #     "body": node.body
                    # })

                return node
        return Parallelizer(loopvars).visit(func_def)
        # for loop in func_def.defn:
        #     if ndim > 1:
        #         loop.pragma = "omp for collapse(2)"
        #     else:
        #         loop.pragma = "omp for collapse(2)"

        #     if self.nowait:
        #         loop.pragma += " nowait"

    def _reshape_buffer(self, args, ensemble, tiled_buffers):
        for arg in args:
            name = arg.arg
            buf = self.buffers[name]
            buf_shape = list(buf.shape)
            field = name.replace(ensemble.name, "")
            if isinstance(ensemble, ActivationEnsemble) and \
                    field in ["value", "grad", "inputs", "grad_inputs"]:
                # ActivationEnsembles execute in place, if a field is tiled
                # then the buffer should have already been tiled when handling
                # the input ensemble
                continue
            if field in ensemble.tiling_info and field not in ["inputs", "grad_inputs"]:
                if name not in self.reshaped_buffers:
                    buf_shape = compute_tiled_shape(buf_shape, field, ensemble)
                    self.reshaped_buffers[name] = buf_shape
                    self.buffers[name] = buf.reshape(buf_shape)

    def _gen_graph_nodes_from_loop(self, loop, id):
        loopvar1 = C.SymbolRef(loop.init.left.name)
        looplen1 = loop.test.right
        loopincr = loop.incr.value
        body = loop.body
        return StringTemplate("""
        std::vector<ContinueNode *> $node_list;
        for (int __z = 0; __z < $looplen1 / $loopincr; __z++) {
          ContinueNode *node = new ContinueNode(&graph, [=]() {
            int $loopvar1 = __z * $loopincr;
            $body;
          });
          for (int i = 0; i < $loopincr; i++)
            $node_list.push_back(node);
        }
        """, {'loopvar1': loopvar1, 'looplen1': looplen1, 'loopincr': loopincr,
              'body': body, 
              'node_list': C.SymbolRef("node_list_" + str(id))
        })

    def _gen_graph_edges_for_loop(self, loop, source_id, sink_id):
        loopvar1 = C.SymbolRef(loop.init.left.name)
        looplen1 = loop.test.right
        loopincr = loop.incr.value.value
        return StringTemplate("""
          for (int i = 0; i < $looplen1; ++i) {
            make_edge($prev_node_list[i], $node_list[i]);
          }
        """, {
            'looplen1': C.Constant(looplen1.value), 'loopincr': C.Constant(loopincr),
            'node_list': C.SymbolRef("node_list_" + str(sink_id)),
            'prev_node_list': C.SymbolRef("node_list_" + str(source_id)),
        })

    def _gen_execute_for_loop(self, loop, id):
        looplen1 = loop.test.right
        loopincr = loop.incr.value.value
        return StringTemplate("""
          for (int i = 0; i < $looplen1; i+=$loopincr) {
            $node_list[i]->execute();
          }
        """, {
            'looplen1': C.Constant(looplen1.value), 'loopincr': C.Constant(loopincr), 
            'node_list': C.SymbolRef("node_list_" + str(id)),
        })

    def _gen_reduce_for_loop(self, loop, var, size, id):
        looplen1 = loop.test.right
        loopincr = loop.incr.value.value
        return StringTemplate("""
            //{
            //  ContinueNode *$reduce_node = new ContinueNode(&graph, [=]() {
              parallel_for(0,$size,
                [=](int low, int high) {
                  #pragma simd
                  for (int x = low; x < high; ++x) {
                    float sum = _$arr[x];
                    #pragma unroll($batch_size - 1)
                    for (int i = 1; i < $batch_size; ++ i) {
                      sum += _$arr[i * $size + x];
                    }
                    _$arr[x] = sum;
                  }
                });
            //  });
            //  for (int i = 0; i < $looplen1; i+=$loopincr) {
            //    make_edge($node_list[i], $reduce_node);
            //  }
            //};
            """, {'size': C.Constant(size),
                  'batch_size': C.Constant(self.batch_size),
                  'arr': C.SymbolRef(var),
                  'node_list': C.SymbolRef("node_list_" + str(id)),
                  'reduce_node': C.SymbolRef("reduce_node_" + str(id)),
                  'looplen1': C.Constant(looplen1.value),  'loopincr': C.Constant(loopincr)
                  })

    def _gen_transposes(self, transposed_buffers):
        pre_trans = []
        post_trans = []
        for buffer_name, trans_dim in transposed_buffers.items():
            curr_body = []
            shape = self.buffers[buffer_name].shape

            # node = util.gen_for("x0", 0, shape[0], curr_body)
            node = StringTemplate("""
            parallel_for(0, $len,
              [=](int low, int high) {
                for (int x0 = low; x0 < high; x0++) {
                  $body
                } 
              }
            );
            """, {'body': curr_body, 'len': C.Constant(shape[0])})

            parfor_len = 2 if len(shape) - trans_dim - 1 > 1 else 1
            # node.pragma = "omp for collapse({})".format(parfor_len)

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
                assert False
                curr_body.append(C.FunctionCall(C.SymbolRef("transpose<SIMDWIDTH,SIMDWIDTH>"), 
                    [C.Ref(util.gen_index_expr(C.SymbolRef(buffer_name + "_transposed"), idx)),
                     C.Ref(util.gen_index_expr(C.SymbolRef(buffer_name), idx))]))
                post_trans.append(node)
            else:
                shape_str = ""
                for d in shape:
                    shape_str += "[{}]".format(d)
                    # pre_trans.append(
                    #         StringTemplate("float {}{} __attribute__((aligned(64)));".format(buffer_name + "_transposed", shape_str)))
                curr_body.append(C.FunctionCall(C.SymbolRef("transpose<SIMDWIDTH,SIMDWIDTH>"), 
                    [C.Ref(util.gen_index_expr(C.SymbolRef(buffer_name), idx)), 
                     C.Ref(util.gen_index_expr(C.SymbolRef(buffer_name + "_transposed"), idx))]))
                pre_trans.append(node)
        return pre_trans, post_trans

    def _synthesize_ast(self, ensemble, neuron, direction):
        # get_ast returns an ast.Module, grab the first item for the function
        if direction == "forward":
            fn_def = util.get_ast(neuron.forward).body[0]
        else:
            # Don't backpropogate to DataEnsemble
            if not isinstance(self.connections_map[ensemble][0].source, DataEnsemble) or \
                    self.force_backward:
                fn_def = util.get_ast(neuron.backward).body[0]
                if hasattr(neuron, 'update_internal'):
                    fn_def.body += util.get_ast(neuron.update_internal).body[0].body
                    fn_def = transformers.simple_fusion(fn_def)
            elif hasattr(neuron, 'update_internal'):
                fn_def = util.get_ast(neuron.update_internal).body[0]
            else:
                # No-op
                return [], [], set()

        # transform domain constructs
        transformer = transformers.NeuronTransformer(ensemble,
                self.connections_map[ensemble], self.buffer_dim_info)
        fn_def = transformer.visit(fn_def)

        # Grab seen variables
        args = [ast.arg(arg, None) for arg in transformer.seen_vars]

        loop_vars = ["_neuron_index_{}".format(i) for i in range(ensemble.ndim + 1)]
        shape = list(ensemble.shape)

        # rename to reorder storage
        # TODO: ASSUMING VALUE AND GRAD ARE TILED THE SAME
        if "value" in ensemble.tiling_info:
            for dim, factor in ensemble.tiling_info["value"]:
                if shape[dim] % factor != 0:
                    raise Exception(("Invalid tiling factor of {} on dimension "
                        + "{} for {}'s value buffer (shape={})").format(factor,
                            dim, ensemble.name, shape))
                    
                shape[dim] //= factor
                shape.append(factor)
                loop_vars.append(loop_vars[dim + 1] + "_inner")
                loop_vars[dim + 1] += "_outer"

        # Reverse ([::-1]) iteration space for row major indexing
        loop_vars = loop_vars[::-1]
        loop_ranges = ([self.batch_size] + [d for d in shape])[::-1]

        body = fn_def.body
        body = [util.gen_loop_nest(body, loop_vars, loop_ranges)]

        # This placeholder function is used to wrap the body of statements for
        # convenience, after transformations have completed, the function body
        # is returned and this function node will be discarded
        func_def = ast.FunctionDef('func',
                ast.arguments(args, None, [], [], None, []), body,
                [], None)

        # TODO: MAKE THIS A FUNCTION
        for loop in func_def.body:
            for dim in range(len(loop_vars) - 1):
                loop = loop.body[0]

            mapping = self.connections_map[ensemble][0].mapping
            mapping_func = mapping.ast

            body = []

            for dim in range(0, ensemble.ndim):
                is_tiled_dim = "inputs" in ensemble.tiling_info and \
                    any(tiled_dim == dim 
                        for tiled_dim, _ in ensemble.tiling_info["inputs"])

                if is_tiled_dim:
                    for tiled_dim, factor in ensemble.tiling_info["inputs"]:
                        if tiled_dim == dim:
                            # factor is now the tiling factor for tiled_dim
                            break
                    mapping.set_arg(dim, ast.BinOp(
                            ast.BinOp(
                                ast.Name("_neuron_index_{}_outer".format(dim + 1), ast.Load()), 
                                ast.Mult(),
                                ast.Num(factor)),
                            ast.Add(),
                            ast.Name("_neuron_index_{}_inner".format(dim + 1), ast.Load())
                    ))
                else:
                    mapping.set_arg(dim, ast.Name("_neuron_index_{}".format(dim + 1), ast.Store()))
                offset = mapping.get_offset(dim)
                input_offset = "_input_offset_{}".format(dim + 1)

                # If the offset is not a constant, we need to collect
                # dependent statements
                if not isinstance(offset, ast.Num):
                    body += util.get_dependent_statements(mapping_func.body[:-1], offset)

                if is_tiled_dim:
                    outer_offset = ast.BinOp(offset, ast.Div(), ast.Num(factor))
                    body.append(ast.Assign([ast.Name(input_offset + "_outer", ast.Store())], outer_offset))
                    inner_offset = ast.BinOp(offset, ast.Mod(), ast.Num(factor))
                    body.append(ast.Assign([ast.Name(input_offset + "_inner", ast.Store())], inner_offset))
                else:
                    # Store the offset value
                    body.append(ast.Assign([ast.Name(input_offset, ast.Store())], offset))

            # Prepend to body
            loop.body = body + loop.body

        # convert [x, y, z] exprs into [x][y][z]
        func_def = transformers.convert_tuple_subscripts(func_def)
        # convert semantic (domain) ast nodes introduced by the neuron
        # transformer
        # FIXME: Deprecate tiled_buffers
        func_def, tiled_buffers = transformers.convert_enumerate_ranges(func_def, direction, ensemble)
        func_def = PyBasicConversions().visit(func_def)
        func_def = transformers.PatternMatchMath().visit(func_def)

        for loop in func_def.defn:
            # convert loopvars from long to int
            # we do this because PyBasicConversion defaults to using longs for
            # range based for loops
            loop.init.left.type = ctypes.c_int()
            for dim in range(len(loop_vars) - 1):
                loop = loop.body[0]
                loop.init.left.type = ctypes.c_int()
                input_shape = self.connections_map[ensemble][0].source.shape

                if dim == 0:
                    # Do not need to clamp batch dimension
                    continue

                input_offset = "_input_offset_{}".format(dim)
                if mapping.clamp:
                    if dim in ensemble.tiling_info:
                        gen_clamp = gen_gen_clamp(input_shape[dim - 1] // SIMDWIDTH - 1)
                        loop.body = [util.ClampInputIndex(input_offset, gen_clamp).visit(s) for s in loop.body]
                        gen_clamp = gen_gen_clamp(SIMDWIDTH - 1)
                        loop.body = [util.ClampInputIndex(input_offset + "_inner", gen_clamp).visit(s) for s in loop.body]
                    else:
                        gen_clamp = gen_gen_clamp(input_shape[dim - 1] - 1)
                        loop.body = [util.ClampInputIndex(input_offset, gen_clamp).visit(s) for s in loop.body]

        # Seed the argument types as pointers for type inference
        for arg in func_def.params:
            buf = self.buffers[arg.name]
            arg.type = np.ctypeslib.ndpointer(buf.dtype, buf.ndim, buf.shape)()

        # Basic type inference and constant propogation
        func_def = analyzer.type_infer(func_def)
        func_def = optimizer.propogate_constants(func_def)

        for loop_var1, loop_var2 in ensemble.loops_to_swap[direction]:
            loop1 = util.find_loop(func_def, loop_var1)
            loop2 = util.find_loop(func_def, loop_var2)
            # loop.init = (int x = 0)
            # loop.test = (x < 5)
            # loop.incr = (x += 1)
            loop1.init, loop2.init = loop2.init, loop1.init
            loop1.test, loop2.test = loop2.test, loop1.test
            loop1.incr, loop2.incr = loop2.incr, loop1.incr

        if direction in ensemble.vectorize_info:
            # RAJ hack here
            func_def, transposed_buffers = vectorizer.vectorize_loop(func_def, 
                    ensemble.vectorize_info[direction][0])
            # func_def = transformers.push_inner_loop_down(func_def)
            func_def = vectorizer.register_promote_vector_loads_stores(func_def)
            func_def = code_motion.lift_invariant_load_stores(func_def)
            func_def = vectorizer.fuse_multiply_adds(func_def)

        if direction in ensemble.simd_info:
            for loopvar in ensemble.simd_info[direction]:
                func_def = transformers.insert_pragma_simd(func_def, loopvar)

        if direction in ensemble.unroll_info:
            unroll_var, unroll_factor = ensemble.unroll_info[direction]
            unroller.unroll_loop(func_def, unroll_var, unroll_factor)

        self._parallelize_loops(func_def, ensemble.parallel_info[direction])

        type_sig = []
        casts = []
        # FIXME: Check if still needed
        self._reshape_buffer(args, ensemble, tiled_buffers)

        for arg in args:
            name = arg.arg
            buf = self.buffers[name]
            casts.insert(0, StringTemplate("__assume_aligned({}, 64);\n".format(name)))
            self._insert_cast(casts, buf.shape[1:], name, buf.dtype)

        if direction in ensemble.vectorize_info:
            pre_trans, post_trans = self._gen_transposes(transposed_buffers)

            for buffer_name, trans_dim in transposed_buffers.items():
                curr_body = []
                shape = self.buffers[buffer_name].shape
                shape_str = "".join("[{}]".format(d) for d in shape)

                args.append(ast.arg(buffer_name + "_transposed", None))
                self.buffers[buffer_name + "_transposed"] = util.zeros(shape, np.float32)
                self._insert_cast(casts, shape[1:], buffer_name + "_transposed", self.buffers[buffer_name + "_transposed"].dtype)
        else:
            pre_trans = []
            post_trans = []

        assert isinstance(func_def.defn[0], C.For)
        func_def.defn[0].pre_trans = pre_trans
        reduce_vars = []
        func_def.defn[0].reduce_vars = reduce_vars
        for arg in args:
            # if "grad_" in arg.arg and "inputs" not in arg.arg:
            if arg.arg.replace(ensemble.name, "") in ensemble.private_info:
                reduce_vars.append(arg.arg)
        return casts, func_def.defn, args

    def _insert_cast(self, body, shape, name, dtype):
        shape_str = "".join("[{}]".format(d) for d in shape)

        body.insert(0, StringTemplate(
            # "$type (* __restrict $arg_name)$shape = ($type (*)$cast) _$arg_name;",
            "$type (* $arg_name)$shape = ($type (*)$cast) _$arg_name;",
            {
                "arg_name": C.SymbolRef(name), 
                "shape": C.SymbolRef(shape_str),
                "cast": C.SymbolRef(shape_str),
                "type": C.SymbolRef(ctree.types.codegen_type(ctree.types.get_c_type_from_numpy_dtype(dtype)()))
            }))

    def test(self):
        for task in self.forward_tasks:
            task()

    def forward(self):
        # os.environ["KMP_AFFINITY"] = "compact,granularity=fine,0,0"
        for task in self.forward_tasks:
            task()

    def backward(self):
        # os.environ["KMP_AFFINITY"] = "scatter,granularity=fine,0,0"
        for task in self.backward_tasks:
            task()

