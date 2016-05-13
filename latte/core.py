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
from ctree.templates.nodes import StringTemplate
import ctypes
import latte.transformers as transformers
import os
# import logging
# logging.basicConfig(level=20)
import multiprocessing
import inspect

num_threads = int(os.getenv("OMP_NUM_THREADS", multiprocessing.cpu_count()))
os.environ["OMP_NUM_THREADS"] = str(num_threads)

os.environ["KMP_AFFINITY"] = "compact,0,0,granularity=fine"

SIMDWIDTH = 8
TILE_SIZE = SIMDWIDTH
UNROLL_FACTOR = 12

include = StringTemplate("""
#include <immintrin.h>
#include <mkl.h>
#include <stdio.h>
#include <cmath>
#include <omp.h>
#define SIMDWIDTH 8
#define TILE_SIZE 8
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define MAX(x, y) (((x) > (y)) ? (x) : (y))

template<int in_width, int out_width>
void transpose(float *in, float *out)
{
    __m256i r0, r1, r2, r3, r4, r5, r6, r7;
    __m256i t0, t1, t2, t3, t4, t5, t6, t7;
    if((in_width & 0x7 != 0)  || (out_width & 0x7 != 0)) {printf("Transpose8x8: Invalid in or out width\\n"); return;}
    
    r0 = _mm256_load_si256((const __m256i *)(in + 0*in_width));
    r1 = _mm256_load_si256((const __m256i *)(in + 1*in_width));
    r2 = _mm256_load_si256((const __m256i *)(in + 2*in_width));
    r3 = _mm256_load_si256((const __m256i *)(in + 3*in_width));
    r4 = _mm256_load_si256((const __m256i *)(in + 4*in_width));
    r5 = _mm256_load_si256((const __m256i *)(in + 5*in_width));
    r6 = _mm256_load_si256((const __m256i *)(in + 6*in_width));
    r7 = _mm256_load_si256((const __m256i *)(in + 7*in_width));

    t0 = _mm256_unpacklo_epi32(r0,r1); 
    t1 = _mm256_unpackhi_epi32(r0,r1); 
    t2 = _mm256_unpacklo_epi32(r2,r3); 
    t3 = _mm256_unpackhi_epi32(r2,r3); 
    t4 = _mm256_unpacklo_epi32(r4,r5); 
    t5 = _mm256_unpackhi_epi32(r4,r5); 
    t6 = _mm256_unpacklo_epi32(r6,r7); 
    t7 = _mm256_unpackhi_epi32(r6,r7); 

    r0 = _mm256_unpacklo_epi64(t0,t2); 
    r1 = _mm256_unpackhi_epi64(t0,t2); 
    r2 = _mm256_unpacklo_epi64(t1,t3); 
    r3 = _mm256_unpackhi_epi64(t1,t3); 
    r4 = _mm256_unpacklo_epi64(t4,t6); 
    r5 = _mm256_unpackhi_epi64(t4,t6); 
    r6 = _mm256_unpacklo_epi64(t5,t7); 
    r7 = _mm256_unpackhi_epi64(t5,t7); 

    t0 = _mm256_permute2f128_si256(r0, r4, 0x20); 
    t1 = _mm256_permute2f128_si256(r1, r5, 0x20); 
    t2 = _mm256_permute2f128_si256(r2, r6, 0x20); 
    t3 = _mm256_permute2f128_si256(r3, r7, 0x20); 
    t4 = _mm256_permute2f128_si256(r0, r4, 0x31); 
    t5 = _mm256_permute2f128_si256(r1, r5, 0x31); 
    t6 = _mm256_permute2f128_si256(r2, r6, 0x31); 
    t7 = _mm256_permute2f128_si256(r3, r7, 0x31); 

    _mm256_store_si256((__m256i *)(out + 0*out_width), t0);
    _mm256_store_si256((__m256i *)(out + 1*out_width), t1);
    _mm256_store_si256((__m256i *)(out + 2*out_width), t2);
    _mm256_store_si256((__m256i *)(out + 3*out_width), t3);
    _mm256_store_si256((__m256i *)(out + 4*out_width), t4);
    _mm256_store_si256((__m256i *)(out + 5*out_width), t5);
    _mm256_store_si256((__m256i *)(out + 6*out_width), t6);
    _mm256_store_si256((__m256i *)(out + 7*out_width), t7);
}

extern "C"
""")

class Mapping:
    def __init__(self, mapping_func):
        self.mapping_func = mapping_func
        ast = util.get_ast(mapping_func).body[0]

        closure_vars = inspect.getclosurevars(mapping_func)
        for var, value in closure_vars.nonlocals.items():
            ast = util.inline_variable(var, value, ast)
        self.ast = ast
        self.ndim = len(self.ast.args.args)
        self.shape = mapping_func(*[1 for _ in range(self.ndim)])

    def get_offset(self, dim):
        if self.mapping_func == one_to_one:
            # return ast.Name("_neuron_index_{}".format(dim + 1), ast.Load())
            return ast.Num(0)
        range_expr = self.ast.body[-1].value.elts[dim]
        if len(range_expr.args) == 2:
            return range_expr.args[0]
        elif len(range_expr.args) == 3:
            raise NotImplementedError()
        else:
            return ast.Num(0)


class Connection:
    def __init__(self, source_ens, sink_ens, mapping, reshape):
        self.source = source_ens
        self.sink = sink_ens
        self.mapping = Mapping(mapping)
        self.mapping_inserted = False
        self.reshape = reshape

class Task:
    def __init__(self, fn, args):
        self.fn = fn
        self.args = args

    def __call__(self):
        self.fn(*self.args)

def one_to_one(*args):
    return tuple(range(a,a+1) for a in args)

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

    def _init_buffers(self, ensemble):
        neuron = ensemble.neurons.flat[0]
        for field in vars(neuron):
            if field in ["value", "grad"]:
                pass
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
                                    buff[i, _index] = getattr(ensemble.neurons[index], field)
                            else:
                                for i in range(len(uniform_across_dim)):
                                    if not uniform_across_dim[i]:
                                        _index.append(index[i])
                                buff[_index] = getattr(ensemble.neurons[index], field)
                else:
                    raise NotImplementedError(field)

        for field in ["value", "grad"]:
            if isinstance(ensemble, ActivationEnsemble):
                target_map = {
                    "value": "inputs",
                    "grad": "grad_inputs"
                }
                target_buf = self.buffers[ensemble.name + target_map[field]]
                self.buffers[ensemble.name + field] = target_buf
            else:
                _shape = (self.batch_size, ) + ensemble.shape
                self.buffers[ensemble.name + field] = util.zeros(_shape, np.float32)

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

        for ensemble in self.ensembles:
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
            type_sig = []
            arg_bufs = []
            params = []
            for arg in args:
                buf = self.buffers[arg.arg]
                type_sig.append(np.ctypeslib.ndpointer(buf.dtype, buf.ndim, buf.shape))
                arg_bufs.append(buf)
                params.append(
                    C.SymbolRef("_" + arg.arg, 
                        np.ctypeslib.ndpointer(buf.dtype, buf.ndim, buf.shape)()))

            type_sig = ctypes.CFUNCTYPE(None, *type_sig)
            c_file = C.CFile(direction + self._uniqueid(), [
                include, 
                C.FunctionDecl(None, C.SymbolRef(direction), params, casts + body)
            ], path=".compiled")
            c_file._ext = "cpp"
            # c_file = transformers.simple_fusion(c_file)
            module = ctree.nodes.Project([c_file]).codegen()
            fn = module.get_callable(direction, type_sig)
            tasks.append(Task(fn, arg_bufs))

    unique_id = -1
    def _uniqueid(self):
        Net.unique_id += 1
        return str(self.unique_id)

    def _synthesize_ast(self, ensemble, fn, direction):
        fn_def = util.get_ast(fn).body[0]

        self.connections_map[ensemble][0].mapping_inserted = False
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
        func_name = util.generate_unique_function_name()

        func_def = ast.FunctionDef(func_name,
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

            for i, arg in enumerate(mapping_func.args.args):
                i += 1  # offset batch
                mapping_func = util.inline_variable(arg.arg, "_neuron_index_{}".format(i), mapping_func)
            body = [
                # _neuron_index_1 = _neuron_index_1_outer * SIMDWIDTH + _neuron_index_1_inner
                ast.Assign([ast.Name("_neuron_index_1", ast.Load())],
                    ast.BinOp(
                        ast.BinOp(
                            ast.Name("_neuron_index_1_outer", ast.Load()),
                            ast.Mult(), 
                            ast.Num(SIMDWIDTH)
                        ),
                        ast.Add(), 
                        ast.Name("_neuron_index_1_inner", ast.Load())
                    )
                )
            ]
            for dim in range(1, ensemble.ndim):
                offset = mapping.get_offset(dim)
                input_offset = "_input_offset_{}".format(dim + 1)
                if isinstance(offset, ast.Name):
                    body += transformers.convert_enumerate_range.get_dependent_statements(mapping_func.body[:-1], offset.id)
                elif offset == -1:
                    offset = ast.Name("_neuron_index_{}".format(dim + 1), ast.Load())
                body.append(ast.Assign([ast.Name(input_offset, ast.Store())], offset))
            if mapping == one_to_one:
                body.append(ast.Assign([ast.Name("_input_offset_1", ast.Store())], ast.Name("_neuron_index_1_outer", ast.Load())))
                body.append(ast.Assign([ast.Name("_input_offset_1_inner", ast.Store())], ast.Name("_neuron_index_1_inner", ast.Load())))
            else:
                offset = mapping.get_offset(0)
                body.append(ast.Assign([ast.Name("_input_offset_1", ast.Store())], ast.BinOp(offset, ast.Div(), ast.Num(SIMDWIDTH))))
                body.append(ast.Assign([ast.Name("_input_offset_1_inner", ast.Store())], ast.BinOp(offset, ast.Mod(), ast.Num(SIMDWIDTH))))
            loop.body = body + loop.body
        func_def = transformers.convert_tuple_subscripts(func_def)

        func_def, tiled_buffers = transformers.convert_enumerate_ranges(func_def, direction)

        func_def = transformers.convert_sgemm_calls(func_def)
        func_def = PyBasicConversions().visit(func_def)

        # convert loopvars from long to int
        for loop in func_def.defn:
            loop.init.left.type = ctypes.c_int()
            for dim in range(ensemble.ndim + 1):
                loop = loop.body[0]
                loop.init.left.type = ctypes.c_int()

        for arg in func_def.params:
            buf = self.buffers[arg.name]
            arg.type = np.ctypeslib.ndpointer(buf.dtype, buf.ndim, buf.shape)()
        func_def = transformers.BasicTypeInference().visit(func_def)

        vectorized_buffers = {key: [(value, TILE_SIZE)] for key, value in tiled_buffers.items()}
        vectorized_buffers[ensemble.name+"inputs"] = [(2, SIMDWIDTH)]
        vectorized_buffers[ensemble.name+"grad_inputs"] = [(2, SIMDWIDTH)]
        # func_def, tiled_buffers = transformers.tile_outer_loop(func_def, ensemble.ndim)
        # func_def = transformers.interchange_tiled_loops(func_def)

        # for key in tiled_buffers.keys():
        #     if key in vectorized_buffers:
        #         vectorized_buffers[key].append((tiled_buffers[key], TILE_SIZE))
        #     else:
        #         vectorized_buffers[key] = [(tiled_buffers[key], TILE_SIZE)]

        candidate = transformers.get_loop_to_vectorize(func_def)

        if candidate == "_neuron_index_1_inner":
            unroll_target_loop_var = "_neuron_index_{}".format(ensemble.ndim)
        else:
            unroll_target_loop_var = "_neuron_index_1_inner"
        # func_def = transformers.register_promote_value_refs(func_def, ensemble, direction, self.batch_size, target_loop_var)
        if candidate is not None:
            func_def, transposed_buffers = transformers.vectorize_loop(func_def, candidate)
            # func_def = transformers.interchange_inner_loop(func_def)
            func_def = transformers.register_promote_vector_loads_stores(func_def)
            func_def = transformers.lift_invariant_load_stores(func_def)
            func_def = transformers.fma_replace(func_def)
            # func_def = transformers.unroll_constant_loops(func_def)

        for loop in func_def.defn:
            # count = 1
            # curr_loop = loop
            # while len(curr_loop.body) == 1 and isinstance(curr_loop.body[0], C.For):
            #     count += 1
            #     curr_loop = curr_loop.body[0]
            # loop.pragma = "omp parallel for collapse({})".format(count)
            loop.pragma = "omp parallel for collapse(2)"
            # loop.pragma = "omp parallel for"

        unroll = True
        if candidate is not None and unroll:
            if unroll_target_loop_var == "_neuron_index_1_inner":
                unroll_factor = SIMDWIDTH
            else:
                unroll_factor = UNROLL_FACTOR
                unroll_dim = ensemble.shape[-1]
                if ensemble.ndim == 1:
                    unroll_dim //= SIMDWIDTH
                while unroll_factor > unroll_dim or unroll_dim % unroll_factor != 0 :
                    unroll_factor -= 2
                    if unroll_factor == 0:
                        break
            if unroll_factor > 1:
                func_def = transformers.unroll_inner_neuron_loop(func_def, unroll_target_loop_var, unroll_factor)
                # func_def = transformers.promote_single_use_registers(func_def)
                # func_def = transformers.interleave_loads(func_def)

        type_sig = []
        casts = []
        for arg in args:
            name = arg.arg
            buf = self.buffers[name]
            buf_shape = list(buf.shape)
            if buf.ctypes._data not in self.reshaped_buffers:
                # if candidate is not None and name in transposed_buffers:
                #     dim = transposed_buffers[name]
                #     buf_shape[dim] //= SIMDWIDTH
                #     buf_shape.append(SIMDWIDTH)
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
            else:
                buf_shape = self.reshaped_buffers[buf.ctypes._data]
                self.buffers[name] = buf.reshape(buf_shape)
            # casts.insert(0, StringTemplate("__assume_aligned({}, 64);\n".format(name)))
            self._insert_cast(casts, buf_shape[1:], name, buf.dtype)
            # casts.insert(0, StringTemplate("__assume_aligned(_{}, 64);\n".format(name)))

        if candidate is not None:
            for buffer_name, trans_dim in transposed_buffers.items():
                curr_body = []
                shape = self.buffers[buffer_name].shape
                # if buffer_name in vectorized_buffers:
                #     shape = list(shape)
                #     for (dim, factor) in vectorized_buffers[buffer_name]:
                #         dim_to_vectorize = len(shape) - dim - 1
                #         shape[dim_to_vectorize] //= factor
                #         shape.append(factor)
                if "grad_" in buffer_name:
                    # node = C.For(
                    #     C.Assign(C.SymbolRef("x0", ctypes.c_int()), C.Constant(0)),
                    #     C.Lt(C.SymbolRef("x0"), C.Constant(shape[0])),
                    #     C.PostInc(C.SymbolRef("x0")),
                    #     curr_body,
                    #     "omp parallel for simd"
                    # )
                    # idx = [ C.SymbolRef("x0")]
                    # for i, d in enumerate(shape[1:]):
                    #     i += 1  # offset range
                    #     curr_body.append(C.For(
                    #         C.Assign(C.SymbolRef("x" + str(i), ctypes.c_int()), C.Constant(0)),
                    #         C.Lt(C.SymbolRef("x" + str(i)), C.Constant(d)),
                    #         C.PostInc(C.SymbolRef("x" + str(i))),
                    #         []
                    #     ))
                    #     idx.append(C.SymbolRef("x" + str(i)))
                    #     curr_body = curr_body[-1].body
                    # curr_body.append(
                    #      C.AddAssign(
                    #          util.gen_index_expr(C.SymbolRef(buffer_name + "_transposed"), [C.Constant(0)] + idx[1:]),
                    #         util.gen_index_expr(C.SymbolRef(buffer_name + "_transposed"), idx)))

                    # func_def.defn.append(node)
                    curr_body = []

                    node = C.For(
                        C.Assign(C.SymbolRef("x0", ctypes.c_int()), C.Constant(0)),
                        C.Lt(C.SymbolRef("x0"), C.Constant(shape[0])),
                        C.PostInc(C.SymbolRef("x0")),
                        curr_body
                    )
                    idx = [C.SymbolRef("x0")]
                    for i, d in enumerate(shape[1:-trans_dim-1]):
                        i += 1  # offset range
                        curr_body.append(C.For(
                            C.Assign(C.SymbolRef("x" + str(i), ctypes.c_int()), C.Constant(0)),
                            C.Lt(C.SymbolRef("x" + str(i)), C.Constant(d)),
                            C.PostInc(C.SymbolRef("x" + str(i))),
                            []
                        ))
                        idx.append(C.SymbolRef("x" + str(i)))
                        curr_body = curr_body[-1].body
                    idx += [C.Constant(0), C.Constant(0)]
                    curr_body.append(C.FunctionCall(C.SymbolRef("transpose<SIMDWIDTH,SIMDWIDTH>"), 
                        [C.Ref(util.gen_index_expr(C.SymbolRef(buffer_name + "_transposed"), idx)),
                         C.Ref(util.gen_index_expr(C.SymbolRef(buffer_name), idx))]))
                    func_def.defn.append(node)
                else:
                    node = C.For(
                        C.Assign(C.SymbolRef("x0", ctypes.c_int()), C.Constant(0)),
                        C.Lt(C.SymbolRef("x0"), C.Constant(shape[0])),
                        C.PostInc(C.SymbolRef("x0")),
                        curr_body
                    )
                    idx = [C.SymbolRef("x0")]
                    for i, d in enumerate(shape[1:-trans_dim-1]):
                        i += 1  # offset range
                        curr_body.append(C.For(
                            C.Assign(C.SymbolRef("x" + str(i), ctypes.c_int()), C.Constant(0)),
                            C.Lt(C.SymbolRef("x" + str(i)), C.Constant(d)),
                            C.PostInc(C.SymbolRef("x" + str(i))),
                            []
                        ))
                        idx.append(C.SymbolRef("x" + str(i)))
                        curr_body = curr_body[-1].body
                    idx += [C.Constant(0), C.Constant(0)]
                    curr_body.append(C.FunctionCall(C.SymbolRef("transpose<SIMDWIDTH,SIMDWIDTH>"), 
                        [C.Ref(util.gen_index_expr(C.SymbolRef(buffer_name), idx)), 
                         C.Ref(util.gen_index_expr(C.SymbolRef(buffer_name + "_transposed"), idx))]))
                    func_def.defn.insert(0, node)
                shape_str = "".join("[{}]".format(d) for d in shape)

                args.append(ast.arg(buffer_name + "_transposed", None))
                self.buffers[buffer_name + "_transposed"] = util.zeros(shape, np.float32)
                self._insert_cast(casts, shape[1:], buffer_name + "_transposed", self.buffers[buffer_name + "_transposed"].dtype)
                # func_def.defn.insert(0, StringTemplate(
                #     "float $arg_name$shape = {};",
                #     {
                #         "arg_name": C.SymbolRef(buffer_name + "_transposed"), 
                #         "shape": C.SymbolRef(shape_str),
                #     }))
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
