import numpy as np
import numbers
import itertools
import ast
from .ensemble import Ensemble, DataEnsemble
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
# import logging
# logging.basicConfig(level=20)

SIMDWIDTH = 8
TILE_SIZE = SIMDWIDTH
UNROLL_FACTOR = 16

include = StringTemplate("""
#include <immintrin.h>
#include <mkl.h>
#include <stdio.h>
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

                func, args = self._synthesize_ast(ensemble, neuron.backward, "backward")
                self.backward_tasks.insert(0, Task(func, args))

    def _synthesize_ast(self, ensemble, fn, direction):
        fn_def = util.get_ast(fn).body[0]

        self.connections_map[ensemble][0].mapping_inserted = False
        transformer = transformers.NeuronTransformer(ensemble,
                self.connections_map[ensemble], self.buffer_dim_info)
        fn_def = transformer.visit(fn_def)

        # Reverse iteration space for row major indexing
        loop_vars = ["_neuron_index_{}".format(i) for i in range(ensemble.ndim + 1)][::-1]
        shape = ensemble.shape
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

        func_def = transformers.convert_tuple_subscripts(func_def)

        func_def, tiled_buffers = transformers.convert_enumerate_ranges(func_def)

        func_def = transformers.convert_sgemm_calls(func_def)
        func_def = PyBasicConversions().visit(func_def)

        vectorized_buffers = {key: [(value, TILE_SIZE)] for key, value in tiled_buffers.items()}
        func_def, tiled_buffers = transformers.tile_outer_loop(func_def, ensemble.ndim)

        for key in tiled_buffers.keys():
            if key in vectorized_buffers:
                vectorized_buffers[key].append((tiled_buffers[key], TILE_SIZE))
            else:
                vectorized_buffers[key] = [(tiled_buffers[key], TILE_SIZE)]

        candidate = transformers.get_loop_to_vectorize(func_def)

        if candidate == "_neuron_index_1_inner":
            unroll_target_loop_var = "_neuron_index_{}".format(ensemble.ndim)
        else:
            unroll_target_loop_var = "_neuron_index_1_inner"
        # func_def = transformers.register_promote_value_refs(func_def, ensemble, direction, self.batch_size, target_loop_var)
        func_def, transposed_buffers = transformers.vectorize_loop(func_def, candidate)
        func_def = transformers.register_promote_vector_loads_stores(func_def)
        func_def = transformers.lift_invariant_load_stores(func_def)
        func_def = transformers.fma_replace(func_def)

        unroll = True
        if unroll:
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
                func_def = transformers.promote_single_use_registers(func_def)
                # func_def = transformers.interleave_loads(func_def)

        for buffer_name, trans_dim in transposed_buffers.items():
            curr_body = []
            shape = self.buffers[buffer_name].shape
            if buffer_name in vectorized_buffers:
                shape = list(shape)
                for (dim, factor) in vectorized_buffers[buffer_name]:
                    dim_to_vectorize = len(shape) - dim - 1
                    shape[dim_to_vectorize] //= factor
                    shape.append(factor)
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
            if "grad_" in buffer_name:
                curr_body.append(C.FunctionCall(C.SymbolRef("transpose<SIMDWIDTH,SIMDWIDTH>"), 
                    [C.Ref(util.gen_index_expr(C.SymbolRef(buffer_name + "_transposed"), idx)),
                     C.Ref(util.gen_index_expr(C.SymbolRef(buffer_name), idx))]))
                func_def.defn.append(node)
            else:
                curr_body.append(C.FunctionCall(C.SymbolRef("transpose<SIMDWIDTH,SIMDWIDTH>"), 
                    [C.Ref(util.gen_index_expr(C.SymbolRef(buffer_name), idx)), 
                     C.Ref(util.gen_index_expr(C.SymbolRef(buffer_name + "_transposed"), idx))]))
                func_def.defn.insert(0, node)
            shape_str = "".join("[{}]".format(d) for d in shape)

            func_def.defn.insert(0, StringTemplate(
                "float $arg_name$shape = {};",
                {
                    "arg_name": C.SymbolRef(buffer_name + "_transposed"), 
                    "shape": C.SymbolRef(shape_str),
                }))

        type_sig = []
        for arg in func_def.params:
            name = arg.name
            arg.type = ctypes.POINTER(ctypes.c_float)()
            # arg._restrict = True
            arg.name = "_{}".format(name)
            buf = self.buffers[name]
            type_sig.append(np.ctypeslib.ndpointer(buf.dtype, buf.ndim, buf.shape))
            buf_shape = buf.shape
            if name.endswith("value") or name.endswith("grad"): # or "inputs" in name:
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
        # func_def.defn.insert(0, StringTemplate("#pragma omp parallel \n{\n"))
        # func_def.defn[-1].pragma = "omp for collapse(2)"
        # func_def.defn.append(StringTemplate("\n}\n"))
        c_file = C.CFile(func_name, [include, func_def], path=".compiled")
        # print(ctree.util.highlight(c_file.codegen()))
        module = ctree.nodes.Project([c_file]).codegen()
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
