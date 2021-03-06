'''
Copyright (c) 2015, Intel Corporation

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''
import numpy as np
import numbers
import itertools
import ast
import math
import latte.util as util
import astor
from itertools import product
import ctree
from ctree.transformations import PyBasicConversions
import ctree.c.nodes as C
from ctree.templates.nodes import StringTemplate, FileTemplate
import ctypes
import latte.transformers as transformers
import os
import inspect
import latte.config
from .ensemble import Ensemble, DataEnsemble, LRNEnsemble, ConcatEnsemble, ActivationEnsemble, LossEnsemble, AccuracyEnsemble, EnsembleGroup
from latte.mapping import Mapping, one_to_one
from latte.connection import Connection
from latte.task import Task
import latte.transformers.vectorize as vectorizer
import latte.transformers.prefetch as prefetcher
import latte.transformers.loop_simplify as loopsimplifier
import latte.transformers.copy_propagation as copypropagator
import latte.transformers.parallelize as parallelizer
import latte.transformers.code_motion as code_motion
import latte.transformers.unroll as unroller
import latte.analysis as analyzer
import latte.optimizations as optimizer
import latte.type_converter as type_converter
import latte.transformers.copy_to_register as  copy_to_register
import latte.transformers.scalar_expand as ScalarExpansion
#alternative tiling
#import latte.transformers.tile_loop as tile_loop

if "OPENCL" in latte.config.parallel_strategy:
    import pycl as cl

import logging
logger = logging.getLogger("latte")

transpose_path = {
    "AVX": "/templates/transpose_256.tmp.c",
    "AVX-2": "/templates/transpose_256.tmp.c",
    "AVX-512": "/templates/transpose_512.tmp.c"
}[latte.config.vec_config]
package_path = os.path.dirname(os.path.abspath(__file__))

if latte.config.MODE in ["DEV"]:
    forward_path = "/templates/forward0.cpp"
    backward_path = "/templates/backward1.cpp"
    forward_pre_gen = FileTemplate(package_path + forward_path)
    backward_pre_gen = FileTemplate(package_path + backward_path)


transpose = FileTemplate(package_path + transpose_path)

include = FileTemplate(package_path + "/templates/includes.tmpl.c", {
    "LATTE_PACKAGE_PATH": StringTemplate(package_path),
    "INCLUDE_RUNTIME": C.SymbolRef("1" if latte.config.parallel_strategy in ["SIMPLE_LOOP", "FLOWGRAPH_LOOP"] else "0"),
    "TRANSPOSE": transpose,
    "SIMDWIDTH": C.Constant(latte.config.SIMDWIDTH),
    "INCLUDE_OPENCL": C.SymbolRef("1" if "OPENCL" in latte.config.parallel_strategy else "0"),
    "INCLUDE_LIBXSMM": C.SymbolRef("1" if latte.config.parallel_strategy in ["LIBXSMMOPENMP"] else "0")
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
        self.cl_buffers = {}
        self.fuse_map ={}
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
        self.cbr_fusion = False

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

    def add_to_connection_map(self, sink_ens, connection):
        if sink_ens not in self.connections_map:
          self.connections_map[sink_ens]=[]
        self.connections_map[sink_ens].append(connection)

    def add_connections(self, source_ens, sink_ens, mapping, reshape=None, clamp=False):
        if isinstance(source_ens, EnsembleGroup):
            source_ens = source_ens.ensembles[-1]
        #self.connections.append(Connection(source_ens, sink_ens, mapping, reshape, clamp))
        connection=Connection(source_ens, sink_ens, mapping, reshape, clamp)
        self.connections.append(connection)
        self.add_to_connection_map(sink_ens, connection)
          

    def add_loss_connection(self, source_ens, sink_ens):
        #self.connections.append(Connection(source_ens, sink_ens, one_to_one, None))
        connection = Connection(source_ens, sink_ens, one_to_one, None)
        self.connections.append(connection)
        self.add_to_connection_map(sink_ens, connection)

    def add_one_to_one_connections(self, source_ens, sink_ens):
        if isinstance(source_ens, EnsembleGroup):
            source_ens = source_ens.ensembles[-1]
        #self.connections.append(Connection(source_ens, sink_ens, one_to_one, None))
        connection = Connection(source_ens, sink_ens, one_to_one, None)
        self.connections.append(connection)
        self.add_to_connection_map(sink_ens, connection)

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

    def _initialize_inputs(self, ensemble, source_target, buffer_name, connection=0):
        conn = self.connections_map[ensemble][connection]
        source_name = conn.source.name
        buff = self.buffers[source_name + source_target]
        if conn.reshape is not None:
            assert False, "Deprecated"
            buff = buff.reshape((self.batch_size, ) + conn.reshape)
        if isinstance(ensemble, ConcatEnsemble):
            self.buffers[buffer_name] = self.buffers[source_name + source_target]

            if "OPENCL" in latte.config.parallel_strategy:
                raise NotImplementedError("OpenCL not yet implemented for Concat") 
        else:
            self.buffers[buffer_name] = buff

        if "OPENCL" in latte.config.parallel_strategy:
            self.cl_buffers[buffer_name] = self.cl_buffers[source_name + source_target]

    def _initialize_numeric_field(self, ensemble, field):
        try:
            neuron = ensemble.neurons.flat[0]
            value = getattr(neuron, field)
            assert type(value) != None
            if field in ensemble.batch_fields:
                buff = util.empty((self.batch_size, ) + ensemble.shape, type(value))
                self.buffers[ensemble.name + field] = buff
                for index, neuron in ensemble:
                    buff[(slice(None),) + index] = getattr(neuron, field)   
                if "OPENCL" in latte.config.parallel_strategy:
                    buf, evt = cl.buffer_from_ndarray(latte.config.cl_queue, buff)
                    evt.wait()
                    self.cl_buffers[ensemble.name + field] = buf
            else:
                buff = util.empty(ensemble.shape, type(value))
                self.buffers[ensemble.name + field] = buff
                for index, neuron in ensemble:
                    buff[index] = getattr(neuron, field)
                if "OPENCL" in latte.config.parallel_strategy:
                    buf, evt = cl.buffer_from_ndarray(latte.config.cl_queue, buff)
                    evt.wait()
                    self.cl_buffers[ensemble.name + field] = buf
        except Exception as e:
            logger.error("error initializing numeric field for " + str(type(ensemble.neurons.flat[0])) + ", field: " + str(field))
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
        assert value.dtype != None
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
        if "OPENCL" in latte.config.parallel_strategy:
            buf, evt = cl.buffer_from_ndarray(latte.config.cl_queue, buff)
            evt.wait()
            self.cl_buffers[ensemble.name + field] = buf

    def _initialize_value_grad_activation(self, ensemble):
        for field, target in [("value", "inputs"), ("grad", "grad_inputs")]:
            target_buf = self.buffers[ensemble.name + target]
            self.buffers[ensemble.name + field] = target_buf
            if "OPENCL" in latte.config.parallel_strategy:
                target_buf = self.cl_buffers[ensemble.name + target]
                self.cl_buffers[ensemble.name + field] = target_buf

    def _initialize_value_grad(self, ensemble):
        for field in ["value", "grad"]:
            # p = (bottom_pad, top_pad)
            # d = size of a dimension
            shape = (self.batch_size, ) + \
                   tuple(p[0] + p[1] + d for p, d in zip(ensemble.pad, ensemble.shape))
            self.buffers[ensemble.name + field] = util.zeros(shape, np.float32)
            if "OPENCL" in latte.config.parallel_strategy:
                buf, evt = cl.buffer_from_ndarray(latte.config.cl_queue, self.buffers[ensemble.name + field])
                evt.wait()
                self.cl_buffers[ensemble.name + field] = buf

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
                if isinstance(ensemble, ConcatEnsemble):
                    for i in range(1, len(self.connections_map[ensemble])):
                            source_target = "value" if field == "inputs"  else "grad"
                            self._initialize_inputs(ensemble, source_target, buffer_name+str(i), i)
            else:
                value = getattr(neuron, field)
                if isinstance(value, numbers.Real) or isinstance(value, numbers.Integral):
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
            if isinstance(ensemble, ConcatEnsemble): 
                if field in ["inputs", "grad_inputs"]:
                    buffer_name2  = buffer_name
                    ensemble.set_buffer(field, self.buffers[buffer_name2])

                    for i in range(1,len(self.connections_map[ensemble])):
                        buffer_name2  = buffer_name + str(i)
                        if "OPENCL" in latte.config.parallel_strategy: 
                            raise NotImplementedError(field)#ensemble.set_buffer(field, self.buffers[buffer_name], self.cl_buffers[buffer_name])
                        else:
                            ensemble.set_buffer(field+str(i), self.buffers[buffer_name2])
                else:
                    ensemble.set_buffer(field, self.buffers[buffer_name])

            else:
                if "OPENCL" in latte.config.parallel_strategy:
                    ensemble.set_buffer(field, self.buffers[buffer_name], self.cl_buffers[buffer_name])
                else:
                    ensemble.set_buffer(field, self.buffers[buffer_name])

    def compile(self):
        task_groups = {}
        '''
        self.connections_map = {ensemble: [] for ensemble in self.ensembles}
        for connection in self.connections:
            self.connections_map[connection.sink].append(connection)
        '''

        in_place_buffer_map = {}
        forward_body = []
        forward_args = set()
 
        backward_body = []
        backward_args = set()
        weight_update_body = []

        logger.info("Initializing ensembles and synthesizing functions...")
        #logger.info("Compiling functions...")

        for ensemble in self.ensembles:
          logger.info("    {} [shape={}]".format(ensemble.name, ensemble.shape))
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
            if latte.config.MODE in ["DEV"]:
              args = self._synthesize_args(ensemble, neuron, "forward")
              forward_args = forward_args.union(args)

              args = self._synthesize_args(ensemble, neuron, "backward")
              backward_args = backward_args.union(args)

              args = self._synthesize_args(ensemble, neuron, "update_internal") 
              backward_args = backward_args.union(args)
            else:      
              body, args = self._synthesize_ast(ensemble, neuron, "forward")
              forward_args = forward_args.union(args)
              forward_body += body

              body, args = self._synthesize_ast(ensemble, neuron, "backward")
              backward_args = backward_args.union(args)
              backward_body = body + backward_body

              #body, args = self._synthesize_ast(ensemble, neuron, "update_internal")
              #backward_args = backward_args.union(args)
              #backward_body = body + backward_body
              body, args = self._synthesize_ast(ensemble, neuron, "update_internal")
              backward_args = backward_args.union(args)
              weight_update_body = body + weight_update_body

        if isinstance(ensemble, ActivationEnsemble):
                                source = self.connections_map[ensemble][0].source
                                in_place_buffer_map[source.name + "value"] = [ensemble.name + "inputs"]

        logger.info("Compiling functions...")
        backward_body += weight_update_body

        for args, direction, body, tasks, in zip([forward_args, backward_args], 
                                                   ["forward", "backward"],
                                                   [forward_body, backward_body],
                                                   [self.forward_tasks, self.backward_tasks],
                                                   ):
          args = list(args)
          #if latte.config.MODE in ["DEV"]:
          args.sort() 
          arg_bufs = [self.buffers[arg] for arg in args]
          type_sig = [np.ctypeslib.ndpointer(buf.dtype, buf.ndim, buf.shape) for buf in arg_bufs]
          params   = [C.SymbolRef("_" + arg, typ()) for arg, typ in zip(args, type_sig)]
          _id = self._uniqueid()
          if latte.config.MODE in ["RELEASE"]:  
            c_file = C.CFile("dummy"+ _id, [
               include, 
               C.FunctionDecl(None, C.SymbolRef(direction + _id), params, body)
               ], path=".compiled")

            #c_file._ext = "cpp"
              
            if direction=="forward" and latte.config.codegen_strategy == "AUTOVEC" and  "ON" in latte.config.SIMPLE_FUSION:
 
                c_file = transformers.simple_fusion(c_file)
          
            if direction == "forward" and (self.cbr_fusion or "ON" in latte.config.AUTO_FUSION):
                #print("FUSION ENABLED")
                #print(self.fuse_map)
                  
                c_file = code_motion.lift_intermediate_loads(c_file, self.fuse_map)
                c_file = transformers.simple_fusion(c_file)
                c_file = copy_to_register.register_copy(c_file, self.fuse_map)
            if "ON" in latte.config.TIMER:
                c_file = transformers.timer(c_file)
        
            new_body = []
            incr = -1
            kernels = {}
            '''
            for stmt in c_file.body[1].defn:
                    incr += 1
                    if isinstance(stmt, C.For): 
                               if hasattr(stmt, 'pre_trans') and stmt.pre_trans is not None:
                                   new_body.extend(stmt.pre_trans)
                               stmt = parallelizer.parallelize(stmt, self.buffers, self.cl_buffers, kernels, self.batch_size)
                               new_body.append(stmt)
                    else:
                            new_body.append(stmt)
            for arg in args:
              name = arg
              buf = self.buffers[name]
              new_body.insert(0, StringTemplate("__assume_aligned({}, 64);\n".format(name)))
              util.insert_cast(new_body, buf.shape[1:], name, buf.dtype)
            '''
            #c_file.body[1].defn = new_body 
            #c_file = transformers.promote_in_place_load_stores(c_file, in_place_buffer_map)
            if latte.config.parallel_strategy == "OPENMP": # or latte.config.parallel_strategy == "LIBXSMMOPENMP":
              outliner = transformers.Outliner(self.buffers, direction)  
              c_file = outliner.visit(c_file)
              main_func = C.FunctionDecl(None, C.SymbolRef(direction + _id), params, c_file.body[1].defn)
              for arg in args:
                name = arg
                buf = self.buffers[name]
                main_func.defn.insert(0, StringTemplate("__assume_aligned({}, 64);\n".format(name)))
                util.insert_cast(main_func.defn, buf.shape[1:], name, buf.dtype)
              all_funcs = [ main_func]+ outliner.new_funcs
              new_funcs = []
              for func in all_funcs:
                new_body=[]          
                for stmt in func.defn:
                  if isinstance(stmt, C.For): 
                    #if hasattr(stmt, 'pre_trans') and stmt.pre_trans is not None:
                    #    new_body.extend(stmt.pre_trans)
                    stmt = parallelizer.parallelize(stmt, self.buffers, self.cl_buffers, kernels, self.batch_size)
                    new_body.append(stmt)
                  else:
                    new_body.append(stmt)

                func.defn = new_body       
                new_funcs.append(func)   
            else: #no outliner for tbb, and other cases..
              prologue=[]
              incr = -1
              reduce_incr = -1
              execute_stmts=[]
              for stmt in c_file.body[1].defn:
                incr += 1
                reduce_incr +=1
                if isinstance(stmt, C.For): 
                  if latte.config.parallel_strategy == "FLOWGRAPH_LOOP":#flow_graph model
                    if hasattr(stmt, 'pre_trans') and stmt.pre_trans is not None:
                      prologue.append(stmt.pre_trans)  #new_body.extend(stmt.pre_trans)
                    if direction == "forward":
                      new_body.append(self._gen_graph_nodes_from_loop(stmt, incr, reduce_incr))
                      if incr > 0:
                        if "index_0" in stmt.init.left.name: #weight_update code needs reduction -- TODO
                          new_body.append(self._gen_graph_edges_for_loop(stmt, incr-1, incr))
                        else:
                          new_body.append(self._gen_graph_edges_reduce_loop(stmt, incr, reduce_incr-1))
                      else: 
                        execute_stmts.append(self._gen_execute_for_loop(stmt, incr))
                    else: #for backward and update use tbb code generation
                      stmt = parallelizer.tbb_parallelize(stmt, self.buffers, self.cl_buffers, kernels, self.batch_size)
                      new_body.append(stmt)
                  else: 
                    if hasattr(stmt, 'pre_trans') and stmt.pre_trans is not None:
                      new_body.append(stmt.pre_trans)  #new_body.extend(stmt.pre_trans)
                    stmt = parallelizer.parallelize(stmt, self.buffers, self.cl_buffers, kernels, self.batch_size)
                    new_body.append(stmt)
                  '''
                  stmt = parallelizer.tbb_parallelize(stmt, self.buffers, self.cl_buffers, kernels, self.batch_size)
                  new_body.append(stmt)
                  '''
                else:
                  new_body.append(stmt)
              new_body = prologue + new_body + execute_stmts  
              for arg in args:
                name = arg
                buf = self.buffers[name]
                new_body.insert(0, StringTemplate("__assume_aligned({}, 64);\n".format(name)))
                util.insert_cast(new_body, buf.shape[1:], name, buf.dtype)
              c_file.body[1].defn = new_body

            if len(args) > 0:
                   #shape_str = "{}* ".format(self.buffers[args2[0]].dtype) + args2[0].join(", {}* ".format(self.buffers[d].dtype) + "{}".format(d) for d in args2[1:])
                   shape_str = "{}* ".format(   ctree.types.codegen_type(ctree.types.get_c_type_from_numpy_dtype(self.buffers[args[0]].dtype)())) + \
                   "_"+ args[0].join(", {}* ".format( ctree.types.codegen_type(ctree.types.get_c_type_from_numpy_dtype(self.buffers[d].dtype)())) + \
                   "_"+ "{}".format(d) for d in args[1:])
            else:
                   shape_str =""
            first_func = StringTemplate("void  $func ($args);",
              {"func":C.SymbolRef(direction+ _id),
               "args":C.SymbolRef(shape_str)
              })
            if latte.config.parallel_strategy == "OPENMP": # or latte.config.parallel_strategy == "LIBXSMMOPENMP":
              c_file = C.CFile(direction + _id, [
                  include,first_func,outliner.func_headers,
                  new_funcs,
                  ], path=".compiled")
            else:
              c_file = C.CFile(direction + _id, [
                  include,c_file.body[1],
                  ], path=".compiled")

            c_file._ext = "cpp"
            c_file = transformers.promote_in_place_load_stores(c_file, in_place_buffer_map)
            if latte.config.parallel_strategy == "FLOWGRAPH_LOOP":
                c_file.body[1].defn.insert(0, StringTemplate("static FlowGraph graph;"))
                c_file.body[1].defn.append(StringTemplate("graph.wait_for_all();"))
            elif "OPENCL" in latte.config.parallel_strategy:
                arg_bufs.append(latte.config.cl_queue)
                type_sig.append(cl.cl_command_queue)
                params.append(C.SymbolRef("queue", cl.cl_command_queue()))
                for name, kernel in kernels.items():
                    arg_bufs.append(kernel)
                    type_sig.append(cl.cl_kernel)
                    params.append(C.SymbolRef(name, cl.cl_kernel()))
          else: #DEV mode
            if direction == "forward":
                c_file = C.CFile(direction + _id, [
                forward_pre_gen
                ], path=".compiled")
            else:
                c_file = C.CFile(direction + _id, [
                backward_pre_gen
                ], path=".compiled")

            c_file._ext = "cpp"
          module = util.mpi_compile(ctree.nodes.Project([c_file]))
                           # get_callable(functions_handle, type_signature)
          type_sig = ctypes.CFUNCTYPE(None, *type_sig)
          fn = module.get_callable(direction + _id, type_sig)
          tasks.append(Task(fn, arg_bufs))

        '''
        =======
        for ensemble in self.ensembles:
            logger.info("    {} [shape={}]".format(ensemble.name, ensemble.shape))
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

                body, args = self._synthesize_ast(ensemble, neuron, "forward")
                forward_args = forward_args.union(args)
                forward_body += body

                body, args = self._synthesize_ast(ensemble, neuron, "backward")
                backward_args = backward_args.union(args)
                backward_body = body + backward_body

                body, args = self._synthesize_ast(ensemble, neuron, "update_internal")
                backward_args = backward_args.union(args)
                backward_body = body + backward_body

            if isinstance(ensemble, ActivationEnsemble):
                source = self.connections_map[ensemble][0].source
                in_place_buffer_map[source.name + "value"] = [ensemble.name + "inputs"]
        logger.info("Compiling functions...")
        for args, direction, body, tasks, in zip([forward_args, backward_args], 
                                                 ["forward", "backward"],
                                                 [forward_body, backward_body],
                                                 [self.forward_tasks, self.backward_tasks],
                                                 ):
            args = list(args)
            
            #if latte.config.MODE in ["DEV"]:
            args.sort() 

            arg_bufs = [self.buffers[arg] for arg in args]
            type_sig = [np.ctypeslib.ndpointer(buf.dtype, buf.ndim, buf.shape) for buf in arg_bufs]
            params   = [C.SymbolRef("_" + arg, typ()) for arg, typ in zip(args, type_sig)]
            args2   = [C.SymbolRef(arg, typ()) for arg, typ in zip(args, type_sig)]

            #args_duplicate = 
            _id = self._uniqueid()
     
            c_file = C.CFile(direction + _id, [
                    include, 
                    C.FunctionDecl(None, C.SymbolRef(direction + _id), params, body)
                ], path=".compiled")

            c_file._ext = "cpp"
            
            c_file = transformers.simple_fusion(c_file)
            if "ON" in latte.config.TIMER:
              c_file = transformers.timer(c_file)
            
            new_body = []
            incr = -1
            kernels = {}

            #for stmt in func_def:
            for stmt in c_file.body[1].defn:
 
                incr += 1
                #stmt = analyzer.type_infer(stmt)
                #stmt = optimizer.propogate_constants(stmt)
                #stmt = analyzer.type_infer(stmt)
                #stmt = optimizer.propogate_constants(stmt)

                if isinstance(stmt, C.For): 
                    if hasattr(stmt, 'pre_trans') and stmt.pre_trans is not None:
                        new_body.extend(stmt.pre_trans)
                    # loopvar1 = C.SymbolRef(stmt.init.left.name)
                    # looplen1 = stmt.test.right
                    # body = stmt.body
                    # new_body.append(self._gen_graph_nodes_from_loop(stmt, incr))
                    stmt = parallelizer.parallelize(stmt, self.buffers, self.cl_buffers, kernels, self.batch_size)
                    new_body.append(stmt)
                    # if incr > 0:
                    #     new_body.append(self._gen_graph_edges_for_loop(stmt, incr-1, incr))
                    # else:
                    #     execute_stmt = self._gen_execute_for_loop(stmt, incr)
                else:
                    new_body.append(stmt)

            for arg in args:
                name = arg
                buf = self.buffers[name]
                new_body.insert(0, StringTemplate("__assume_aligned({}, 64);\n".format(name)))
                generate_malloc = 0
                if generate_malloc == 1:
                  util.insert_malloc(new_body, buf.shape, name, buf.dtype)
                  new_body.append(util.insert_free(name))
                else :
                  util.insert_cast(new_body, buf.shape[1:], name, buf.dtype)

            c_file.body[1].defn = new_body 

            c_file = transformers.promote_in_place_load_stores(c_file, in_place_buffer_map)
            if latte.config.parallel_strategy == "FLOWGRAPH_LOOP":
                c_file.body[1].defn.insert(0, StringTemplate("static FlowGraph graph;"))
                c_file.body[1].defn.append(StringTemplate("graph.wait_for_all();"))
            elif "OPENCL" in latte.config.parallel_strategy:
                arg_bufs.append(latte.config.cl_queue)
                type_sig.append(cl.cl_command_queue)
                params.append(C.SymbolRef("queue", cl.cl_command_queue()))
                for name, kernel in kernels.items():
                    arg_bufs.append(kernel)
                    type_sig.append(cl.cl_kernel)
                    params.append(C.SymbolRef(name, cl.cl_kernel()))
            # c_file = transformers.remove_repeated_declarations(c_file)
           
            if latte.config.MODE in ["DEV"]:
 
                if direction == "forward":
                    c_file = C.CFile(direction + _id, [
                    forward_pre_gen
                    ], path=".compiled")
                else:
                    c_file = C.CFile(direction + _id, [
                    backward_pre_gen
                    ], path=".compiled")
 
                c_file._ext = "cpp"
 

            module = util.mpi_compile(ctree.nodes.Project([c_file]))
            # get_callable(functions_handle, type_signature)

            type_sig = ctypes.CFUNCTYPE(None, *type_sig)
            fn = module.get_callable(direction + _id, type_sig)
            tasks.append(Task(fn, arg_bufs))
        >>>>>>> b5dd63644aff23a71704523a7f19b60e050d76d2
        '''
        self._collect_value_grad_bufs()
        logger.info("Finished compiling Net")

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

    def _mark_parallel_loops(self, func_def, loopvars):
        class Marker(ast.NodeTransformer):
            def __init__(self, loopvars):
                self.loopvars = loopvars

            def visit_For(self, node):
                node.body = [self.visit(s) for s in node.body]
                if node.init.left.name in self.loopvars:
                    node.parallel = True
                else:
                    node.parallel = False
                return node

        return Marker(loopvars).visit(func_def)

    def _reshape_buffer(self, args, ensemble):
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
            #ANAND: 09/21/2016
            if field in ensemble.tiling_info and  "inputs" not in field and "grad_inputs" not in field :
                if name not in self.reshaped_buffers:
                    buf_shape = compute_tiled_shape(buf_shape, field, ensemble)
                    self.reshaped_buffers[name] = buf_shape
                    self.buffers[name] = buf.reshape(buf_shape)

    def _gen_graph_nodes_from_loop(self, loop, id, reduce_id):
        loopvar1 = C.SymbolRef(loop.init.left.name)
        looplen1 = loop.test.right
        loopincr = loop.incr.value
        #if direction == "forward" :
        #  body = parallelizer.tbb_parallelize_inner(loop.body, self.buffers, self.batch_size)
        #else:
        body = loop.body
        to_return = []
        if all(isinstance(s, C.For) and hasattr(s, 'parallel') and s.parallel for s in loop.body):
            for s in loop.body:
              if isinstance(s, C.For):
                loopvar2 = C.SymbolRef(s.init.left.name)
                looplen2 = s.test.right
                loopincr2 = s.incr.value
                inner_body = s.body
                '''
                if loopincr == 1:
                '''
                to_return.append(StringTemplate("""
                  std::vector<ContinueNode *> $node_list;
                  for (int __z = 0; __z < $looplen1 / $loopincr; __z+=$block) {
                    ContinueNode *node = new ContinueNode(&graph, [=]() {
                      for (int ___z=__z; ___z<__z+$block; ___z++) {
                        int $loopvar1 = ___z * $loopincr;
                        parallel_for(0,$looplen2 / $loopincr2,
                          [=](int low, int high) {
                          for (int tmp_$loopvar2 = low; tmp_$loopvar2 < high; tmp_$loopvar2++) {
                            int $loopvar2 = tmp_$loopvar2 * $loopincr2;
                            $body;
                          }
                        }
                        );
                      }
                    });
                    for (int i = 0; i < $loopincr; i++)
                      $node_list.push_back(node);
                  }
                  """, {'loopvar1': loopvar1, 'looplen1': looplen1, 'loopincr': loopincr,
                  'loopvar2': loopvar2, 'looplen2': looplen2, 'loopincr2': loopincr2,
                  'body': inner_body, 
                  'node_list': C.SymbolRef("node_list_" + str(id)),
                  'block': C.Constant(latte.config.img_block_size)
                }));
                '''
                else:
                  to_return.append(StringTemplate("""
                    std::vector<ContinueNode *> $node_list;
                    for (int __z = 0; __z < $looplen1 / $loopincr; __z++) {
                      ContinueNode *node = new ContinueNode(&graph, [=]() {
                        int $loopvar1 = __z * $loopincr;
                        parallel_for(0,$looplen2 / $loopincr2,
                          [=](int low, int high) {
                          for (int tmp_$loopvar2 = low; tmp_$loopvar2 < high; tmp_$loopvar2++) {
                            int $loopvar2 = tmp_$loopvar2 * $loopincr2;
                            $body;
                          }
                        }
                        );
                      });
                      for (int i = 0; i < $loopincr; i++)
                        $node_list.push_back(node);
                    }
                    """, {'loopvar1': loopvar1, 'looplen1': looplen1, 'loopincr': loopincr,
                    'loopvar2': loopvar2, 'looplen2': looplen2, 'loopincr2': loopincr2,
                    'body': inner_body, 
                    'node_list': C.SymbolRef("node_list_" + str(id))
                }));
                '''
        else:
            to_return = [StringTemplate("""
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
            })]
        if hasattr(loop, 'reduce_vars') and len(loop.reduce_vars) > 0:
          for var in loop.reduce_vars:
            size = np.prod(self.buffers[var].shape[1:])
            to_return.append(StringTemplate("""
              ContinueNode *$reduce_node = new ContinueNode(&graph, [=]() {
              parallel_for(0,$size,
                [=](int low, int high) {
                  #pragma simd
                  for (int x = low; x < high; ++x) {
                    float sum = _$arr[x];
                    #pragma unroll
                    for (int i = 1; i < $batch_size; ++ i) {
                      sum += _$arr[i * $size + x];
                    }
                    _$arr[x] = sum;
                  }
              });
              });
              for (int i = 0; i < $looplen1; i+=$loopincr) {
                make_edge($node_list[i], $reduce_node);
              };
            """, {'size': C.Constant(size),
                  'batch_size': C.Constant(self.batch_size),
                  'arr': C.SymbolRef(var),
                  'node_list': C.SymbolRef("node_list_"+str(id)),
                  'reduce_node': C.SymbolRef("reduce_node_"+str(reduce_id)),
                  'looplen1': C.Constant(looplen1.value),  
                  'loopincr': C.Constant(loopincr)
          }))
        return to_return

    def _gen_graph_edges_for_loop(self, loop, source_id, sink_id):
        loopvar1 = C.SymbolRef(loop.init.left.name)
        looplen1 = loop.test.right
        loopincr = loop.incr.value.value
        return StringTemplate("""
          for (int i = 0; i < $looplen1/$block; i+=1) {
            make_edge($prev_node_list[i], $node_list[i]);
          }
        """, {
            'looplen1': C.Constant(looplen1.value), 'loopincr': C.Constant(loopincr),
            'node_list': C.SymbolRef("node_list_" + str(sink_id)),
            'prev_node_list': C.SymbolRef("node_list_" + str(source_id)),
            'block': C.Constant(latte.config.img_block_size)
        })

    def _gen_graph_edges_reduce_loop(self, loop, source_id, sink_id):
        loopvar1 = C.SymbolRef(loop.init.left.name)
        looplen1 = loop.test.right
        loopincr = loop.incr.value.value
        return StringTemplate("""
          for (int i = 0; i < $looplen1; ++i) {
            make_edge($prev_node_list[i], $node_list);
          }
        """, {
            'looplen1': C.Constant(looplen1.value), 'loopincr': C.Constant(loopincr),
            'node_list': C.SymbolRef("reduce_node_" + str(sink_id)),
            'prev_node_list': C.SymbolRef("node_list_" + str(source_id)),
        })

    def _gen_execute_for_loop(self, loop, id):
        looplen1 = loop.test.right
        loopincr = loop.incr.value.value
        return StringTemplate("""
           for (int __z = 0; __z < $looplen1 / $loopincr / $block; __z+=1) {
            $node_list[__z]->execute();
          }
        """, {
            'looplen1': C.Constant(looplen1.value), 'loopincr': C.Constant(loopincr), 
            'node_list': C.SymbolRef("node_list_" + str(id)),
            'block': C.Constant(latte.config.img_block_size)
        })
        '''
        return StringTemplate("""
          for (int i = 0; i < $looplen1; i+=$loopincr) {
            $node_list[i]->execute();
          }
        """, {
            'looplen1': C.Constant(looplen1.value), 'loopincr': C.Constant(loopincr), 
            'node_list': C.SymbolRef("node_list_" + str(id)),
        })
        '''

    def _gen_libxsmm_function(self, ensemble, neuron, direction):
      #print("weights:" , neuron.weights.shape)
      #print("ensamble.shape:" , ensemble.shape)
      #print("nifm:", self.connections_map[ensemble][0].source.shape[0], " ifh:", self.connections_map[ensemble][0].source.shape[1], " ifw:", self.connections_map[ensemble][0].source.shape[2])
      #print("stride:", ensemble.stride)
      #print("pad_in_0:", self.connections_map[ensemble][0].source.pad[0], "pad_in_1:" , self.connections_map[ensemble][0].source.pad[1] , "pad_out_0" , ensemble.pad[0])
      #print("name:",ensemble.name)
      input_name = ensemble.name + "inputs"
      output_name = ensemble.name+"value"
      filter_name = ensemble.name+"weights_transposed"
      #print("filter:",C.SymbolRef(filter_name))
      if direction == "forward" :
        return StringTemplate("""
      {
      libxsmm_dnn_conv_desc conv_desc;
      libxsmm_dnn_layer*  libxsmm_handle;
      libxsmm_dnn_buffer* libxsmm_input;
      libxsmm_dnn_buffer* libxsmm_output;
      libxsmm_dnn_filter* libxsmm_filter;
      libxsmm_dnn_err_t status; 
      void* scratch;
      unsigned int kind;
      conv_desc.N = $nImg;
      conv_desc.C = $nIfm;
      conv_desc.H = $ifh + $pad_in_0 + $pad_in_1 ;
      conv_desc.W = $ifw + $pad_in_0 + $pad_in_1 ;
      conv_desc.K = $nOfm;
      conv_desc.R = $kh;
      conv_desc.S = $kw;
      conv_desc.u = $stride_h;
      conv_desc.v = $stride_w;
      conv_desc.pad_h = $pad_in_0;
      conv_desc.pad_w = $pad_in_0;
      conv_desc.pad_h_in = $pad_in_0;
      conv_desc.pad_w_in = $pad_in_0;
      conv_desc.pad_h_out = $pad_out_0;
      conv_desc.pad_w_out = $pad_out_0;
      conv_desc.threads = omp_get_max_threads();
      conv_desc.algo = LIBXSMM_DNN_CONV_ALGO_DIRECT;//ANAND: changed from AUTO
      conv_desc.buffer_format = LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM;
      conv_desc.filter_format = LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM;
      conv_desc.fuse_ops = LIBXSMM_DNN_CONV_FUSE_NONE;
      conv_desc.options = LIBXSMM_DNN_CONV_OPTION_NONE;
      conv_desc.datatype = LIBXSMM_DNN_DATATYPE_F32;
      libxsmm_handle = libxsmm_dnn_create_conv_layer( conv_desc, &status );
      /* setup LIBXSMM buffers and filter */
      libxsmm_input = libxsmm_dnn_link_buffer(libxsmm_handle, LIBXSMM_DNN_INPUT, $input, LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM_PTR, &status);
      libxsmm_output = libxsmm_dnn_link_buffer(libxsmm_handle, LIBXSMM_DNN_OUTPUT, $output, LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM_PTR, &status);
      libxsmm_filter = libxsmm_dnn_link_filter(libxsmm_handle, LIBXSMM_DNN_FILTER, $filter, LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM_PTR, &status);
      libxsmm_dnn_zero_buffer(libxsmm_output);
      libxsmm_dnn_bind_buffer(libxsmm_handle, libxsmm_input, LIBXSMM_DNN_REGULAR_INPUT);
      libxsmm_dnn_bind_buffer(libxsmm_handle, libxsmm_output, LIBXSMM_DNN_REGULAR_OUTPUT);
      libxsmm_dnn_bind_filter(libxsmm_handle, libxsmm_filter, LIBXSMM_DNN_REGULAR_FILTER);
      scratch = (void*)libxsmm_aligned_scratch( libxsmm_dnn_get_scratch_size( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL, &status ), 2097152);
      libxsmm_dnn_bind_scratch( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL, scratch );
      # pragma omp parallel
      {
        const int tid = omp_get_thread_num();
        libxsmm_dnn_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid ) ;
      }
      /* clean up */
      libxsmm_dnn_release_scratch( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL ); 
      libxsmm_dnn_release_buffer( libxsmm_handle, LIBXSMM_DNN_REGULAR_INPUT ); 
      libxsmm_dnn_release_buffer( libxsmm_handle, LIBXSMM_DNN_REGULAR_OUTPUT ); 
      libxsmm_dnn_release_filter( libxsmm_handle, LIBXSMM_DNN_REGULAR_FILTER ); 
      libxsmm_dnn_destroy_buffer(libxsmm_input); 
      libxsmm_dnn_destroy_buffer(libxsmm_output); 
      libxsmm_dnn_destroy_filter(libxsmm_filter); 
      libxsmm_dnn_destroy_conv_layer(libxsmm_handle); 
      libxsmm_free(scratch); 
      }
      """, {'nImg': C.Constant(self.batch_size)
      , 'nIfm': C.Constant(self.connections_map[ensemble][0].source.shape[0]) 
      , 'ifh': C.Constant(self.connections_map[ensemble][0].source.shape[1])
      , 'ifw': C.Constant(self.connections_map[ensemble][0].source.shape[2])
      , 'nOfm': C.Constant(ensemble.shape[0])
      , 'kh': C.Constant(neuron.weights.shape[1])
      , 'kw': C.Constant(neuron.weights.shape[2])
      , 'stride_h': C.Constant(ensemble.stride)
      , 'stride_w': C.Constant(ensemble.stride)
      , 'pad_in_0' : C.Constant(self.connections_map[ensemble][0].source.pad[0][0])
      , 'pad_in_1' : C.Constant(self.connections_map[ensemble][0].source.pad[1][0])
      , 'pad_out_0' : C.Constant(ensemble.pad[0][0])
      , 'input': C.SymbolRef(input_name)
      , 'output': C.SymbolRef(output_name)
      , 'filter': C.SymbolRef(filter_name)})

      elif direction == "backward":
        return StringTemplate("""
      {
      libxsmm_dnn_conv_desc conv_desc;
      libxsmm_dnn_layer*  libxsmm_handle;
      libxsmm_dnn_buffer* libxsmm_input;
      libxsmm_dnn_buffer* libxsmm_output;
      libxsmm_dnn_filter* libxsmm_filter;
      libxsmm_dnn_err_t status; 
      void* scratch;

      conv_desc.N = $nImg;
      conv_desc.C = $nIfm;
      conv_desc.H = $ifh + $pad_in_0 + $pad_in_1 ;
      conv_desc.W = $ifw + $pad_in_0 + $pad_in_1 ;
      conv_desc.K = $nOfm;
      conv_desc.R = $kh;
      conv_desc.S = $kw;
      conv_desc.u = $stride_h;
      conv_desc.v = $stride_w;
      conv_desc.pad_h = $pad_in_0;
      conv_desc.pad_w = $pad_in_0;
      conv_desc.pad_h_in = $pad_in_0;
      conv_desc.pad_w_in = $pad_in_0;
      conv_desc.pad_h_out = $pad_out_0;
      conv_desc.pad_w_out = $pad_out_0;
      conv_desc.threads = omp_get_max_threads();
      conv_desc.algo = LIBXSMM_DNN_CONV_ALGO_DIRECT;
      conv_desc.buffer_format = LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM;
      conv_desc.filter_format = LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM;
      conv_desc.fuse_ops = LIBXSMM_DNN_CONV_FUSE_NONE;
      conv_desc.options = LIBXSMM_DNN_CONV_OPTION_NONE;
      conv_desc.datatype = LIBXSMM_DNN_DATATYPE_F32;

      libxsmm_handle = libxsmm_dnn_create_conv_layer( conv_desc, &status );

      libxsmm_input = libxsmm_dnn_link_buffer(libxsmm_handle, LIBXSMM_DNN_INPUT, $input, LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM_PTR, &status);
      libxsmm_output = libxsmm_dnn_link_buffer(libxsmm_handle, LIBXSMM_DNN_OUTPUT, $output, LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM_PTR, &status);
      libxsmm_filter = libxsmm_dnn_link_filter(libxsmm_handle, LIBXSMM_DNN_FILTER, $filter, LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM_PTR, &status);
 
      libxsmm_dnn_zero_buffer(libxsmm_input);
      libxsmm_dnn_bind_buffer(libxsmm_handle, libxsmm_input, LIBXSMM_DNN_GRADIENT_INPUT);
      libxsmm_dnn_bind_buffer(libxsmm_handle, libxsmm_output, LIBXSMM_DNN_GRADIENT_OUTPUT);
      libxsmm_dnn_bind_filter(libxsmm_handle, libxsmm_filter, LIBXSMM_DNN_REGULAR_FILTER),

      scratch = (void*)libxsmm_aligned_scratch( libxsmm_dnn_get_scratch_size( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL, &status ), 2097152);
      libxsmm_dnn_bind_scratch( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL, scratch );
      libxsmm_dnn_transpose_filter(libxsmm_handle, LIBXSMM_DNN_FILTER);

      /* setup LIBXSMM buffers and filter
      libxsmm_input = libxsmm_dnn_link_input_buffer_check( libxsmm_handle,  $input, LIBXSMM_DNN_CONV_FORMAT_LIBXSMM_PTR, &status );
      libxsmm_output = libxsmm_dnn_link_output_buffer_check( libxsmm_handle,  $output, LIBXSMM_DNN_CONV_FORMAT_LIBXSMM_PTR, &status );
      libxsmm_filter = libxsmm_dnn_link_filter_check( libxsmm_handle,  $filter, LIBXSMM_DNN_CONV_FORMAT_LIBXSMM_PTR, &status );

      libxsmm_dnn_bind_input_buffer( libxsmm_handle, libxsmm_input ) ;
      libxsmm_dnn_bind_output_buffer( libxsmm_handle, libxsmm_output ) ;
      libxsmm_dnn_bind_filter( libxsmm_handle, libxsmm_filter ) ;
      */
    # pragma omp parallel
      {
        const int tid = omp_get_thread_num();
        libxsmm_dnn_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_BWD, 0, tid ) ;
      }
      
    /* clean up */
    libxsmm_dnn_release_scratch( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL );
    libxsmm_dnn_release_buffer( libxsmm_handle, LIBXSMM_DNN_GRADIENT_INPUT );
    libxsmm_dnn_release_buffer( libxsmm_handle, LIBXSMM_DNN_GRADIENT_OUTPUT );
    libxsmm_dnn_release_filter( libxsmm_handle, LIBXSMM_DNN_REGULAR_FILTER );
    libxsmm_dnn_destroy_buffer(libxsmm_input);
    libxsmm_dnn_destroy_buffer(libxsmm_output);
    libxsmm_dnn_destroy_filter(libxsmm_filter);
 
    libxsmm_dnn_destroy_conv_layer(libxsmm_handle);
    libxsmm_free(scratch);
 



      }
      """, {'nImg': C.Constant(self.batch_size)
      , 'nIfm': C.Constant(self.connections_map[ensemble][0].source.shape[0]) 
      , 'ifh': C.Constant(self.connections_map[ensemble][0].source.shape[1])
      , 'ifw': C.Constant(self.connections_map[ensemble][0].source.shape[2])
      , 'nOfm': C.Constant(ensemble.shape[0])
      , 'kh': C.Constant(neuron.weights.shape[1])
      , 'kw': C.Constant(neuron.weights.shape[2])
      , 'stride_h': C.Constant(ensemble.stride)
      , 'stride_w': C.Constant(ensemble.stride)
      , 'pad_in_0' : C.Constant(self.connections_map[ensemble][0].source.pad[0][0])# may need revsision:ANAND
      , 'pad_in_1' : C.Constant(self.connections_map[ensemble][0].source.pad[1][0])#''
      , 'pad_out_0' : C.Constant(ensemble.pad[0][0])
      , 'input': C.SymbolRef(ensemble.name + "inputs")
      , 'output': C.SymbolRef(ensemble.name+"grad")
      , 'filter': C.SymbolRef(ensemble.name+"grad_weights")})

      else:
        return StringTemplate("""
    {
    libxsmm_dnn_conv_desc conv_desc;
    libxsmm_dnn_layer*  libxsmm_handle;
    libxsmm_dnn_buffer* libxsmm_input;
    libxsmm_dnn_buffer* libxsmm_output;
    libxsmm_dnn_filter* libxsmm_filter;
    libxsmm_dnn_err_t status; 
    void* scratch;

    conv_desc.N = $nImg;
    conv_desc.C = $nIfm;
    conv_desc.H = $ifh + $pad_in_0 + $pad_in_1 ;
    conv_desc.W = $ifw + $pad_in_0 + $pad_in_1 ;
    conv_desc.K = $nOfm;
    conv_desc.R = $kh;
    conv_desc.S = $kw;
    conv_desc.u = $stride_h;
    conv_desc.v = $stride_w;
    conv_desc.pad_h = $pad_in_0;
    conv_desc.pad_w = $pad_in_0;
    conv_desc.pad_h_in = $pad_in_0;
    conv_desc.pad_w_in = $pad_in_0;
    conv_desc.pad_h_out = $pad_out_0;
    conv_desc.pad_w_out = $pad_out_0;
    conv_desc.threads = omp_get_max_threads();
    conv_desc.algo = LIBXSMM_DNN_CONV_ALGO_DIRECT;
    conv_desc.buffer_format = LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM;
    conv_desc.filter_format = LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM;
    conv_desc.fuse_ops = LIBXSMM_DNN_CONV_FUSE_NONE;
    //conv_desc.options = LIBXSMM_DNN_CONV_OPTION_NONE;
    conv_desc.options = LIBXSMM_DNN_CONV_OPTION_WU_EXT_FILTER_REDUCE;

    conv_desc.datatype = LIBXSMM_DNN_DATATYPE_F32;

    libxsmm_handle = libxsmm_dnn_create_conv_layer( conv_desc, &status );

    /* setup LIBXSMM buffers and filter*/

      libxsmm_input = libxsmm_dnn_link_buffer(libxsmm_handle, LIBXSMM_DNN_INPUT, $input, LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM_PTR, &status);
      libxsmm_output = libxsmm_dnn_link_buffer(libxsmm_handle, LIBXSMM_DNN_OUTPUT, $output, LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM_PTR, &status);
      libxsmm_filter = libxsmm_dnn_link_filter(libxsmm_handle, LIBXSMM_DNN_FILTER, $filter, LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM_PTR, &status);


    libxsmm_dnn_zero_filter(libxsmm_filter);
    libxsmm_dnn_bind_buffer(libxsmm_handle, libxsmm_input, LIBXSMM_DNN_REGULAR_INPUT);
    libxsmm_dnn_bind_buffer(libxsmm_handle, libxsmm_output, LIBXSMM_DNN_GRADIENT_OUTPUT);
    libxsmm_dnn_bind_filter(libxsmm_handle, libxsmm_filter, LIBXSMM_DNN_GRADIENT_FILTER);

    /* setup LIBXSMM buffers and filter 
    libxsmm_input = libxsmm_dnn_link_input_buffer_check( libxsmm_handle,  $input, LIBXSMM_DNN_CONV_FORMAT_LIBXSMM_PTR, &status );
    libxsmm_output = libxsmm_dnn_link_output_buffer_check( libxsmm_handle,  $output, LIBXSMM_DNN_CONV_FORMAT_LIBXSMM_PTR, &status );
    libxsmm_filter = libxsmm_dnn_link_filter_check( libxsmm_handle,  $filter, LIBXSMM_DNN_CONV_FORMAT_LIBXSMM_PTR, &status );

    libxsmm_dnn_bind_input_buffer( libxsmm_handle, libxsmm_input ) ;
    libxsmm_dnn_bind_output_buffer( libxsmm_handle, libxsmm_output ) ;
    libxsmm_dnn_bind_filter( libxsmm_handle, libxsmm_filter ) ;
    */
    scratch = (void*)libxsmm_aligned_scratch( libxsmm_dnn_get_scratch_size( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL, &status ), 2097152);
    libxsmm_dnn_bind_scratch( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL, scratch );

    # pragma omp parallel
    {
      const int tid = omp_get_thread_num();
      libxsmm_dnn_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_UPD, 0, tid );
    }
    
      libxsmm_dnn_reduce_wu_filters( libxsmm_handle, LIBXSMM_DNN_GRADIENT_FILTER );

    /* clean up */
    libxsmm_dnn_release_scratch( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL ); 
    libxsmm_dnn_release_buffer( libxsmm_handle, LIBXSMM_DNN_REGULAR_INPUT ); 
    libxsmm_dnn_release_buffer( libxsmm_handle, LIBXSMM_DNN_GRADIENT_OUTPUT ); 
    libxsmm_dnn_release_filter( libxsmm_handle, LIBXSMM_DNN_GRADIENT_FILTER ); 
    libxsmm_dnn_destroy_buffer(libxsmm_input); 
    libxsmm_dnn_destroy_buffer(libxsmm_output); 
    libxsmm_dnn_destroy_filter(libxsmm_filter); 
 
    libxsmm_dnn_destroy_conv_layer(libxsmm_handle); 
    libxsmm_free(scratch); 




     }
      """, {'nImg': C.Constant(self.batch_size)
      , 'nIfm': C.Constant(self.connections_map[ensemble][0].source.shape[0]) 
      , 'ifh': C.Constant(self.connections_map[ensemble][0].source.shape[1])
      , 'ifw': C.Constant(self.connections_map[ensemble][0].source.shape[2])
      , 'nOfm': C.Constant(ensemble.shape[0])
      , 'kh': C.Constant(neuron.weights.shape[1])
      , 'kw': C.Constant(neuron.weights.shape[2])
      , 'stride_h': C.Constant(ensemble.stride)
      , 'stride_w': C.Constant(ensemble.stride)
      , 'pad_in_0' : C.Constant(self.connections_map[ensemble][0].source.pad[0])#may need revision # ANAND two-sided padding, padding on each side may not be identical
      , 'pad_in_1' : C.Constant(self.connections_map[ensemble][0].source.pad[1])
      , 'pad_out_0' : C.Constant(ensemble.pad[0])
      , 'input': C.SymbolRef(ensemble.name + "inputs")
      , 'output': C.SymbolRef(ensemble.name+"grad")
      , 'filter': C.SymbolRef(ensemble.name+"grad_weights")})
              
    def _gen_transposes(self, transposed_buffers):
        pre_trans = []
        post_trans = []
        for buffer_name, trans_dim in transposed_buffers.items():
            curr_body = []
            shape = self.buffers[buffer_name].shape

            # node = util.gen_for("x0", 0, shape[0], curr_body)
            if latte.config.parallel_strategy == "OPENMP" or latte.config.parallel_strategy == "LIBXSMMOPENMP":
                node = StringTemplate("""
                #pragma omp parallel for
                for (int x0 = 0; x0 < $len; x0++) {
                  $body
                } 
                """, {'body': curr_body, 'len': C.Constant(shape[0])})
            else:
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



    def _synthesize_args(self, ensemble, neuron, direction):
       # get_ast returns an ast.Module, grab the first item for the function
        if direction == "forward":
            fn_def = util.get_ast(neuron.forward).body[0]
        elif direction == "update_internal":
            fn_def = util.get_ast(neuron.update_internal).body[0]
        else:
            # Don't backpropogate to DataEnsemble
            if not isinstance(self.connections_map[ensemble][0].source, DataEnsemble) or \
                    self.force_backward:
                fn_def = util.get_ast(neuron.backward).body[0]
                #if hasattr(neuron, 'update_internal'):
                #    fn_def.body += util.get_ast(neuron.update_internal).body[0].body
                #    fn_def = transformers.simple_fusion(fn_def)
            # elif hasattr(neuron, 'update_internal'):
            #     fn_def = util.get_ast(neuron.update_internal).body[0]
            else:
                # No-op
                return set()
        if isinstance(fn_def.body[0], ast.Pass):
            return set()
 
        #util.print_ast(fn_def)
        # transform domain constructs
        transformer = transformers.NeuronTransformer(ensemble,
                self.connections_map[ensemble], self.buffer_dim_info)
        fn_def = transformer.visit(fn_def)
        #util.print_ast(fn_def)
        # Grab seen variables
 
        args = [ast.arg(arg, None) for arg in transformer.seen_vars]

        if not isinstance(ensemble, ConcatEnsemble):
            shape = list(ensemble.shape)
        else:
            shape = list(self.connections_map[ensemble][0].source.shape)
            if direction == "forward":
                for i in range(1,len(self.connections_map[ensemble])):
                    source = ensemble.name
                    args.append(ast.arg(source + "inputs" + str(i), None))
            elif direction == "backward":
                for i in range(1,len(self.connections_map[ensemble])):
                    source = ensemble.name
                    args.append( ast.arg(source + "grad_inputs" + str(i), None))

        #for arg in args:
        #   buf = self.buffers[arg.arg]
        #   arg.type = np.ctypeslib.ndpointer(buf.dtype, buf.ndim, buf.shape)()
        





        if direction in ensemble.vectorize_info:
             # RAJ hack here

          loop_vars = ["_neuron_index_{}".format(i) for i in range(ensemble.ndim + 1)]

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
          func_def = ast.FunctionDef('func',
                ast.arguments(args, None, [], [], None, []), body,
                [], None)
      #util.print_ast(func_def)
          func_def = transformers.convert_tuple_subscripts(func_def)
          # convert domain ast nodes introduced by the neuron transformer
          func_def = transformers.convert_enumerate_ranges(func_def, direction, ensemble)
          # basic python -> C conversion
          func_def = PyBasicConversions().visit(func_def)

          _, transposed_buffers,_ = vectorizer.vectorize_loop(func_def,
                    ensemble.vectorize_info[direction][0])
 
 
          for buffer_name, trans_dim in transposed_buffers.items():
                curr_body = []
                shape = self.buffers[buffer_name].shape
                shape_str = "".join("[{}]".format(d) for d in shape)
 
                args.append(ast.arg(buffer_name + "_transposed", None))
                self.buffers[buffer_name + "_transposed"] = util.zeros(shape, np.float32)



        return [arg.arg for arg in args]


    def _synthesize_ast(self, ensemble, neuron, direction):
        # get_ast returns an ast.Module, grab the first item for the function
        if direction == "forward":
            fn_def = util.get_ast(neuron.forward).body[0]
        elif direction == "update_internal":
            fn_def = util.get_ast(neuron.update_internal).body[0]
        else:
            # Don't backpropogate to DataEnsemble
            if not isinstance(self.connections_map[ensemble][0].source, DataEnsemble) or \
                    self.force_backward:
                fn_def = util.get_ast(neuron.backward).body[0]
                #if hasattr(neuron, 'update_internal'):
                #    fn_def.body += util.get_ast(neuron.update_internal).body[0].body
                #    fn_def = transformers.simple_fusion(fn_def)
            # elif hasattr(neuron, 'update_internal'):
            #     fn_def = util.get_ast(neuron.update_internal).body[0]
            else:
                # No-op
                return [], set()
        if isinstance(fn_def.body[0], ast.Pass):
            return [], set()
        
        #util.print_ast(fn_def)
        # transform domain constructs
        transformer = transformers.NeuronTransformer(ensemble,
                self.connections_map[ensemble], self.buffer_dim_info)
        fn_def = transformer.visit(fn_def)
        #util.print_ast(fn_def)
        # Grab seen variables

        args = [ast.arg(arg, None) for arg in transformer.seen_vars]
        loop_vars = ["_neuron_index_{}".format(i) for i in range(ensemble.ndim + 1)]
        if not isinstance(ensemble, ConcatEnsemble):
            shape = list(ensemble.shape)
        else:
            shape = list(self.connections_map[ensemble][0].source.shape)
            if direction == "forward":
                for i in range(1,len(self.connections_map[ensemble])):
                    source = ensemble.name
                    args.append(ast.arg(source + "inputs" + str(i), None))                 
            elif direction == "backward":
                for i in range(1,len(self.connections_map[ensemble])):
                    source = ensemble.name
                    args.append( ast.arg(source + "grad_inputs" + str(i), None)) 
         # rename to reorder storage)
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

        # inserted by Raj
        if latte.config.parallel_strategy == "LIBXSMMOPENMP" and ensemble.use_libxsmm_lib == 1:
          body = self._gen_libxsmm_function(ensemble, neuron, direction)
          #print("body:", body)
          self._reshape_buffer(args, ensemble)

          args.append(ast.arg(ensemble.name + "weights_transposed", None))
              
          self.buffers[ensemble.name + "weights_transposed"] = util.zeros(shape, np.float32)

          func_def = ast.FunctionDef('func',
                ast.arguments(args, None, [], [], None, []), body,
                [], None)

          #util.print_ast(func_def)
          func_def = transformers.convert_tuple_subscripts(func_def)
          # convert domain ast nodes introduced by the neuron transformer
          func_def = transformers.convert_enumerate_ranges(func_def, direction, ensemble)
          # basic python -> C conversion
          func_def = PyBasicConversions().visit(func_def)
          # Seed the argument types as pointers for type inference
          for arg in func_def.params:
            buf = self.buffers[arg.name]
            arg.type = np.ctypeslib.ndpointer(buf.dtype, buf.ndim, buf.shape)()
          # Basic type inference and constant propogation
            func_def, type_table  = analyzer.type_infer(func_def)
          # func_def = optimizer.propogate_constants(func_def) [], None)
          

          #print(args[2])
          return func_def.defn, [arg.arg for arg in args]

        else:
          if not isinstance(ensemble, ConcatEnsemble):
              loop_ranges = ([self.batch_size] + [d for d in shape])[::-1]
              body = fn_def.body
              

              body = [util.gen_loop_nest(body, loop_vars, loop_ranges)]
          else:
              fullbody = []
              channel_offset = 0
              for j in range(len(self.connections_map[ensemble])):
                  loop_vars2 = loop_vars[0:ensemble.ndim]
                  shape = list(self.connections_map[ensemble][j].source.shape)
                  
                  if "value" in ensemble.tiling_info:
                      for dim, factor in ensemble.tiling_info["value"]:
                          if shape[dim] % factor != 0:
                              raise Exception(("Invalid tiling factor of {} on dimension "
                              + "{} for {}'s value buffer (shape={})").format(factor,
                                  dim, ensemble.name, shape))
   
                          shape[dim] //= factor
                          shape.append(factor)
                          loop_vars2.append(loop_vars2[dim + 1] + "_inner")
                          loop_vars2[dim + 1] += "_outer"



                  loop_ranges = ([d for d in shape])[::-1]
                  body = []
                  
                  #for dim in range(0, ensemble.ndim):
                  #    is_tiled_dim = "inputs" in ensemble.tiling_info and \
                  #        any(tiled_dim == dim
                  #                for tiled_dim, _ in ensemble.tiling_info["inputs"])
   
                  #      if is_tiled_dim:
                  #        for tiled_dim, factor in ensemble.tiling_info["inputs"]:
                  #            if tiled_dim == dim:
                  #                # factor is now the tiling factor for tiled_dim
                  #                break
                  #        mapping.set_arg(dim, ast.BinOp(
                  #                ast.BinOp(
                  #                ast.Name("_neuron_index_{}_outer".format(dim + 1), ast.Load()),
                  #                ast.Mult(),
                  #                ast.Num(factor)),
                  #                ast.Add(),
                  #                ast.Name("_neuron_index_{}_inner".format(dim + 1), ast.Load())
                  #            ))
                  #   else:
                  #        #else i ==  0 and dim == 0i:
                  #        #    mapping.set_arg(dim, ast.Name("_neuron_index_{}".format(dim + 1), ast.Store()))
                  #        #if i > 0 and dim == 0:
                  #        #    print("Entered %d", channel_offset) 
                  #        #     mapping.set_arg(dim, ast.BinOp(ast.Name("_neuron_index_{}".format(dim+1),ast.Add(),ast.Num(channel_offset))         
                  #        #    mapping.set_arg(dim, 
                  #        #        ast.BinOp(
                  #        #        ast.Name("_neuron_index_{}".format(dim + 1), ast.Store()),
                  #        #        ast.Add(),
                  #        #        ast.Num(channel_offset)))
                  #        #else :
                  #        mapping.set_arg(dim, ast.Name("_neuron_index_{}".format(dim + 1), ast.Store()))
                  #        #util.print_ast(mapping_func)    
                  #    offset = mapping.get_offset(dim)
                  #    input_offset = "_input_offset_{}".format(dim + 1)
   
                      #if dim == 0:
                  if j  == 0:
                      output_offset = "_output_offset_{}".format(1)
                  elif j > 0:
                      output_offset += str(1)
                       
                  body.append(ast.Assign([ast.Name(output_offset, ast.Store())], ast.Num(channel_offset)))

                  if direction == "forward":
   
                          
                      name = ensemble.name + "value"
                      if  j > 0:
                          name2 = ensemble.name + "inputs" + str(j)
                      else:
                          name2 = ensemble.name + "inputs"     
                      args1 = []
                      args2 = []
                      for i in range(ensemble.ndim + 1):
                          name3 = "_neuron_index_{}".format(i)
   
                          if "value" in ensemble.tiling_info:
                              for dim, factor in ensemble.tiling_info["value"]:
                                  if dim == i-1:
                                      name3 +="_outer"  

                          if i == 1:
                              s = str(1)
                              for k in range(j):
                                  s += str(1)

                              #name3 = "_neuron_index_{}".format(i)  
                                    
                              #if "value" in ensemble.tiling_info:
                              #    for dim, factor in ensemble.tiling_info["value"]:
                              #        if dim == 0                                                                                      
                              #            name3 +="_outer"    
                              args1.append(ast.BinOp(ast.Name(name3, ast.Load()), ast.Add(), ast.Name("_output_offset_" + s, ast.Load())))
                          else:
                              args1.append(ast.Name(name3, ast.Load()))#args1.append(ast.BinOp(ast.Name("_neuron_index_{}".format(i + offset), ast.Load()), ast.Add(), ast.Name(name2, ast.Load())))
                          args2.append(ast.Name(name3, ast.Load()))
       
                          #name3 = "_neuron_index_{}".format(i)
   
                      if "value" in ensemble.tiling_info:
                          for dim, factor in ensemble.tiling_info["value"]:
                              name3 = "_neuron_index_{}".format(dim+1)
                              args1.append(ast.Name(name3+"_inner", ast.Load()))    
                              args2.append(ast.Name(name3+"_inner", ast.Load()))             

                      body.append(ast.Assign([ast.Subscript(ast.Name(name, ast.Load()), ast.Index(ast.Tuple(args1, ast.Load())),\
                                      ast.Store())],ast.Subscript(ast.Name(name2, ast.Load()), ast.Index(ast.Tuple(args2, ast.Load())), ast.Load())))
                  elif direction == "backward":
                      name = ensemble.name + "grad"
                      if  j > 0:
                          name2 = ensemble.name + "grad_inputs" + str(j)
                      else:
                          name2 = ensemble.name + "grad_inputs"

                      args1 = []
                      args2 = []
   
                      for i in range(ensemble.ndim + 1):
                          name3 = "_neuron_index_{}".format(i)
      
                          if "grad" in ensemble.tiling_info:
                              for dim, factor in ensemble.tiling_info["grad"]:
                                  if dim == i-1:
                                      name3 +="_outer"
   
                          if i == 1:
                              s = str(1)
                              for k in range(j):
                                  s += str(1)
                              args1.append(ast.BinOp(ast.Name(name3, ast.Load()), ast.Add(), ast.Name("_output_offset_" + s, ast.Load())))
                          else:
                              args1.append(ast.Name(name3, ast.Load()))#args1.append(ast.BinOp(ast.Name("_neuron_index_{}".format(i + offset), ast.Load()), ast.Add(), ast.Name(name2, ast.Load())))
                          args2.append(ast.Name(name3, ast.Load()))
                      

                      if "grad" in ensemble.tiling_info:
                          for dim, factor in ensemble.tiling_info["grad"]:
                              name3 = "_neuron_index_{}".format(dim+1)
                              args1.append(ast.Name(name3+"_inner", ast.Load()))               
                              args2.append(ast.Name(name3+"_inner", ast.Load()))  
                      
                      body.append(ast.Assign([ast.Subscript(ast.Name(name2, ast.Load()), ast.Index(ast.Tuple(args2, ast.Load())),\
                                 ast.Store())],ast.Subscript(ast.Name(name, ast.Load()), ast.Index(ast.Tuple(args1, ast.Load())), ast.Load())))
                 


                                              
                  fullbody += [util.gen_loop_nest(body, loop_vars, loop_ranges)]
                  channel_offset += shape[0]
              
              body = [util.gen_loop_nest(fullbody, ["_neuron_index_0"],[self.batch_size])]
        
              #loop.body = body

        # This placeholder function is used to wrap the body of statements for
        # convenience, after transformations have completed, the function body
        # is returned and this function node will be discarded
        func_def = ast.FunctionDef('func',
                ast.arguments(args, None, [], [], None, []), body,
                [], None)

        #util.print_ast(func_def)
        #body = []
        # TODO: MAKE THIS A FUNCTION
       
        if not isinstance(ensemble, ConcatEnsemble):
            for loop in func_def.body:
                for dim in range(len(loop_vars) - 1):
                    loop = loop.body[0]

                mapping = self.connections_map[ensemble][0].mapping
                mapping_func = mapping.ast
                body = []
                #ANAND: 10/11/2016
                #tiled_vars holds loop indices that were tiled
                #index_vars are loop index variables
                #check for a variable name in array subscript that is not a loop index
                #update with proper tiling expression
                #TODO: special case for LRN, needs to be generalized
                if isinstance(ensemble, LRNEnsemble):
                    for var in transformer.tiled_vars: 
                        if var not in transformer.index_vars and "_neuron_index" not in var:
                            dim = transformer.tiled_vars[var]
                            loop.body = [util.replace_name(ast.Name(var+"_outer", ast.Load()), C.Div( C.Add(C.SymbolRef(var), C.SymbolRef("_input_offset_{}_inner".format(dim))), C.Constant( latte.config.SIMDWIDTH)), s)\
                                        for s in loop.body]
                            loop.body = [util.replace_name(ast.Name("_input_offset_{}_inner".format(dim), ast.Load()), C.Constant(0), s) for s in loop.body ]
                            loop.body = [util.replace_name(ast.Name(var+"_inner", ast.Load()), C.Mod( C.Add(C.SymbolRef(var), C.SymbolRef("_input_offset_{}_inner".format(dim))), C.Constant( latte.config.SIMDWIDTH)), s)\
                                        for s in loop.body]



                for dim in range(0, ensemble.ndim):
                    offset_num  = -1

                    is_tiled_dim = "inputs" in ensemble.tiling_info and \
                        any(tiled_dim == dim 
                            for tiled_dim, _ in ensemble.tiling_info["inputs"])

                    if is_tiled_dim:
                        for tiled_dim, factor in ensemble.tiling_info["inputs"]:
                            if tiled_dim == dim:
                                # factor is now the tiling factor for tiled_dim
                                break
                        
                        length = 0
                        if len(mapping.shape) > dim:
                            length = len(mapping.shape[dim])
 
                        if length == 1:
                              offset_num = 0

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
                        if offset_num == 0 :
                            body.append(ast.Assign([ast.Name(input_offset + "_outer", ast.Store())],  ast.Name("_neuron_index_{}_outer".format(dim + 1), ast.Load())))
                            body.append(ast.Assign([ast.Name(input_offset + "_inner", ast.Store())],  ast.Name("_neuron_index_{}_inner".format(dim + 1), ast.Load())))
                        else:
                            outer_offset = ast.BinOp(offset, ast.Div(), ast.Num(factor))
                            body.append(ast.Assign([ast.Name(input_offset + "_outer", ast.Store())], outer_offset))
                            inner_offset = ast.BinOp(offset, ast.Mod(), ast.Num(factor))
                            body.append(ast.Assign([ast.Name(input_offset + "_inner", ast.Store())], inner_offset))
                    else:
                        # Store the offset value
                        body.append(ast.Assign([ast.Name(input_offset, ast.Store())], offset))
                    # Prepend to body


                loop.body = body + loop.body

        func_def = transformers.convert_tuple_subscripts(func_def)
        # convert domain ast nodes introduced by the neuron transformer
        func_def = transformers.convert_enumerate_ranges(func_def, direction, ensemble)
        # basic python -> C conversion
        func_def = PyBasicConversions().visit(func_def)
        # handle math functions that are different in C than python
        func_def = transformers.PatternMatchMath().visit(func_def)

                       
        for loop in func_def.defn:
            # convert loopvars from long to int
            # we do this because PyBasicConversion defaults to using longs for
            # range based for loops
            loop.init.left.type = ctypes.c_int()
            for dim in range(len(loop_vars) - 1):
                loop = loop.body[0]
                #print(loop)
                loop.init.left.type = ctypes.c_int()
                input_shape = self.connections_map[ensemble][0].source.shape

                if latte.config.codegen_strategy == "AUTOVEC":

                  if dim == 0 or dim==1:
                    # Do not need to clamp batch dimension or channels dimension
                    continue

                  input_offset = "_input_offset_{}".format(dim)
#<<<<<<< HEAD
                  if not isinstance(ensemble, ConcatEnsemble): 
                #    if mapping.clamp:
                #        if dim in ensemble.tiling_info:
                #            gen_clamp = gen_gen_clamp(input_shape[dim - 1] // latte.config.SIMDWIDTH - 1)
                #            loop.body = [util.ClampInputIndex(input_offset, gen_clamp).visit(s) for s in loop.body]
                #            gen_clamp = gen_gen_clamp(latte.config.SIMDWIDTH - 1)
                #            loop.body = [util.ClampInputIndex(input_offset + "_inner", gen_clamp).visit(s) for s in loop.body]
                #        else:
                #            gen_clamp = gen_gen_clamp(input_shape[dim - 1] - 1)
                #            loop.body = [util.ClampInputIndex(input_offset, gen_clamp).visit(s) for s in loop.body]
#=======
                      if mapping.clamp:
                        if dim in ensemble.tiling_info:
                            gen_clamp = gen_gen_clamp(input_shape[dim - 1] // latte.config.SIMDWIDTH - 1)
                            loop.body = [util.ClampInputIndex(input_offset, gen_clamp).visit(s) for s in loop.body]
                            gen_clamp = gen_gen_clamp(latte.config.SIMDWIDTH - 1)
                            loop.body = [util.ClampInputIndex(input_offset + "_inner", gen_clamp).visit(s) for s in loop.body]
                        else:
                            gen_clamp = gen_gen_clamp(input_shape[dim - 1] - 1)
                            loop.body = [util.ClampInputIndex(input_offset, gen_clamp).visit(s) for s in loop.body]
                            if dim+1 == (len(loop_vars) - 1):
                                loop.body = [util.ClampInputIndex("_input_offset_{}".format(dim+1), gen_clamp).visit(s) for s in loop.body]

        #ANAND-- IN PROGRESS 1/5/2017
        if isinstance(ensemble, ConcatEnsemble):
             func_def = type_converter.loop_init_long_to_int(func_def)    
                ##for loop in func_def.defn:
                #loop.init.left.type = ctypes.c_int()
                #inner = loop.body[0]
                #inner.init.left.type = ctypes.c_int()
                #inner = inner.body[0]   
                #inner.init.left.type = ctypes.c_int()
                #inner = inner.body[0]
                #inner.init.left.type = ctypes.c_int() 
                #util.loop_init_long_to_int

        # Seed the argument types as pointers for type inference
        for arg in func_def.params:
           buf = self.buffers[arg.name]
           arg.type = np.ctypeslib.ndpointer(buf.dtype, buf.ndim, buf.shape)()
        # Basic type inference and constant propogation
        func_def, type_table  = analyzer.type_infer(func_def)
        func_def = optimizer.propogate_constants(func_def)
        

        if latte.config.codegen_strategy == "AUTOVEC" or ensemble.use_libxsmm_lib != 1:
          # optimizations applied
          for loop_var1, loop_var2 in ensemble.loops_to_swap[direction]:
            loop1 = util.find_loop(func_def, loop_var1)
            loop2 = util.find_loop(func_def, loop_var2)
            # loop.init = (int x = 0)
            # loop.test = (x < 5)
            # loop.incr = (x += 1)
            loop1.init, loop2.init = loop2.init, loop1.init
            loop1.test, loop2.test = loop2.test, loop1.test
            loop1.incr, loop2.incr = loop2.incr, loop1.incr

          # drop loops that iterate for one iteration only and constant propagate indices and hoist address computations
          #func_def = loopsimplifier.simplify_loops(func_def)
          #func_def = optimizer.propogate_constants(func_def)

 #HACK-- Needs cleaner handling -- Anand(27th June 2017)
          if direction in ensemble.scalar_expand_info and direction in ensemble.vectorize_info:
            func_def, transposed_buffers, symbol_map = vectorizer.vectorize_loop(func_def,
                    ensemble.vectorize_info[direction][0])
            # func_def = transformers.push_inner_loop_down(func_def)
            func_def, symbol_map2 = vectorizer.register_promote_vector_loads_stores(func_def, symbol_map)
            func_def, type_map, vector_type_map = ScalarExpansion.scalar_expand_vars(func_def, ensemble.scalar_expand_info[direction], type_table, symbol_map2)
 
            if direction in ensemble.if_convert_info:
               func_def  = ScalarExpansion.if_convert(func_def,type_map, vector_type_map)



          elif direction in ensemble.vectorize_info:
            # RAJ hack here
            func_def, transposed_buffers, symbol_map = vectorizer.vectorize_loop(func_def, 
                    ensemble.vectorize_info[direction][0])
            # func_def = transformers.push_inner_loop_down(func_def)
            func_def, symbol_map2 = vectorizer.register_promote_vector_loads_stores(func_def, symbol_map)
            func_def = code_motion.lift_invariant_load_stores(func_def)
            func_def = vectorizer.fuse_multiply_adds(func_def)

          if direction in ensemble.simd_info:
            for loopvar in ensemble.simd_info[direction]:
                func_def = transformers.insert_pragma_simd(func_def, loopvar)

          #for var, value, phase in ensemble.tiling_loop_info:
          #      #'tile' is the field
          #      #if phase == 'forward':
          #          #print("forward")
          #  tile_loop.shallow_tile(func_def,var,value)

          if "ON" in latte.config.unroll_option:
            if direction in ensemble.unroll_no_jam_info:
              unroll_dict_list = ensemble.unroll_no_jam_info[direction]
 
              for unroll_var, unroll_factor, unroll_type in unroll_dict_list:
              #  unroll_var = field
              #   unroll_factor, unroll_type = value
                func_def = unroller.unroll_no_jam_loop(func_def, unroll_var, unroll_factor, unroll_type)
                #func_def = loopsimplifier.simplify_loops(func_def)
 
            func_def = loopsimplifier.simplify_loops(func_def)

            if direction in ensemble.unroll_info:
              unroll_dict_list = ensemble.unroll_info[direction]
              for field, value in unroll_dict_list.items():
                unroll_var = field
                unroll_factor, unroll_type = value
                unroller.unroll_loop(func_def, unroll_var, unroll_factor, unroll_type)
          self._mark_parallel_loops(func_def, ensemble.parallel_info[direction])

          type_sig = []
          self._reshape_buffer(args, ensemble)

          if direction in ensemble.vectorize_info:
            pre_trans, post_trans = self._gen_transposes(transposed_buffers)

            for buffer_name, trans_dim in transposed_buffers.items():
                curr_body = []
                shape = self.buffers[buffer_name].shape
                shape_str = "".join("[{}]".format(d) for d in shape)

                args.append(ast.arg(buffer_name + "_transposed", None))
                self.buffers[buffer_name + "_transposed"] = util.zeros(shape, np.float32)
          else:
            pre_trans = []
            post_trans = []

          # check for fused code
          func_def = copypropagator.propagate_copies(func_def)
          if "ON" in latte.config.prefetch_option and direction in ensemble.prefetch_info:
              prefetch_dict_list = ensemble.prefetch_info[direction]
              for field, value in prefetch_dict_list.items():
                  if len(value) > 0 :
                    if value[0] == 1:
                        prefetch_type, enclosing_loop_var, dim, prefetch_count, prefetch_loop_var, prefetch_multiplier, prefetch_constant, cacheline_hint = value
                        prefetcher.insert_simple_prefetches(func_def, field, prefetch_type, enclosing_loop_var, dim, prefetch_count, prefetch_loop_var, prefetch_multiplier, prefetch_constant, cacheline_hint)
                    elif value[0] == 2:
                        prefetch_type, enclosing_loop_var, dim, prefetch_count, prefetch_offset, prefetch_dest_loop, prefetch_init_loop, prefetch_loop_var, prefetch_multiplier, prefetch_constant, prefetch_num_zeroes, cacheline_hint = value
                        prefetcher.insert_strided_prefetches(func_def, field, prefetch_type, enclosing_loop_var, dim, prefetch_count, prefetch_offset, prefetch_dest_loop, prefetch_init_loop, prefetch_loop_var, prefetch_multiplier, prefetch_constant, prefetch_num_zeroes, cacheline_hint)
                    elif value[0] == 3:
                        prefetch_type, enclosing_loop_var, dim, prefetch_count, prefetch_loop_var, prefetch_multiplier, prefetch_constant, prefetch_num_zeroes, cacheline_hint = value
                        prefetcher.insert_simple_hoist_prefetches(func_def, field, prefetch_type, enclosing_loop_var, dim, prefetch_count, prefetch_loop_var, prefetch_multiplier, prefetch_constant, prefetch_num_zeroes, cacheline_hint)
          # drop loops that iterate for one iteration only and constant propagate indices and hoist address computations
          func_def = loopsimplifier.simplify_loops(func_def)
          #func_def = optimizer.propogate_constants(func_def)

        #else: #GEMM formulation
        #   pre_trans = []
        #   post_trans = []  
        #  func_def = transformers.pattern_match_gemm(func_def)
        #  raise NotImplementedError("GEMM formulation is not complete yet")

        #assert isinstance(func_def.defn[0], C.For)
        func_def.defn[0].pre_trans = pre_trans
        reduce_vars = []
        func_def.defn[0].reduce_vars = reduce_vars
        for arg in args:
            if arg.arg.replace(ensemble.name, "") in ensemble.private_info:
                reduce_vars.append(arg.arg)
        return func_def.defn, [arg.arg for arg in args]

    def test(self):
        for task in self.forward_tasks:
            task()

    def forward(self):
        for task in self.forward_tasks:
            task()

    def backward(self):
        for task in self.backward_tasks:
            task()

    def fuse_pattern_cbr(self):
      self.cbr_fusion = True


    def fuse_cbrm(self,input_ens, pooling_ens, pooling_window, pooling_stride,output_width):
        # get relu ens unroll factors for h and w
 
        h_unroll_factor = 1
        w_unroll_factor = 1
 
        if 'forward' in input_ens.unroll_info:
            unroll_dict_list = input_ens.unroll_info['forward']
            for field, value in unroll_dict_list.items():
              unroll_var = field
              unroll_factor, unroll_type = value
              if unroll_var == "_neuron_index_2" and unroll_type == 1:
                  h_unroll_factor = unroll_factor
              elif unroll_var == "_neuron_index_3" and unroll_type == 1:
                  w_unroll_factor = unroll_factor
        # get relu bounds
 
        # assert relu bounds divide pooling window
        #if not relu_bounds % pooling_window == 0:
        #    return
 
        # relu_bounds
 
        # Case 1 : Assume relu bounds divides unroll factor and pooling_stride
        # B = k*u && B = l*s
        # if u > s  => l > k
        #if u = m*s  k*m*s = l*s => l = m*k   
        #if u = m*s + c   k*m*s + c*k = l*s  => l = k*(m + c/s)
 
        pooling_unroll_factor_h = 1
        if h_unroll_factor >= pooling_stride:
            if h_unroll_factor % pooling_stride == 0:
                if (h_unroll_factor/pooling_stride  + pooling_window - 1) <= h_unroll_factor:
                    pooling_unroll_factor_h = h_unroll_factor//pooling_stride
 
 
        pooling_unroll_factor_w = 1
        if w_unroll_factor >= pooling_stride:
            if w_unroll_factor % pooling_stride == 0:
                if (w_unroll_factor/pooling_stride  + pooling_window - 1) <= w_unroll_factor:
                    pooling_unroll_factor_w = w_unroll_factor//pooling_stride
 
 
        #if pooling_unroll_factor_w > 0:
        if pooling_unroll_factor_w > 0 and output_width%pooling_unroll_factor_w ==0:
           pooling_ens.unroll_no_jam(phase="forward", loop_var="_neuron_index_3", factor=pooling_unroll_factor_w, unroll_type= 1)



    def fuse_cbr(self, conv_bias_ens, relu_ens, is_fc=False):
      if isinstance(conv_bias_ens, EnsembleGroup): 
        if is_fc:
            bias_ens = conv_bias_ens.ensembles[-1]
            conv_ens = conv_bias_ens.ensembles[-2]
            self.fuse_map[bias_ens.name + "inputs"] = conv_ens.name + "value"
            return  

        bias_ens = conv_bias_ens.ensembles[-1]
        conv_ens = conv_bias_ens.ensembles[-2]
        self.fuse_map[bias_ens.name + "inputs"] = conv_ens.name + "value" 
        self.fuse_map[relu_ens.name + "inputs"] = bias_ens.name + "value"
        #fully connected

      else:
        conv_ens = conv_bias_ens
        self.fuse_map[relu_ens.name + "inputs"] = conv_ens.name + "value"
    
      IFM_BLOCK_THRESHOLD = 4
      HEIGHT_THRESHOLD = 7
      WIDTH_THRESHOLD = 7
      #print("Entered cbr\n") 
      channels = self.connections_map[conv_ens][0].source.shape[0]
      input_height = self.connections_map[conv_ens][0].source.shape[1]
      input_width = self.connections_map[conv_ens][0].source.shape[2]
      if channels < latte.config.SIMDWIDTH:
        channels = latte.config.SIMDWIDTH
      cache_line = 64
      l1_size = 32768
      l1_cacheline = int(l1_size/cache_line)
      half_l1_size = 0.5 * l1_size
      output_width = conv_ens.shape[2]
      output_height = conv_ens.shape[1]
      kernel_h=conv_ens.neurons[0,0,0].weights.shape[1]
      kernel_w=conv_ens.neurons[0,0,0].weights.shape[2]
      cacheline_needed_by_each_ofh = output_width + (channels*kernel_h*kernel_w) + (channels*kernel_h*kernel_w*latte.config.SIMDWIDTH)
      cacheline_needed_by_each_ofw = 1 + (channels*kernel_h*kernel_w) + (channels*kernel_h*kernel_w*latte.config.SIMDWIDTH)
      cacheline_needed_by_each_ifm = (kernel_h*kernel_w) + (kernel_h*kernel_w*latte.config.SIMDWIDTH)
      #print("channels=", channels, " output_width=", output_width, " kernel_h=", kernel_h, " kernel_w=", kernel_w, " cacheline_needed_ofh=", cacheline_needed_by_each_ofh, " cacheline_needed_ofw=", cacheline_needed_by_each_ofw, " cacheline_needed_ifm=", cacheline_needed_by_each_ifm, " l1_cacheline=", l1_cacheline)
      SIMDWIDTH=latte.config.SIMDWIDTH
      

      #First try to fuse CBR
      #if bias_ens is not None and net.connections_map[bias_ens][0].source.shape[0] < IFM_BLOCK_THRESHOLD and conv_ens.shape[1]  > HEIGHT_THRESHOLD and conv_ens.shape[2] > WIDTH_THRESHOLD:
      #if bias_ens is not None and conv_ens.shape[1]  > HEIGHT_THRESHOLD and conv_ens.shape[2] > WIDTH_THRESHOLD:
      if bias_ens is not None and channels == SIMDWIDTH: #should this be equal
        conv_ens.reset_prefetch(phase="forward")
        #bring ifm inside
        conv_ens.swap_loops(phase="forward", loop_vars=("i_outer", "_neuron_index_2"))
        conv_ens.swap_loops(phase="forward", loop_vars=("i_outer", "_neuron_index_3"))
        
        #perform register blocking on h and w
        #determine h_unroll_factor, w_unroll_factor
        

        h_unroll_factor = 7
        while conv_ens.shape[1] % h_unroll_factor != 0:
          h_unroll_factor -= 1
        w_unroll_factor = int(28/h_unroll_factor)
        while conv_ens.shape[2] % w_unroll_factor != 0:
          w_unroll_factor -= 1
       
        '''
        print("h is \n")
        print(conv_ens.shape[1])
        print("w is \n")
        print(conv_ens.shape[2])


        print("h unroll factor is \n")
        print(h_unroll_factor)
        print("w unroll factor is \n")
        print(w_unroll_factor)
        '''
        conv_ens.unroll(phase="forward", loop_var="_neuron_index_2", factor=h_unroll_factor, unroll_type= 1)
        conv_ens.unroll(phase="forward", loop_var="_neuron_index_3", factor=w_unroll_factor, unroll_type=1)
        relu_ens.unroll(phase="forward", loop_var="_neuron_index_2", factor=h_unroll_factor, unroll_type= 1)
        relu_ens.unroll(phase="forward", loop_var="_neuron_index_3", factor=w_unroll_factor, unroll_type=1)
        bias_ens.unroll(phase="forward", loop_var="_neuron_index_2", factor=h_unroll_factor, unroll_type = 1)
        bias_ens.unroll(phase="forward", loop_var="_neuron_index_3", factor=w_unroll_factor, unroll_type =1)
      elif bias_ens is not None and cacheline_needed_by_each_ifm < l1_cacheline:
        conv_ens.reset_prefetch(phase="forward")
        #bring ifm inside
        conv_ens.swap_loops(phase="forward", loop_vars=("i_outer", "_neuron_index_2"))
        conv_ens.swap_loops(phase="forward", loop_vars=("i_outer", "_neuron_index_3"))
        
        #perform register blocking on h and w
        #determine h_unroll_factor, w_unroll_factor
        '''
        h_unroll_factor = 7
        while conv_ens.shape[1] % h_unroll_factor != 0:
          h_unroll_factor -= 1
        w_unroll_factor = int(28/h_unroll_factor)
        while conv_ens.shape[2] % w_unroll_factor != 0:
          w_unroll_factor -= 1
        '''
        w_unroll_factor = 28
        while conv_ens.shape[2] % w_unroll_factor != 0:
          w_unroll_factor -= 1
        h_unroll_factor = int(28/w_unroll_factor)
        while conv_ens.shape[1] % h_unroll_factor != 0:
          h_unroll_factor -= 1
        '''
        w_unroll_factor = 7
        while conv_ens.shape[2] % w_unroll_factor != 0:
          w_unroll_factor -= 1
        h_unroll_factor = int(28/w_unroll_factor)
        while conv_ens.shape[1] % h_unroll_factor != 0:
          h_unroll_factor -= 1
        '''
        '''
        print("h is \n")
        print(conv_ens.shape[1])
        print("w is \n")
        print(conv_ens.shape[2])
 
 
        print("h unroll factor is \n")
        print(h_unroll_factor)
        print("w unroll factor is \n")
        print(w_unroll_factor)
        '''
        conv_ens.unroll(phase="forward", loop_var="_neuron_index_2", factor=h_unroll_factor, unroll_type= 1)
        conv_ens.unroll(phase="forward", loop_var="_neuron_index_3", factor=w_unroll_factor, unroll_type=1)
        relu_ens.unroll(phase="forward", loop_var="_neuron_index_2", factor=h_unroll_factor, unroll_type= 1)
        relu_ens.unroll(phase="forward", loop_var="_neuron_index_3", factor=w_unroll_factor, unroll_type=1)
        bias_ens.unroll(phase="forward", loop_var="_neuron_index_2", factor=h_unroll_factor, unroll_type = 1)
        bias_ens.unroll(phase="forward", loop_var="_neuron_index_3", factor=w_unroll_factor, unroll_type =1)
        '''
         w_unroll_factor = 14 # Changed by Anand to 14 so that h_unroll_factor is at least 2
        while w_unroll_factor > 0 and  conv_ens.shape[2] % w_unroll_factor != 0 :
          w_unroll_factor -= 2 # change from 1 to 2
 
        if w_unroll_factor > 0:
            h_unroll_factor = int(28/w_unroll_factor)
            while h_unroll_factor > 0 and conv_ens.shape[1] % h_unroll_factor != 0:
                h_unroll_factor -= 2
 
 
            if h_unroll_factor == 0:
                h_unroll_factor = 1
 
        if w_unroll_factor == 0:
            w_unroll_factor = 28
            h_unroll_factor = 1
 
 
 
        conv_ens.unroll(phase="forward", loop_var="_neuron_index_2", factor=h_unroll_factor, unroll_type= 1)
        conv_ens.unroll(phase="forward", loop_var="_neuron_index_3", factor=w_unroll_factor, unroll_type=1)
        relu_ens.unroll(phase="forward", loop_var="_neuron_index_2", factor=h_unroll_factor, unroll_type= 1)
        relu_ens.unroll(phase="forward", loop_var="_neuron_index_3", factor=w_unroll_factor, unroll_type=1)
        bias_ens.unroll(phase="forward", loop_var="_neuron_index_2", factor=h_unroll_factor, unroll_type = 1)
        bias_ens.unroll(phase="forward", loop_var="_neuron_index_3", factor=w_unroll_factor, unroll_type =1)
        ''' 



        if "AVX-512" in latte.config.vec_config:
          if (kernel_h == 1 and kernel_w == 1) or (conv_ens.stride >1): 
            inner_unroll_factor = 2
          else:
            inner_unroll_factor = 4
          cacheline_needed_by_each_ifm_reg_block = w_unroll_factor + (kernel_h*kernel_w*w_unroll_factor) + (kernel_h*kernel_w*latte.config.SIMDWIDTH)
          #print ("ifm_reg_blpck=", cacheline_needed_by_each_ifm_reg_block);

          #prefetch
          if h_unroll_factor == 1 and cacheline_needed_by_each_ifm_reg_block  < l1_cacheline:
            fp_pf_factor = ((kernel_h*kernel_w*w_unroll_factor))/(kernel_h*kernel_w*(SIMDWIDTH/inner_unroll_factor))
            #print ("fp_pf_factor=", fp_pf_factor)
            if fp_pf_factor > 1.0:
              fp_pf_factor = math.ceil(fp_pf_factor)
              fp_pf_loop = "i_inner"
            else:
              fp_pf_factor = ((kernel_h*kernel_w*w_unroll_factor))/(kernel_h*kernel_w)
              #print ("fp_pf_factor=", fp_pf_factor)
              if fp_pf_factor > 1.0:
                fp_pf_factor = math.ceil(fp_pf_factor)
                fp_pf_loop = "k"
              else:
                fp_pf_factor = math.ceil(((kernel_h*kernel_w*w_unroll_factor))/(kernel_h))
                #print ("fp_pf_factor=", fp_pf_factor)
                fp_pf_loop = "j"
            #conv_ens.prefetch(phase="forward", prefetch_dict_list={'value': [1, "_neuron_index_3", -3, w_unroll_factor, "_neuron_index_2", 1, 1, 0], 'inputs': [3, "i_inner", -2, w_unroll_factor, "k", 1, stride_w * w_unroll_factor, 0]})
            if w_unroll_factor == output_width:
              conv_ens.prefetch(phase="forward", prefetch_dict_list={'value': [1, "_neuron_index_3", -3, w_unroll_factor, "_neuron_index_2", 1, 1, 0], 'inputs': [3, "i_inner", -4, fp_pf_factor, fp_pf_loop, 1, 1, 2, 0], 'weights': [1, "i_inner", -4, 1, "i_outer", 1, 1, 0]})
            else:
              conv_ens.prefetch(phase="forward", prefetch_dict_list={'value': [1, "_neuron_index_3", -2, w_unroll_factor, "_neuron_index_3", 1, 1, 0], 'inputs': [3, "i_inner", -4, fp_pf_factor, fp_pf_loop, 1, 1, 2, 0], 'weights': [1, "i_inner", -4, 1, "i_outer", 1, 1, 0]})
          
          '''
          elif h_unroll_factor == 2 and cacheline_needed_by_each_ifm_reg_block  < l1_cacheline:
            if w_unroll_factor == output_width:
              conv_ens.prefetch(phase="forward", prefetch_dict_list={'value': [1, "_neuron_index_3", -3, 2*w_unroll_factor, "_neuron_index_2", 1, w_unroll_factor, 0]})
            else:
              conv_ens.prefetch(phase="forward", prefetch_dict_list={'value': [1, "_neuron_index_3", -2, 2*w_unroll_factor, "_neuron_index_3", 1, w_unroll_factor, 0]})
          '''


      elif bias_ens is not None: #try fusing B and R
        h_unroll_factor = 2
        w_unroll_factor = 2
        if bias_ens.shape[1] % h_unroll_factor == 0 and bias_ens.shape[2] % w_unroll_factor == 0 and relu_ens.shape[1] % h_unroll_factor == 0 and relu_ens.shape[2] % w_unroll_factor == 0:
          relu_ens.unroll(phase="forward", loop_var="_neuron_index_2", factor=h_unroll_factor, unroll_type= 1)
          relu_ens.unroll(phase="forward", loop_var="_neuron_index_3", factor=w_unroll_factor, unroll_type=1)
          bias_ens.unroll(phase="forward", loop_var="_neuron_index_2", factor=h_unroll_factor, unroll_type = 1)
          bias_ens.unroll(phase="forward", loop_var="_neuron_index_3", factor=w_unroll_factor, unroll_type =1)


