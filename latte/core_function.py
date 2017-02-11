import numpy as np
import numbers
import itertools
import ast
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
    "INCLUDE_RUNTIME": C.SymbolRef("1" if latte.config.parallel_strategy in ["SIMPLE_LOOP"] else "0"),
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
        self.connections_map = {ensemble: [] for ensemble in self.ensembles}
        for connection in self.connections:
            self.connections_map[connection.sink].append(connection)

        in_place_buffer_map = {}

        logger.info("Initializing ensembles and synthesizing functions...")
        logger.info("Compiling functions...")

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
                       forward_body = []
                       forward_args = set()
 
                       backward_body = []
                       backward_args = set()
 
 

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

    def _gen_libxsmm_function(self, ensemble, neuron, direction):
      #print("weights:" , neuron.weights.shape)
      #print("ensamble.shape:" , ensemble.shape)
      #print("nifm:", self.connections_map[ensemble][0].source.shape[0], " ifh:", self.connections_map[ensemble][0].source.shape[1], " ifw:", self.connections_map[ensemble][0].source.shape[2])
      #print("stride:", ensemble.stride)
      #print("name:",ensemble.name)
      if direction == "forward" :
        return StringTemplate("""
      {
      libxsmm_dnn_conv_desc conv_desc;
      libxsmm_dnn_conv_handle* libxsmm_handle;
      libxsmm_dnn_buffer* libxsmm_input;
      libxsmm_dnn_buffer* libxsmm_output;
      libxsmm_dnn_filter* libxsmm_filter;
      libxsmm_dnn_err_t status; 
      unsigned int kind;
      conv_desc.N = $nImg;
      conv_desc.C = $nIfm;
      conv_desc.H = $ifh;
      conv_desc.W = $ifw;
      conv_desc.K = $nOfm;
      conv_desc.R = $kh;
      conv_desc.S = $kw;
      conv_desc.u = $stride_h;
      conv_desc.v = $stride_w;
      conv_desc.pad_h_in = 0;
      conv_desc.pad_w_in = 0;
      conv_desc.pad_h_out = 0;
      conv_desc.pad_w_out = 0;
      conv_desc.threads = omp_get_max_threads();
      conv_desc.algo = LIBXSMM_DNN_CONV_ALGO_AUTO;
      conv_desc.buffer_format = LIBXSMM_DNN_CONV_FORMAT_LIBXSMM;
      conv_desc.filter_format = LIBXSMM_DNN_CONV_FORMAT_LIBXSMM;
      conv_desc.fuse_ops = LIBXSMM_DNN_CONV_FUSE_NONE;
      conv_desc.options = LIBXSMM_DNN_CONV_OPTION_NONE;
      conv_desc.datatype_in = LIBXSMM_DNN_DATATYPE_F32;
      conv_desc.datatype_out = LIBXSMM_DNN_DATATYPE_F32;

      libxsmm_handle = libxsmm_dnn_create_conv_handle_check( conv_desc, &status );

      /* setup LIBXSMM buffers and filter */
      libxsmm_input = libxsmm_dnn_link_input_buffer_check( libxsmm_handle,  $input, LIBXSMM_DNN_CONV_FORMAT_LIBXSMM_PTR, &status );
      libxsmm_output = libxsmm_dnn_link_output_buffer_check( libxsmm_handle,  $output, LIBXSMM_DNN_CONV_FORMAT_LIBXSMM_PTR, &status );
      libxsmm_filter = libxsmm_dnn_link_filter_check( libxsmm_handle,  $filter, LIBXSMM_DNN_CONV_FORMAT_LIBXSMM_PTR, &status );

      /* bind buffers and filter to handle */
       libxsmm_dnn_bind_input_buffer( libxsmm_handle, libxsmm_input ) ;
       libxsmm_dnn_bind_output_buffer( libxsmm_handle, libxsmm_output ) ;
       libxsmm_dnn_bind_filter( libxsmm_handle, libxsmm_filter );
      # pragma omp parallel
      {
        const int tid = omp_get_thread_num();
        libxsmm_dnn_convolve_st( libxsmm_handle, LIBXSMM_DNN_CONV_KIND_FWD, 0, tid ) ;
      }
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
      , 'input': C.SymbolRef(ensemble.name + "inputs")
      , 'output': C.SymbolRef(ensemble.name+"value")
      , 'filter': C.SymbolRef(ensemble.name+"weights")})

      elif direction == "backward":
        return StringTemplate("""
      {
      libxsmm_dnn_conv_desc conv_desc;
      libxsmm_dnn_conv_handle* libxsmm_handle;
      libxsmm_dnn_buffer* libxsmm_input;
      libxsmm_dnn_buffer* libxsmm_output;
      libxsmm_dnn_filter* libxsmm_filter;
      libxsmm_dnn_err_t status; 
      conv_desc.N = $nImg;
      conv_desc.C = $nIfm;
      conv_desc.H = $ifh;
      conv_desc.W = $ifw;
      conv_desc.K = $nOfm;
      conv_desc.R = $kh;
      conv_desc.S = $kw;
      conv_desc.u = $stride_h;
      conv_desc.v = $stride_w;
      conv_desc.pad_h_in = 0;
      conv_desc.pad_w_in = 0;
      conv_desc.pad_h_out = 0;
      conv_desc.pad_w_out = 0;
      conv_desc.threads = omp_get_max_threads();
      conv_desc.algo = LIBXSMM_DNN_CONV_ALGO_AUTO;
      conv_desc.buffer_format = LIBXSMM_DNN_CONV_FORMAT_LIBXSMM;
      conv_desc.filter_format = LIBXSMM_DNN_CONV_FORMAT_LIBXSMM;
      conv_desc.fuse_ops = LIBXSMM_DNN_CONV_FUSE_NONE;
      conv_desc.options = LIBXSMM_DNN_CONV_OPTION_NONE;
      conv_desc.datatype_in = LIBXSMM_DNN_DATATYPE_F32;
      conv_desc.datatype_out = LIBXSMM_DNN_DATATYPE_F32;

      libxsmm_handle = libxsmm_dnn_create_conv_handle_check( conv_desc, &status );

      /* setup LIBXSMM buffers and filter */
      libxsmm_input = libxsmm_dnn_link_input_buffer_check( libxsmm_handle,  $input, LIBXSMM_DNN_CONV_FORMAT_LIBXSMM_PTR, &status );
      libxsmm_output = libxsmm_dnn_link_output_buffer_check( libxsmm_handle,  $output, LIBXSMM_DNN_CONV_FORMAT_LIBXSMM_PTR, &status );
      libxsmm_filter = libxsmm_dnn_link_filter_check( libxsmm_handle,  $filter, LIBXSMM_DNN_CONV_FORMAT_LIBXSMM_PTR, &status );

      /* bind buffers and filter to handle */
      libxsmm_dnn_bind_input_buffer( libxsmm_handle, libxsmm_input ) ;
      libxsmm_dnn_bind_output_buffer( libxsmm_handle, libxsmm_output ) ;
      libxsmm_dnn_bind_filter( libxsmm_handle, libxsmm_filter ) ;

    # pragma omp parallel
      {
        const int tid = omp_get_thread_num();
        libxsmm_dnn_convolve_st( libxsmm_handle, LIBXSMM_DNN_CONV_KIND_BWD, 0, tid ) ;
      }
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
      , 'input': C.SymbolRef(ensemble.name + "grad_inputs")
      , 'output': C.SymbolRef(ensemble.name+"grad")
      , 'filter': C.SymbolRef(ensemble.name+"weights")})

      else:
        return StringTemplate("""
    {
    libxsmm_dnn_conv_desc conv_desc;
    libxsmm_dnn_conv_handle* libxsmm_handle;
    libxsmm_dnn_buffer* libxsmm_input;
    libxsmm_dnn_buffer* libxsmm_output;
    libxsmm_dnn_filter* libxsmm_filter;
    libxsmm_dnn_err_t status; 
    conv_desc.N = $nImg;
    conv_desc.C = $nIfm;
    conv_desc.H = $ifh;
    conv_desc.W = $ifw;
    conv_desc.K = $nOfm;
    conv_desc.R = $kh;
    conv_desc.S = $kw;
    conv_desc.u = $stride_h;
    conv_desc.v = $stride_w;
    conv_desc.pad_h_in = 0;
    conv_desc.pad_w_in = 0;
    conv_desc.pad_h_out = 0;
    conv_desc.pad_w_out = 0;
    conv_desc.threads = omp_get_max_threads();
    conv_desc.algo = LIBXSMM_DNN_CONV_ALGO_AUTO;
    conv_desc.buffer_format = LIBXSMM_DNN_CONV_FORMAT_LIBXSMM;
    conv_desc.filter_format = LIBXSMM_DNN_CONV_FORMAT_LIBXSMM;
    conv_desc.fuse_ops = LIBXSMM_DNN_CONV_FUSE_NONE;
    conv_desc.options = LIBXSMM_DNN_CONV_OPTION_NONE;
    conv_desc.datatype_in = LIBXSMM_DNN_DATATYPE_F32;
    conv_desc.datatype_out = LIBXSMM_DNN_DATATYPE_F32;

    libxsmm_handle = libxsmm_dnn_create_conv_handle_check( conv_desc, &status );

    /* setup LIBXSMM buffers and filter */
    libxsmm_input = libxsmm_dnn_link_input_buffer_check( libxsmm_handle,  $input, LIBXSMM_DNN_CONV_FORMAT_LIBXSMM_PTR, &status );
    libxsmm_output = libxsmm_dnn_link_output_buffer_check( libxsmm_handle,  $output, LIBXSMM_DNN_CONV_FORMAT_LIBXSMM_PTR, &status );
    libxsmm_filter = libxsmm_dnn_link_filter_check( libxsmm_handle,  $filter, LIBXSMM_DNN_CONV_FORMAT_LIBXSMM_PTR, &status );

    /* bind buffers and filter to handle */
    libxsmm_dnn_bind_input_buffer( libxsmm_handle, libxsmm_input ) ;
    libxsmm_dnn_bind_output_buffer( libxsmm_handle, libxsmm_output ) ;
    libxsmm_dnn_bind_filter( libxsmm_handle, libxsmm_filter ) ;
  # pragma omp parallel
    {
      const int tid = omp_get_thread_num();
      libxsmm_dnn_convolve_st( libxsmm_handle, LIBXSMM_DNN_CONV_KIND_UPD, 0, tid );
    }
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

          _, transposed_buffers = vectorizer.vectorize_loop(func_def,
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
          func_def = analyzer.type_infer(func_def)
          #if direction in ensemble.vectorize_info:
            # RAJ hack here
          #  func_def, transposed_buffers = vectorizer.vectorize_loop(func_def, 
          #          ensemble.vectorize_info[direction][0])
          #  pre_trans, post_trans = self._gen_transposes(transposed_buffers)

          #  for buffer_name, trans_dim in transposed_buffers.items():
          #      curr_body = []
          #      shape = self.buffers[buffer_name].shape
          #      shape_str = "".join("[{}]".format(d) for d in shape)

          #      args.append(ast.arg(buffer_name + "_transposed", None))
          #      self.buffers[buffer_name + "_transposed"] = util.zeros(shape, np.float32)
          #func_def.defn[0].pre_trans = pre_trans
          # func_def = optimizer.propogate_constants(func_def) [], None)
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
                
            for loop in func_def.defn:
                #loop.init.left.type = ctypes.c_int()
                inner = loop.body[0]
                inner.init.left.type = ctypes.c_int()
                inner = inner.body[0]   
                inner.init.left.type = ctypes.c_int()
                inner = inner.body[0]
                inner.init.left.type = ctypes.c_int() 


        # Seed the argument types as pointers for type inference
        for arg in func_def.params:
           buf = self.buffers[arg.name]
           arg.type = np.ctypeslib.ndpointer(buf.dtype, buf.ndim, buf.shape)()
        # Basic type inference and constant propogation
        func_def = analyzer.type_infer(func_def)
        func_def = optimizer.propogate_constants(func_def)
        

        if latte.config.codegen_strategy == "AUTOVEC":
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

          if direction == "forward" and direction in ensemble.unroll_2_info and ensemble.unroll_2_info[direction]:
            (unroll_var_2, unroll_factor_2) = ensemble.unroll_2_info[direction]
            unroller.unroll_loop(func_def, unroll_var_2, unroll_factor_2)
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
                        prefetch_type, enclosing_loop_var, dim, prefetch_count, prefetch_offset, prefetch_dest_loop, prefetch_init_loop, prefetch_loop_var, prefetch_multiplier, prefetch_constant, cacheline_hint = value
                        prefetcher.insert_strided_prefetches(func_def, field, prefetch_type, enclosing_loop_var, dim, prefetch_count, prefetch_offset, prefetch_dest_loop, prefetch_init_loop, prefetch_loop_var, prefetch_multiplier, prefetch_constant, cacheline_hint)
                    elif value[0] == 3:
                        prefetch_type, enclosing_loop_var, dim, prefetch_count, prefetch_loop_var, prefetch_multiplier, prefetch_constant, cacheline_hint = value
                        prefetcher.insert_simple_hoist_prefetches(func_def, field, prefetch_type, enclosing_loop_var, dim, prefetch_count, prefetch_loop_var, prefetch_multiplier, prefetch_constant, cacheline_hint)
          # drop loops that iterate for one iteration only and constant propagate indices and hoist address computations
          func_def = loopsimplifier.simplify_loops(func_def)
          #func_def = optimizer.propogate_constants(func_def)

        else: #GEMM formulation
          func_def = transformers.pattern_match_gemm(func_def)
          raise NotImplementedError("GEMM formulation is not complete yet")

        assert isinstance(func_def.defn[0], C.For)
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

