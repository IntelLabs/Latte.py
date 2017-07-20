/*
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
*/
#include <tbb/tbb.h>

tbb::affinity_partitioner *ap;
tbb::task_scheduler_init *scheduler_init;

class FlowGraph {
    public:
        tbb::flow::graph _graph;
        void wait_for_all() {_graph.wait_for_all();}
};

class ContinueNode {
    public:
        tbb::flow::continue_node<tbb::flow::continue_msg> *_node;
        ContinueNode(FlowGraph *g, std::function<void()> kernel) {
            _node = new tbb::flow::continue_node<tbb::flow::continue_msg> (
                g->_graph, [=](const tbb::flow::continue_msg &) {
                    kernel();
                });
        };
        void execute() {
           _node->try_put(tbb::flow::continue_msg());
        };
};

void make_edge(ContinueNode *source, ContinueNode *sink) {
    tbb::flow::make_edge(*(source->_node), *(sink->_node));
}

void parallel_for(int range_start, int range_end, std::function<void(int, int)> kernel) {
    tbb::parallel_for(tbb::blocked_range<int>(range_start, range_end), 
            [=](const tbb::blocked_range<int>& r) {
                kernel(r.begin(), r.end());
            }
    );
    //, *ap);
}
