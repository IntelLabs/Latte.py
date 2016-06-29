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
