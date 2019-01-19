#include <iostream>
#include <string>

#include <test_bayesian_network/circuit_manager.h>
#include <test_bayesian_network/test_bayesian_network.h>
#include <test_bayesian_network/ve_test_circuit_compiler.h>

namespace {
using test_bayesian_network::Node;
using test_bayesian_network::TestBayesianNetwork;
using test_bayesian_network::VeTestCircuitCompiler;
using test_bayesian_network::test_circuit::CircuitManager;
using test_bayesian_network::test_circuit::ZNode;
} // namespace

int main(int argc, const char *argv[]) {
  if (argc <= 2) {
    std::cout << "Usage: tbn_filename output_file_prefix" << std::endl;
    exit(1);
  }
  std::string bn_filename(argv[1]);
  std::string output_file_prefix(argv[2]);
  auto tbn = TestBayesianNetwork::ParseTestBayesianNetworkFromNetFile(
      bn_filename.c_str());
  std::string tac_filename = output_file_prefix + ".tac";
  std::string lmap_filename = output_file_prefix + ".lmap";
  CircuitManager cm;
  auto compiler = VeTestCircuitCompiler(tbn.get(), &cm);
  std::vector<Node *> test_node_order;
  for (Node *cur_node : tbn->nodes()) {
    if (cur_node->type() == test_bayesian_network::node_type::test) {
      test_node_order.push_back(cur_node);
    }
  }
  ZNode *result = compiler.Run(test_node_order);
  CircuitManager::SaveAsTacFile(result, tac_filename.c_str(),
                                lmap_filename.c_str());
}
