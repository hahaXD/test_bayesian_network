#ifndef TEST_BAYESIAN_NETWORK_H
#define TEST_BAYESIAN_NETWORK_H
#include <memory>
#include <string>
#include <test_bayesian_network/node.h>
#include <test_bayesian_network/type.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace test_bayesian_network {
namespace util {
std::vector<std::string> TopologicalSortNodes(
    const std::unordered_map<std::string, std::unordered_set<std::string>>
        &parents);
}
class TestBayesianNetwork {
public:
  // Managed Nodes
  TestBayesianNetwork(std::vector<std::unique_ptr<Node>> nodes);
  // Borrowed Nodes
  TestBayesianNetwork(std::vector<Node *> nodes);
  // Input functions
  static std::unique_ptr<TestBayesianNetwork>
  ParseTestBayesianNetworkFromNetFile(const char *filename);

  // unselected_testing_nodes does not contain test_node
  std::unique_ptr<TestBayesianNetwork> PrunedNetworkForTesting(
      Node *test_node,
      std::unordered_set<Node *> unselected_testing_nodes) const;
  std::unique_ptr<TestBayesianNetwork>
  PrunedNetworkForQuery(Node *query_node) const;
  // Gets an elimination order
  std::vector<Variable *>
  GetEliminationOrder(const std::vector<Node *> &nodes_to_query) const;

  const std::vector<Node *>& nodes() const;

private:
  // topological sorted nodes
  std::vector<std::unique_ptr<Node>> unique_nodes_;
  std::vector<Node *> nodes_;
  // prune network helper
  std::unique_ptr<TestBayesianNetwork>
  PruneNetwork(const std::unordered_set<Node *> &query_nodes,
               std::unordered_set<Node *> unselected_testing_nodes) const;
};
} // namespace test_bayesian_network
#endif
