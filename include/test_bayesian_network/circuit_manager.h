#ifndef CIRCUIT_MANAGER_H
#define CIRCUIT_MANAGER_H
#include <list>
#include <memory>
#include <unordered_map>

#include <test_bayesian_network/circuit_factor.h>
#include <test_bayesian_network/circuit_node.h>
#include <test_bayesian_network/type.h>

namespace test_bayesian_network {
namespace test_circuit {

class CircuitManager {
public:
  CircuitManager();
  Node *NewProductNode(std::vector<Node *> children);
  Node *NewSumNode(std::vector<Node *> children);
  Node *NewTestNode(std::vector<Node *> children);
  Node *NewVariableTerminalNode(Variable *variable, DomainSize value);
  Node *NewParameterTerminalNode(std::vector<Variable *> parent_variables,
                                 Variable *child_variable,
                                 std::vector<DomainSize> parent_configurations,
                                 DomainSize child_configuraiton);
  Node *NewTestThresholdParameterTerminalNode(
      std::vector<Variable *> parent_variables, Variable *child_variable,
      std::vector<DomainSize> parent_configurations);
  Node *NewTestProbabilityParameterTerminalNode(
      bool test_result, std::vector<Variable *> parent_variables,
      Variable *child_variable, std::vector<DomainSize> parent_configurations,
      DomainSize child_configuraiton);
  Node *NewZNode(std::vector<Node *> children);
  static void SaveAsTacFile(Node *root_node, const char *tac_filename,
                            const char *lmap_filename);
  static std::vector<Node *> SerializeNodes(Node *root_node);

private:
  std::list<std::unique_ptr<Node>> node_cache_;
  std::unordered_map<NodeSize, std::vector<Node *>> unique_variable_nodes_;
  std::unordered_map<NodeSize, CircuitFactor> unique_threshold_nodes_;
  uintmax_t next_node_id_;
};
} // namespace test_circuit
} // namespace test_bayesian_network
#endif
