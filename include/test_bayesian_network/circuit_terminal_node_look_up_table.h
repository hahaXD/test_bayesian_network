#ifndef CIRCUIT_TERMINAL_NODE_LOOK_UP_TABLE_H
#define CIRCUIT_TERMINAL_NODE_LOOK_UP_TABLE_H
#include <unordered_map>
#include <vector>

#include <test_bayesian_network/circuit_node.h>
#include <test_bayesian_network/type.h>

namespace test_bayesian_network {
namespace test_circuit {
namespace util {
class CircuitTerminalNodeLookUpTable {
public:
  CircuitTerminalNodeLookUpTable(
      std::unordered_map<NodeSize, std::vector<double>>
          variable_terminal_parameters,
      std::unordered_map<NodeSize, std::vector<double>> cpt_parameters,
      std::unordered_map<NodeSize, std::vector<double>> positive_cpt_parameters,
      std::unordered_map<NodeSize, std::vector<double>> negative_cpt_parameters,
      std::unordered_map<NodeSize, std::vector<double>>
          test_threshold_parameters);
  virtual double
  GetValue(test_bayesian_network::test_circuit::Node *terminal_node);
  virtual ~CircuitTerminalNodeLookUpTable() = default;

private:
  std::unordered_map<NodeSize, std::vector<double>>
      variable_terminal_parameters_;
  std::unordered_map<NodeSize, std::vector<double>> cpt_parameters_;
  std::unordered_map<NodeSize, std::vector<double>> positive_cpt_parameters_;
  std::unordered_map<NodeSize, std::vector<double>> negative_cpt_parameters_;
  std::unordered_map<NodeSize, std::vector<double>> test_threshold_parameters_;
};
} // namespace util
} // namespace test_circuit
} // namespace test_bayesian_network
#endif
