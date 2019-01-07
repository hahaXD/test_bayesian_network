#include <test_bayesian_network/circuit_terminal_node_look_up_table.h>
#include <vector>

namespace test_bayesian_network {
namespace test_circuit {
namespace util {
CircuitTerminalNodeLookUpTable::CircuitTerminalNodeLookUpTable(
    std::unordered_map<NodeSize, std::vector<double>>
        variable_terminal_parameters,
    std::unordered_map<NodeSize, std::vector<double>> cpt_parameters,
    std::unordered_map<NodeSize, std::vector<double>> positive_cpt_parameters,
    std::unordered_map<NodeSize, std::vector<double>> negative_cpt_parameters,
    std::unordered_map<NodeSize, std::vector<double>> test_threshold_parameters)
    : variable_terminal_parameters_(std::move(variable_terminal_parameters)),
      cpt_parameters_(std::move(cpt_parameters)),
      positive_cpt_parameters_(std::move(positive_cpt_parameters)),
      negative_cpt_parameters_(std::move(negative_cpt_parameters)),
      test_threshold_parameters_(std::move(test_threshold_parameters)) {}

double CircuitTerminalNodeLookUpTable::GetValue(
    test_bayesian_network::test_circuit::Node *terminal_node) {
  return 0.0;
}
} // namespace util
} // namespace test_circuit
} // namespace test_bayesian_network
