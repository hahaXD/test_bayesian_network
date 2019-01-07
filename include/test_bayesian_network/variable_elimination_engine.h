#include <vector>

namespace test_bayesian_network {
class VariableEliminationEngine {
public:
  VariableEliminationEngine(std::vector<Node *> network_nodes_to_compile);

private:
  std::vector<Node *> network_nodes_to_compile_;
};
} // namespace test_bayesian_network
