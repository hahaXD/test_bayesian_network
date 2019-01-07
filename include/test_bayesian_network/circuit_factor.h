#ifndef CIRCUIT_FACTOR_H
#define CIRCUIT_FACTOR_H
#include <memory>
#include <test_bayesian_network/circuit_manager.h>
#include <test_bayesian_network/circuit_node.h>
#include <vector>

namespace test_bayesian_network {
class CircuitFactor {
public:
  // The variables must be sorted according to the index order
  CircuitFactor(std::vector<Variable *> variables,
                std::vector<test_circuit::Node *> factor_nodes);
  static size_t GetEntryIndexFromVariableConfiguration(
      const std::vector<Variable *> &variables,
      const std::vector<DomainSize> &variable_config);
  static std::vector<DomainSize> GetVariableConfigurationFromEntryIndex(
      const std::vector<Variable *> &variables, size_t entry_index);
  static bool CompareVariableForSorting(Variable* first, Variable* second);
  test_circuit::Node *GetNodeFromVariableConfiguration(
      const std::vector<DomainSize> &variable_config) const;
  std::unique_ptr<CircuitFactor>
  Multiply(const CircuitFactor &other,
           test_circuit::CircuitManager *circuit_manager) const;
  std::unique_ptr<CircuitFactor>
  SumOut(Variable *sum_out_variable,
         test_circuit::CircuitManager *circuit_manager) const;
  const std::vector<Variable *> &variables() const { return variables_; }

  const std::vector<test_circuit::Node *> factor_nodes() const {
    return factor_nodes_;
  }


private:
  std::vector<Variable *> variables_;
  std::vector<test_circuit::Node *> factor_nodes_;
};
} // namespace test_bayesian_network
#endif
