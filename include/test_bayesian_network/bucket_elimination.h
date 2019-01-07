#ifndef BUCKET_ELIMINATION_H
#define BUCKET_ELIMINATION_H
#include <test_bayesian_network/circuit_factor.h>
#include <test_bayesian_network/circuit_manager.h>
#include <test_bayesian_network/variable.h>
#include <unordered_map>
#include <vector>

namespace test_bayesian_network {
class FactorForElimination {
public:
  FactorForElimination(
      std::unique_ptr<CircuitFactor> circuit_factor,
      std::vector<Variable *> variables_under_elimination_order);

  static std::unique_ptr<FactorForElimination>
  GetFactorForEliminationFromTotalEliminationOrder(
      std::unique_ptr<CircuitFactor> circuit_factor,
      const std::vector<Variable *> &total_elimination_order);

  std::unique_ptr<FactorForElimination>
  SumOut(test_circuit::CircuitManager *circuit_manager);

  std::unique_ptr<FactorForElimination>
  Multiply(const FactorForElimination &other,
           test_circuit::CircuitManager *circuit_manager,
           const std::unordered_map<NodeSize, NodeSize>
               &variable_index_to_elimination_order);

  CircuitFactor *circuit_factor() const { return circuit_factor_.get(); }

  Variable *NextVariableToEliminate() const;

  std::unique_ptr<CircuitFactor> ReleaseCircuitFactor() {
    std::unique_ptr<CircuitFactor> result = nullptr;
    result.swap(circuit_factor_);
    return result;
  }

private:
  std::unique_ptr<CircuitFactor> circuit_factor_;
  std::vector<Variable *> variables_under_elimination_order_;
};

class BucketElimination {
public:
  BucketElimination(std::vector<CircuitFactor *> circuit_factors,
                    std::vector<Variable *> elimination_order,
                    test_circuit::CircuitManager *circuit_manager);

  std::unique_ptr<CircuitFactor> Run();

private:
  void AddFactorToBucket(
      std::unique_ptr<FactorForElimination> factor_to_add,
      std::vector<std::vector<std::unique_ptr<FactorForElimination>>> *buckets);
  std::unordered_map<NodeSize, NodeSize> variable_index_to_elimination_order_;
  std::vector<Variable *> elimination_order_;
  std::vector<CircuitFactor *> circuit_factors_;
  test_circuit::CircuitManager *circuit_manager_;
};
} // namespace test_bayesian_network
#endif
