#ifndef NODE_H
#define NODE_H
#include <memory>
#include <test_bayesian_network/circuit_factor.h>
#include <test_bayesian_network/type.h>
#include <test_bayesian_network/variable.h>
#include <vector>

namespace test_bayesian_network {
enum node_type { regular = 'r', test = 't' };

class Node {
public:
  Node(std::unique_ptr<Variable> variable, std::vector<Node *> parents,
       node_type type)
      : variable_(std::move(variable)), parents_(std::move(parents)),
        circuit_factor_(nullptr), type_(type) {}
  Variable *variable() const { return variable_.get(); }
  const std::vector<Node *> &parents() const { return parents_; }
  node_type type() const { return type_; }
  CircuitFactor *GetCircuitFactor() { return circuit_factor_.get(); }
  void SetCircuitFactor(std::unique_ptr<CircuitFactor> circuit_factor) {
    circuit_factor_.swap(circuit_factor);
  }
  test_bayesian_network::variable_type variable_type() const {
    return variable_->type();
  }

private:
  std::unique_ptr<Variable> variable_;
  std::vector<Node *> parents_;
  std::unique_ptr<CircuitFactor> circuit_factor_;
  node_type type_;
};

} // namespace test_bayesian_network
#endif
