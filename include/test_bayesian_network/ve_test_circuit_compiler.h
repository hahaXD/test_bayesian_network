#ifndef VE_TEST_CIRCUIT_COMPILER_H
#define VE_TEST_CIRCUIT_COMPILER_H
#include <test_bayesian_network/circuit_manager.h>
#include <test_bayesian_network/circuit_node.h>
#include <test_bayesian_network/test_bayesian_network.h>

namespace test_bayesian_network {
class VeTestCircuitCompiler {
public:
  VeTestCircuitCompiler(TestBayesianNetwork *input_network,
                        test_circuit::CircuitManager *circuit_manager);
  test_circuit::ZNode *Run() const;

private:
  TestBayesianNetwork *input_network_;
  test_circuit::CircuitManager *circuit_manager_;
};
} // namespace test_bayesian_network
#endif
