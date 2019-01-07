#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <test_bayesian_network/circuit_manager.h>
#include <test_bayesian_network/circuit_node.h>
#include <test_bayesian_network/node.h>
#include <test_bayesian_network/test_bayesian_network.h>
#include <test_bayesian_network/variable.h>
#include <test_bayesian_network/ve_test_circuit_compiler.h>

namespace test_bayesian_network {
namespace {
std::unique_ptr<TestBayesianNetwork> ConstructNetwork1() {
  std::vector<std::unique_ptr<Variable>> evidence_variables;
  NodeSize next_variable_index = 0;
  auto variable_root = std::make_unique<Variable>(next_variable_index++, 3,
                                                  variable_type::evidence);
  auto variable_a1 = std::make_unique<Variable>(next_variable_index++, 3,
                                                variable_type::hidden);
  auto variable_a2 = std::make_unique<Variable>(next_variable_index++, 3,
                                                variable_type::hidden);
  auto variable_a3 = std::make_unique<Variable>(next_variable_index++, 3,
                                                variable_type::hidden);
  auto variable_b1 = std::make_unique<Variable>(next_variable_index++, 3,
                                                variable_type::hidden);
  auto variable_b2 = std::make_unique<Variable>(next_variable_index++, 3,
                                                variable_type::hidden);
  auto variable_b3 = std::make_unique<Variable>(next_variable_index++, 3,
                                                variable_type::hidden);
  auto variable_query = std::make_unique<Variable>(next_variable_index++, 3,
                                                   variable_type::query);
  std::vector<std::unique_ptr<Node>> nodes_in_network;
  // root_node
  const size_t root_node_index = nodes_in_network.size();
  std::vector<Node *> root_node_parent = {};
  nodes_in_network.emplace_back(
      std::make_unique<Node>(std::move(variable_root),
                             std::move(root_node_parent), node_type::regular));
  // a1_node
  const size_t a1_node_index = nodes_in_network.size();
  std::vector<Node *> a1_node_parent = {
      nodes_in_network[root_node_index].get()};
  nodes_in_network.emplace_back(std::make_unique<Node>(
      std::move(variable_a1), std::move(a1_node_parent), node_type::test));
  // a2_node
  const size_t a2_node_index = nodes_in_network.size();
  std::vector<Node *> a2_node_parent = {nodes_in_network[a1_node_index].get()};
  nodes_in_network.emplace_back(std::make_unique<Node>(
      std::move(variable_a2), std::move(a2_node_parent), node_type::regular));
  // a3_node
  const size_t a3_node_index = nodes_in_network.size();
  std::vector<Node *> a3_node_parent = {nodes_in_network[a2_node_index].get()};
  nodes_in_network.emplace_back(std::make_unique<Node>(
      std::move(variable_a3), std::move(a3_node_parent), node_type::test));
  // b1_node
  const size_t b1_node_index = nodes_in_network.size();
  std::vector<Node *> b1_node_parent = {
      nodes_in_network[root_node_index].get()};
  nodes_in_network.emplace_back(std::make_unique<Node>(
      std::move(variable_b1), std::move(b1_node_parent), node_type::test));
  // b2_node
  const size_t b2_node_index = nodes_in_network.size();
  std::vector<Node *> b2_node_parent = {nodes_in_network[b1_node_index].get()};
  nodes_in_network.emplace_back(std::make_unique<Node>(
      std::move(variable_b2), std::move(b2_node_parent), node_type::regular));
  // b3_node
  const size_t b3_node_index = nodes_in_network.size();
  std::vector<Node *> b3_node_parent = {nodes_in_network[b2_node_index].get()};
  nodes_in_network.emplace_back(std::make_unique<Node>(
      std::move(variable_b3), std::move(b3_node_parent), node_type::test));
  // query_node
  std::vector<Node *> query_node_parent = {
      nodes_in_network[a3_node_index].get(),
      nodes_in_network[b3_node_index].get()};
  nodes_in_network.emplace_back(
      std::make_unique<Node>(std::move(variable_query),
                             std::move(query_node_parent), node_type::test));
  return std::make_unique<TestBayesianNetwork>(std::move(nodes_in_network));
}
} // namespace
TEST(VE_TEST_CIRCUIT_COMPILER_TEST, SIMPLE_TEST) {
  auto input_network = ConstructNetwork1();
  test_circuit::CircuitManager circuit_manager;
  VeTestCircuitCompiler vtcc(input_network.get(), &circuit_manager);
  auto result = vtcc.Run();
  EXPECT_EQ(result->children().size(), (size_t)3);
}
} // namespace test_bayesian_network
