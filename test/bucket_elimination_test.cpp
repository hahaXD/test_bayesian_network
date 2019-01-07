#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <test_bayesian_network/bucket_elimination.h>
#include <test_bayesian_network/circuit_factor.h>
#include <test_bayesian_network/circuit_manager.h>
#include <test_bayesian_network/circuit_node.h>
#include <test_bayesian_network/variable.h>
namespace test_bayesian_network {

TEST(BUCKET_ELIMINATION_TEST, ADD_FACTOR_TO_BUCKET) {
  std::vector<std::unique_ptr<Variable>> variables;
  for (auto i = 0; i < 3; ++i) {
    variables.push_back(std::make_unique<Variable>(i, 3, variable_type::hidden));
  }
  test_circuit::CircuitManager cm;
  // factors for a
  std::vector<Variable *> variables_for_a = {variables[0].get()};
  std::vector<test_circuit::Node *> factor_nodes_a;
  for (auto i = 0; i < 3; ++i) {
    factor_nodes_a.push_back(
        cm.NewParameterTerminalNode({}, variables[0].get(), {}, i));
  }
  CircuitFactor cf_a(std::move(variables_for_a), std::move(factor_nodes_a));
  // factors for b|a
  std::vector<Variable *> variables_for_ba = {variables[0].get(),
                                              variables[1].get()};
  std::vector<test_circuit::Node *> factor_nodes_ba;
  for (auto i = 0; i < 9; ++i) {
    auto variable_configuration =
        CircuitFactor::GetVariableConfigurationFromEntryIndex(variables_for_ba,
                                                              i);
    factor_nodes_ba.push_back(cm.NewParameterTerminalNode(
        {variables[0].get()}, variables[1].get(), {variable_configuration[0]},
        variable_configuration[1]));
  }
  CircuitFactor cf_ba(std::move(variables_for_ba), std::move(factor_nodes_ba));
  // factors for c|b
  std::vector<Variable *> variables_for_cb = {variables[1].get(),
                                              variables[2].get()};
  std::vector<test_circuit::Node *> factor_nodes_cb;
  for (auto i = 0; i < 9; ++i) {
    auto variable_configuration =
        CircuitFactor::GetVariableConfigurationFromEntryIndex(variables_for_cb,
                                                              i);
    factor_nodes_cb.push_back(cm.NewParameterTerminalNode(
        {variables[1].get()}, variables[2].get(), {variable_configuration[0]},
        variable_configuration[1]));
  }
  CircuitFactor cf_cb(std::move(variables_for_cb), std::move(factor_nodes_cb));

  std::vector<CircuitFactor *> circuit_factors = {&cf_a, &cf_ba, &cf_cb};
  std::vector<Variable *> elimination_order = {
      variables[0].get(), variables[1].get(), variables[2].get()};
  BucketElimination be(std::move(circuit_factors), std::move(elimination_order),
                       &cm);
  auto result = be.Run();
  EXPECT_TRUE(result->variables().empty());
  test_circuit::Node *result_node =
      result->GetNodeFromVariableConfiguration({});
  EXPECT_EQ(result_node->type(), test_circuit::node_type::sum);
  EXPECT_EQ(result_node->children().size(), (size_t)3);
  test_circuit::Node *case_c0 = result_node->children()[0];
  EXPECT_EQ(case_c0->type(), test_circuit::node_type::sum);
  test_circuit::Node *case_c0b0 = case_c0->children()[0];
  EXPECT_EQ(case_c0b0->type(), test_circuit::node_type::product);
  test_circuit::Node *case_c0b0_parameter = case_c0b0->children()[0];
  EXPECT_EQ(case_c0b0_parameter->type(), test_circuit::node_type::parameter);
  test_circuit::Node *case_b0 = case_c0b0->children()[1];
  EXPECT_EQ(case_b0->type(), test_circuit::node_type::sum);
  test_circuit::Node *case_a0b0 = case_b0->children()[0];
  EXPECT_EQ(case_a0b0->type(), test_circuit::node_type::product);
  test_circuit::Node *case_a0_parameter = case_a0b0->children()[0];
  test_circuit::Node *case_a0b0_parameter = case_a0b0->children()[1];
  EXPECT_EQ(case_a0_parameter->type(), test_circuit::node_type::parameter);
  EXPECT_EQ(case_a0_parameter->get_parameter_terminal_node()->child_variable(),
            variables[0].get());
  EXPECT_EQ(
      case_a0_parameter->get_parameter_terminal_node()->child_configuration(),
      0);
  EXPECT_EQ(
      case_a0b0_parameter->get_parameter_terminal_node()->child_variable(),
      variables[1].get());
  EXPECT_EQ(
      case_a0b0_parameter->get_parameter_terminal_node()->child_configuration(),
      0);
}
} // namespace test_bayesian_network
