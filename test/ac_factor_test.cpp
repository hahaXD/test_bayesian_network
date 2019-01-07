#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>

#include <test_bayesian_network/circuit_factor.h>
#include <test_bayesian_network/circuit_manager.h>
#include <test_bayesian_network/variable.h>

namespace test_bayesian_network {
namespace test_circuit {

TEST(CIRCUIT_FACTOR_TEST, GET_ENTRY_INDEX_FROM_VARIABLE_CONFIGURATION) {
  std::vector<std::unique_ptr<Variable>> variables;
  for (auto i = 1; i <= 3; ++i) {
    variables.push_back(
        std::make_unique<Variable>(i, i, variable_type::evidence));
  }
  EXPECT_EQ(CircuitFactor::GetEntryIndexFromVariableConfiguration(
                {variables[0].get(), variables[1].get(), variables[2].get()},
                {0, 0, 0}),
            (size_t)0);
  EXPECT_EQ(CircuitFactor::GetEntryIndexFromVariableConfiguration(
                {variables[0].get(), variables[1].get(), variables[2].get()},
                {0, 1, 2}),
            (size_t)5);
}

TEST(CIRCUIT_FACTOR_TEST, GET_VARIABLE_CONFIGURATION_FROM_ENTRY_INDEX_TEST) {
  std::vector<std::unique_ptr<Variable>> variables;
  for (auto i = 1; i <= 3; ++i) {
    variables.push_back(
        std::make_unique<Variable>(i, i, variable_type::evidence));
  }
  const std::vector<DomainSize> expected_conf_0 = {0, 0, 0};
  EXPECT_EQ(
      CircuitFactor::GetVariableConfigurationFromEntryIndex(
          {variables[0].get(), variables[1].get(), variables[2].get()}, 0),
      expected_conf_0);
  const std::vector<DomainSize> expected_conf_5 = {0, 1, 2};
  EXPECT_EQ(
      CircuitFactor::GetVariableConfigurationFromEntryIndex(
          {variables[0].get(), variables[1].get(), variables[2].get()}, 5),
      expected_conf_5);
}

TEST(CIRCUIT_FACTOR_TEST, CONSTRUCT_CIRCUIT_FACTOR_TEST) {
  std::vector<std::unique_ptr<Variable>> variables;
  size_t factor_size = 1;
  for (auto i = 1; i <= 3; ++i) {
    variables.push_back(
        std::make_unique<Variable>(i, i, variable_type::evidence));
    factor_size *= variables.back()->domain_size();
  }
  std::vector<Node *> factor;
  factor.reserve(factor_size);
  CircuitManager circuit_manager;
  for (size_t i = 0; i < factor_size; ++i) {
    factor.push_back(
        circuit_manager.NewParameterTerminalNode({}, nullptr, {}, i));
  }
  CircuitFactor cf({variables[0].get(), variables[1].get(), variables[2].get()},
                   factor);
  Node *node_at_000 = cf.GetNodeFromVariableConfiguration({0, 0, 0});
  EXPECT_EQ(node_at_000->type(), node_type::parameter);
  EXPECT_EQ(node_at_000->get_parameter_terminal_node()->child_configuration(),
            0);
  Node *node_at_012 = cf.GetNodeFromVariableConfiguration({0, 1, 2});
  EXPECT_EQ(node_at_012->type(), node_type::parameter);
  EXPECT_EQ(node_at_012->get_parameter_terminal_node()->child_configuration(),
            5);
}

TEST(CIRCUIT_FACTOR_TEST, SUM_OUT_CIRCUIT_FACTOR_TEST) {
  std::vector<std::unique_ptr<Variable>> variables;
  size_t factor_size = 1;
  for (auto i = 1; i <= 3; ++i) {
    variables.push_back(
        std::make_unique<Variable>(i, i, variable_type::evidence));
    factor_size *= variables.back()->domain_size();
  }
  std::vector<Node *> factor;
  factor.reserve(factor_size);
  CircuitManager circuit_manager;
  for (size_t i = 0; i < factor_size; ++i) {
    factor.push_back(
        circuit_manager.NewParameterTerminalNode({}, nullptr, {}, i));
  }
  CircuitFactor cf({variables[0].get(), variables[1].get(), variables[2].get()},
                   factor);
  auto sumout_0 = cf.SumOut(variables[0].get(), &circuit_manager);
  const std::vector<Variable *> expected_variables_0 = {variables[1].get(),
                                                        variables[2].get()};
  EXPECT_EQ(sumout_0->variables(), expected_variables_0);
  // variable[0] has domain size 1, therefore the factor_size is the same as cf.
  for (size_t i = 0; i < factor_size; ++i) {
    auto cf_conf = CircuitFactor::GetVariableConfigurationFromEntryIndex(
        cf.variables(), i);
    auto sumout_0_conf = CircuitFactor::GetVariableConfigurationFromEntryIndex(
        sumout_0->variables(), i);
    EXPECT_EQ(cf.GetNodeFromVariableConfiguration(cf_conf),
              sumout_0->GetNodeFromVariableConfiguration(sumout_0_conf));
  }
  auto sumout_1 = cf.SumOut(variables[1].get(), &circuit_manager);
  const std::vector<Variable *> expected_variables_1 = {variables[0].get(),
                                                        variables[2].get()};
  EXPECT_EQ(sumout_1->variables(), expected_variables_1);
  // Checks {0,1} it should be {0,0,1} + {0,1,1}
  Node *sumout_1_01 = sumout_1->GetNodeFromVariableConfiguration({0, 1});
  EXPECT_EQ(sumout_1_01->type(), node_type::sum);
  EXPECT_EQ(sumout_1_01->children()[0],
            cf.GetNodeFromVariableConfiguration({0, 0, 1}));
  EXPECT_EQ(sumout_1_01->children()[1],
            cf.GetNodeFromVariableConfiguration({0, 1, 1}));
  auto sumout_2 = cf.SumOut(variables[2].get(), &circuit_manager);
  const std::vector<Variable *> expected_variables_2 = {variables[0].get(),
                                                        variables[1].get()};
  EXPECT_EQ(sumout_2->variables(), expected_variables_2);
  // checks {0,1} should be {0,1,0} + {0,1,1} + {0,1,2}
  Node *sumout_2_01 = sumout_2->GetNodeFromVariableConfiguration({0, 1});
  EXPECT_EQ(sumout_2_01->type(), node_type::sum);
  ASSERT_EQ(sumout_2_01->children().size(), (size_t)3);
  EXPECT_EQ(sumout_2_01->children()[0],
            cf.GetNodeFromVariableConfiguration({0, 1, 0}));
  EXPECT_EQ(sumout_2_01->children()[1],
            cf.GetNodeFromVariableConfiguration({0, 1, 1}));
  EXPECT_EQ(sumout_2_01->children()[2],
            cf.GetNodeFromVariableConfiguration({0, 1, 2}));
}

TEST(CIRCUIT_FACTOR_TEST, MULTIPLY_CIRCUIT_FACTOR_TEST) {
  std::vector<std::unique_ptr<Variable>> variables;
  for (auto i = 1; i <= 6; ++i) {
    variables.push_back(
        std::make_unique<Variable>(i, i, variable_type::evidence));
  }
  std::vector<Variable *> first_variables;
  std::vector<Variable *> second_variables;
  size_t first_factor_size = 1;
  size_t second_factor_size = 1;
  for (auto i = 1; i <= 4; ++i) {
    first_variables.push_back(variables[i - 1].get());
    first_factor_size *= variables[i - 1]->domain_size();
  }
  for (auto i = 3; i <= 6; ++i) {
    second_variables.push_back(variables[i - 1].get());
    second_factor_size *= variables[i - 1]->domain_size();
  }
  CircuitManager circuit_manager;
  std::vector<Node *> first_factors(first_factor_size, nullptr);
  for (size_t i = 0; i < first_factor_size; ++i) {
    first_factors[i] =
        circuit_manager.NewParameterTerminalNode({}, nullptr, {}, i);
  }
  CircuitFactor first_circuit_factor(first_variables, first_factors);
  std::vector<Node *> second_factors(second_factor_size, nullptr);
  for (size_t i = 0; i < second_factor_size; ++i) {
    second_factors[i] =
        circuit_manager.NewParameterTerminalNode({}, nullptr, {}, i);
  }
  CircuitFactor second_circuit_factor(second_variables, second_factors);
  auto result_circuit_factor =
      first_circuit_factor.Multiply(second_circuit_factor, &circuit_manager);
  ASSERT_NE(result_circuit_factor, nullptr);
  ASSERT_EQ(result_circuit_factor->variables().size(), (size_t)6);
  EXPECT_EQ(result_circuit_factor->variables()[0], variables[0].get());
  // checks {0, 0, 0, 0, 1, 0} should be {0, 0, 0, 0} * {0, 0, 1, 0}
  Node *result_node_000010 =
      result_circuit_factor->GetNodeFromVariableConfiguration(
          {0, 0, 0, 0, 1, 0});
  Node *first_node_0000 =
      first_circuit_factor.GetNodeFromVariableConfiguration({0, 0, 0, 0});
  Node *second_node_0010 =
      second_circuit_factor.GetNodeFromVariableConfiguration({0, 0, 1, 0});
  EXPECT_EQ(result_node_000010->type(), node_type::product);
  ASSERT_EQ(result_node_000010->children().size(), (size_t)2);
  EXPECT_EQ(result_node_000010->children()[0], first_node_0000);
  EXPECT_EQ(result_node_000010->children()[1], second_node_0010);
}

TEST(CIRCUIT_FACTOR_TEST, MULTIPLY_CIRCUIT_FACTOR_WITH_SINGLETON_TEST) {
  std::vector<std::unique_ptr<Variable>> total_variables;
  for (auto i = 1; i <= 4; ++i) {
    total_variables.push_back(
        std::make_unique<Variable>(i, i, variable_type::evidence));
  }
  std::vector<Variable *> factor_variables;
  size_t factor_size = 1;
  for (const auto &cur_variable : total_variables) {
    factor_variables.push_back(cur_variable.get());
    factor_size *= cur_variable->domain_size();
  }
  CircuitManager circuit_manager;
  std::vector<Node *> first_factors(factor_size, nullptr);
  for (size_t i = 0; i < factor_size; ++i) {
    first_factors[i] =
        circuit_manager.NewParameterTerminalNode({}, nullptr, {}, i);
  }
  CircuitFactor first_circuit_factor(factor_variables, first_factors);
  CircuitFactor second_circuit_factor(
      {},
      {circuit_manager.NewParameterTerminalNode({}, nullptr, {}, factor_size)});
  auto result_factor_1 =
      first_circuit_factor.Multiply(second_circuit_factor, &circuit_manager);
  auto result_node_1_0001 =
      result_factor_1->GetNodeFromVariableConfiguration({0, 0, 1, 1});
  ASSERT_EQ(result_node_1_0001->children().size(), (size_t)2);
  EXPECT_EQ(
      result_node_1_0001->children()[0],
      first_circuit_factor.GetNodeFromVariableConfiguration({0, 0, 1, 1}));
  EXPECT_EQ(result_node_1_0001->children()[1],
            second_circuit_factor.GetNodeFromVariableConfiguration({}));
  auto result_factor_2 =
      second_circuit_factor.Multiply(first_circuit_factor, &circuit_manager);
  auto result_node_2_0001 =
      result_factor_2->GetNodeFromVariableConfiguration({0, 0, 1, 1});
  ASSERT_EQ(result_node_2_0001->children().size(), (size_t)2);
  EXPECT_EQ(
      result_node_2_0001->children()[0],
      first_circuit_factor.GetNodeFromVariableConfiguration({0, 0, 1, 1}));
  EXPECT_EQ(result_node_2_0001->children()[1],
            second_circuit_factor.GetNodeFromVariableConfiguration({}));
}

} // namespace test_circuit
} // namespace test_bayesian_network
