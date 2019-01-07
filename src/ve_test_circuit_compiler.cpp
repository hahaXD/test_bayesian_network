#include <algorithm>
#include <cassert>
#include <unordered_map>

#include <htd/main.hpp>
#include <test_bayesian_network/bucket_elimination.h>
#include <test_bayesian_network/ve_test_circuit_compiler.h>

namespace test_bayesian_network {
VeTestCircuitCompiler::VeTestCircuitCompiler(
    TestBayesianNetwork *input_network,
    test_circuit::CircuitManager *circuit_manager)
    : input_network_(input_network), circuit_manager_(circuit_manager) {}

test_circuit::ZNode *VeTestCircuitCompiler::Run() const {
  std::vector<Node *> nodes = input_network_->nodes();
  std::unordered_set<Node *> unselected_testing_nodes;
  Node *query_node = nullptr;
  for (Node *cur_node : nodes) {
    if (cur_node->type() == node_type::test) {
      unselected_testing_nodes.insert(cur_node);
    }
    if (cur_node->variable()->type() == variable_type::query) {
      query_node = cur_node;
    }
  }
  if (query_node == nullptr) {
    return nullptr;
  }
  for (Node *cur_node : nodes) {
    Variable *cur_variable = cur_node->variable();
    size_t factor_size = cur_variable->domain_size();
    std::vector<Variable *> variable_in_factor;
    std::vector<Variable *> variable_in_parent;
    variable_in_factor.push_back(cur_variable);
    for (Node *cur_parent : cur_node->parents()) {
      variable_in_factor.push_back(cur_parent->variable());
      variable_in_parent.push_back(cur_parent->variable());
      factor_size *= cur_parent->variable()->domain_size();
    }
    std::sort(variable_in_factor.begin(), variable_in_factor.end(),
              CircuitFactor::CompareVariableForSorting);
    std::sort(variable_in_parent.begin(), variable_in_parent.end(),
              CircuitFactor::CompareVariableForSorting);
    NodeSize child_variable_pos = -1;
    const NodeSize num_variables_in_factor = variable_in_factor.size();
    for (NodeSize i = 0; i < num_variables_in_factor; ++i) {
      if (variable_in_factor[i] == cur_variable) {
        child_variable_pos = i;
      }
    }
    if (cur_node->type() == node_type::regular) {
      // Construct circuit factor for a regular node
      std::vector<test_circuit::Node *> cur_factor(factor_size, nullptr);
      for (size_t i = 0; i < factor_size; ++i) {
        std::vector<DomainSize> configuration =
            CircuitFactor::GetVariableConfigurationFromEntryIndex(
                variable_in_factor, i);
        std::vector<DomainSize> parent_configuration(
            configuration.begin(), configuration.begin() + child_variable_pos);
        parent_configuration.insert(parent_configuration.end(),
                                    configuration.begin() + child_variable_pos +
                                        1,
                                    configuration.end());
        auto parameter_node = circuit_manager_->NewParameterTerminalNode(
            variable_in_parent, cur_node->variable(),
            std::move(parent_configuration), configuration[child_variable_pos]);
        if (cur_node->variable()->type() == variable_type::evidence) {
          auto variable_node = circuit_manager_->NewVariableTerminalNode(
              cur_variable, configuration[child_variable_pos]);
          cur_factor[i] =
              circuit_manager_->NewProductNode({parameter_node, variable_node});
        } else {
          cur_factor[i] = parameter_node;
        }
      }
      auto cur_circuit_factor = std::make_unique<CircuitFactor>(
          std::move(variable_in_factor), std::move(cur_factor));
      cur_node->SetCircuitFactor(std::move(cur_circuit_factor));
      continue;
    } else {
      // Get parent variables for the test node.
      unselected_testing_nodes.erase(cur_node);
      auto pruned_network = input_network_->PrunedNetworkForTesting(
          cur_node, unselected_testing_nodes);
      std::vector<Variable *> elimination_order =
          pruned_network->GetEliminationOrder(cur_node->parents());
      // Get the factor of all parent configurations
      std::vector<CircuitFactor *> circuit_factors_for_ve;
      for (Node *cur_node_in_pruned_network : pruned_network->nodes()) {
        circuit_factors_for_ve.push_back(
            cur_node_in_pruned_network->GetCircuitFactor());
      }
      BucketElimination be(std::move(circuit_factors_for_ve),
                           std::move(elimination_order), circuit_manager_);
      auto result = be.Run();
      test_circuit::Node *pr_e_node =
          circuit_manager_->NewSumNode(result->factor_nodes());
      std::vector<test_circuit::Node *> cur_factor(factor_size, nullptr);
      for (size_t i = 0; i < factor_size; ++i) {
        std::vector<DomainSize> configuration =
            CircuitFactor::GetVariableConfigurationFromEntryIndex(
                variable_in_factor, i);
        std::vector<DomainSize> parent_configuration(
            configuration.begin(), configuration.begin() + child_variable_pos);
        parent_configuration.insert(parent_configuration.end(),
                                    configuration.begin() + child_variable_pos +
                                        1,
                                    configuration.end());
        size_t index_for_parent_config =
            CircuitFactor::GetEntryIndexFromVariableConfiguration(
                variable_in_parent, parent_configuration);
        auto threshold_parameter =
            circuit_manager_->NewTestThresholdParameterTerminalNode(
                variable_in_parent, cur_variable, parent_configuration);
        auto positive_parameter =
            circuit_manager_->NewTestProbabilityParameterTerminalNode(
                true, variable_in_parent, cur_variable, parent_configuration,
                configuration[child_variable_pos]);
        auto negative_parameter =
            circuit_manager_->NewTestProbabilityParameterTerminalNode(
                false, variable_in_parent, cur_variable, parent_configuration,
                configuration[child_variable_pos]);
        auto test_node = circuit_manager_->NewTestNode(
            {result->factor_nodes()[index_for_parent_config], pr_e_node,
             threshold_parameter, negative_parameter, positive_parameter});
        if (cur_node->variable()->type() == variable_type::evidence) {
          auto variable_node = circuit_manager_->NewVariableTerminalNode(
              cur_variable, configuration[child_variable_pos]);
          cur_factor[i] =
              circuit_manager_->NewProductNode({test_node, variable_node});
        } else {
          cur_factor[i] = test_node;
        }
      }
      auto cur_circuit_factor = std::make_unique<CircuitFactor>(
          std::move(variable_in_factor), std::move(cur_factor));
      cur_node->SetCircuitFactor(std::move(cur_circuit_factor));
      continue;
    }
  }
  auto pruned_network_for_query =
      input_network_->PrunedNetworkForQuery(query_node);
  auto elimination_order =
      pruned_network_for_query->GetEliminationOrder({query_node});
  std::vector<CircuitFactor *> circuit_factors_for_ve;
  for (Node *cur_node : pruned_network_for_query->nodes()) {
    circuit_factors_for_ve.push_back(cur_node->GetCircuitFactor());
  }
  BucketElimination be(std::move(circuit_factors_for_ve),
                       std::move(elimination_order), circuit_manager_);
  auto result = be.Run();
  const std::vector<test_circuit::Node *> &nodes_in_result =
      result->factor_nodes();
  assert(nodes_in_result.size() ==
         (size_t)query_node->variable()->domain_size());
  return circuit_manager_->NewZNode(nodes_in_result)->get_z_node();
}
} // namespace test_bayesian_network
