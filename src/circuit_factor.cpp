#include <list>
#include <test_bayesian_network/circuit_factor.h>
#include <test_bayesian_network/circuit_manager.h>

namespace {
using std::list;
using std::vector;
using test_bayesian_network::test_circuit::CircuitManager;
} // namespace

namespace test_bayesian_network {

bool CircuitFactor::CompareVariableForSorting(Variable *first,
                                              Variable *second) {
  return first->variable_index() < second->variable_index();
}

CircuitFactor::CircuitFactor(vector<Variable *> variables,
                             vector<test_circuit::Node *> factor_nodes)
    : variables_(std::move(variables)), factor_nodes_(std::move(factor_nodes)) {
}

size_t CircuitFactor::GetEntryIndexFromVariableConfiguration(
    const std::vector<Variable *> &variables,
    const std::vector<DomainSize> &variable_config) {
  if (variables.empty() && variable_config.empty()) {
    return 0;
  }
  size_t index = 0;
  index += variable_config[0];
  for (NodeSize variable_index = 1;
       variable_index < (NodeSize)variable_config.size(); ++variable_index) {
    index *= variables[variable_index]->domain_size();
    index += variable_config[variable_index];
  }
  return index;
}

test_circuit::Node *CircuitFactor::GetNodeFromVariableConfiguration(
    const std::vector<DomainSize> &variable_config) const {
  size_t entry_index =
      GetEntryIndexFromVariableConfiguration(variables_, variable_config);
  return factor_nodes_[entry_index];
}

std::vector<DomainSize> CircuitFactor::GetVariableConfigurationFromEntryIndex(
    const std::vector<Variable *> &variables, size_t entry_index) {
  std::vector<DomainSize> variable_configuration(variables.size(), 0);
  for (NodeSize i = 0; i < (NodeSize)variables.size(); ++i) {
    const NodeSize considered_variable_it = variables.size() - i - 1;
    Variable *cur_variable = variables[considered_variable_it];
    const DomainSize cur_domain_index =
        entry_index % cur_variable->domain_size();
    entry_index /= cur_variable->domain_size();
    variable_configuration[considered_variable_it] = cur_domain_index;
  }
  return variable_configuration;
}

std::unique_ptr<CircuitFactor>
CircuitFactor::Multiply(const CircuitFactor &other,
                        CircuitManager *circuit_manager) const {
  std::vector<Variable *> new_variables;
  // *_node_indexes indicates the position of variables in *_variables_ in
  // new_variables.
  std::vector<NodeSize> this_node_indexes;
  std::vector<NodeSize> other_node_indexes;
  NodeSize this_variable_it = 0;
  NodeSize other_variable_it = 0;
  while (this_variable_it != (NodeSize)variables_.size() ||
         other_variable_it != (NodeSize)other.variables_.size()) {
    if (this_variable_it == (NodeSize)variables_.size()) {
      new_variables.push_back(other.variables_[other_variable_it]);
      other_node_indexes.push_back(new_variables.size() - 1);
      ++other_variable_it;
      continue;
    }
    if (other_variable_it == (NodeSize)other.variables_.size()) {
      new_variables.push_back(variables_[this_variable_it]);
      this_node_indexes.push_back(new_variables.size() - 1);
      ++this_variable_it;
      continue;
    }
    if (variables_[this_variable_it]->variable_index() ==
        other.variables_[other_variable_it]->variable_index()) {
      new_variables.push_back(variables_[this_variable_it]);
      this_node_indexes.push_back(new_variables.size() - 1);
      other_node_indexes.push_back(new_variables.size() - 1);
      ++this_variable_it;
      ++other_variable_it;
      continue;
    }
    if (variables_[this_variable_it]->variable_index() <
        other.variables_[other_variable_it]->variable_index()) {
      new_variables.push_back(variables_[this_variable_it]);
      this_node_indexes.push_back(new_variables.size() - 1);
      ++this_variable_it;
      continue;
    }
    if (variables_[this_variable_it]->variable_index() >
        other.variables_[other_variable_it]->variable_index()) {
      new_variables.push_back(other.variables_[other_variable_it]);
      other_node_indexes.push_back(new_variables.size() - 1);
      ++other_variable_it;
      continue;
    }
  }
  size_t factor_size = 1;
  for (const Variable *cur_variable : new_variables) {
    factor_size *= cur_variable->domain_size();
  }
  vector<test_circuit::Node *> new_factor_nodes(factor_size, nullptr);
  for (size_t factor_index = 0; factor_index < factor_size; ++factor_index) {
    std::vector<DomainSize> factor_variable_configuration =
        GetVariableConfigurationFromEntryIndex(new_variables, factor_index);
    std::vector<DomainSize> this_variable_configuration(variables_.size(), 0);
    std::vector<DomainSize> other_variable_configuration(
        other.variables_.size(), 0);
    for (NodeSize this_variable_it = 0;
         this_variable_it < (NodeSize)variables_.size(); ++this_variable_it) {
      this_variable_configuration[this_variable_it] =
          factor_variable_configuration[this_node_indexes[this_variable_it]];
    }
    for (NodeSize other_variable_it = 0;
         other_variable_it < (NodeSize)other.variables_.size();
         ++other_variable_it) {
      other_variable_configuration[other_variable_it] =
          factor_variable_configuration[other_node_indexes[other_variable_it]];
    }
    test_circuit::Node *this_circuit_node =
        GetNodeFromVariableConfiguration(this_variable_configuration);
    test_circuit::Node *other_circuit_node =
        other.GetNodeFromVariableConfiguration(other_variable_configuration);
    test_circuit::Node *new_node = circuit_manager->NewProductNode(
        {this_circuit_node, other_circuit_node});
    new_factor_nodes[factor_index] = new_node;
  }
  return std::make_unique<CircuitFactor>(std::move(new_variables),
                                         std::move(new_factor_nodes));
}

std::unique_ptr<CircuitFactor>
CircuitFactor::SumOut(Variable *sum_out_variable,
                      CircuitManager *circuit_manager) const {
  std::vector<Variable *> new_variables;
  new_variables.reserve(variables_.size() - 1);
  NodeSize sum_out_variable_old_index = -1;
  const NodeSize old_variable_size = variables_.size();
  for (NodeSize i = 0; i < old_variable_size; ++i) {
    Variable *cur_variable = variables_[i];
    if (cur_variable->variable_index() == sum_out_variable->variable_index()) {
      sum_out_variable_old_index = i;
    } else {
      new_variables.push_back(cur_variable);
    }
  }
  size_t factor_size = 1;
  for (const Variable *cur_variable : new_variables) {
    factor_size *= cur_variable->domain_size();
  }
  vector<test_circuit::Node *> new_factor_nodes(factor_size, nullptr);
  for (size_t factor_index = 0; factor_index < factor_size; ++factor_index) {
    std::vector<DomainSize> factor_variable_configuration =
        GetVariableConfigurationFromEntryIndex(new_variables, factor_index);
    std::vector<DomainSize> old_variable_configuration_template;
    old_variable_configuration_template.reserve(variables_.size());
    const NodeSize new_variable_size = new_variables.size();
    for (NodeSize i = 0; i < new_variable_size; ++i) {
      if (i == sum_out_variable_old_index) {
        old_variable_configuration_template.push_back(0);
      }
      old_variable_configuration_template.push_back(
          factor_variable_configuration[i]);
    }
    if (sum_out_variable_old_index == new_variable_size) {
      old_variable_configuration_template.push_back(0);
    }
    std::vector<test_circuit::Node *> children(sum_out_variable->domain_size(),
                                               nullptr);
    for (DomainSize i = 0; i < sum_out_variable->domain_size(); ++i) {
      old_variable_configuration_template[sum_out_variable_old_index] = i;
      children[i] =
          GetNodeFromVariableConfiguration(old_variable_configuration_template);
    }
    if (children.size() > 1) {
      test_circuit::Node *new_node =
          circuit_manager->NewSumNode(std::move(children));
      new_factor_nodes[factor_index] = new_node;
    } else {
      new_factor_nodes[factor_index] = children[0];
    }
  }
  return std::make_unique<CircuitFactor>(std::move(new_variables),
                                         std::move(new_factor_nodes));
}
/*
  list<std::vector<DomainSize>> new_configuration_list;
  new_configuration_list.push_back({});
  NodeSize new_variable_it = 0;
  while (new_variable_it < (NodeSize)new_variables.size()) {
    auto cur_configuration_list_it = new_configuration_list.begin();
    Variable *cur_variable = new_variables[new_variable_it];
    const DomainSize cur_variable_domain_size = cur_variable->domain_size();
    while (cur_configuration_list_it != new_configuration_list.end()) {
      cur_configuration_list_it->push_back(0);
      for (DomainSize i = 1; i < cur_variable_domain_size; ++i) {
        std::vector<DomainSize> next_configuration = *cur_configuration_list_it;
        ++cur_configuration_list_it; // points to element that is next to the
                                     // last inserted element
        // update the configuration of last variable to i
        next_configuration.back() = i;
        cur_configuration_list_it = new_configuration_list.insert(
            cur_configuration_list_it,
            std::move(next_configuration)); // points to the element that is
                                            // just inserted.
      }
      ++cur_configuration_list_it;
    }
    ++new_variable_it;
  }
*/

} // namespace test_bayesian_network
