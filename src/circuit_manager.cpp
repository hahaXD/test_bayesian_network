#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <unordered_set>

#include <test_bayesian_network/circuit_manager.h>
#include <test_bayesian_network/circuit_node.h>

namespace {
using test_bayesian_network::test_circuit::Node;
bool compare_node_ids(Node *a, Node *b) { return a->node_id() < b->node_id(); }

} // namespace

namespace test_bayesian_network {
namespace test_circuit {
CircuitManager::CircuitManager() : unique_variable_nodes_(), next_node_id_(0) {}

Node *CircuitManager::NewProductNode(std::vector<Node *> children) {
  std::sort(children.begin(), children.end(), compare_node_ids);
  node_cache_.push_back(
      std::make_unique<ProductNode>(next_node_id_++, std::move(children)));
  return node_cache_.back().get();
}

Node *CircuitManager::NewSumNode(std::vector<Node *> children) {
  std::sort(children.begin(), children.end(), compare_node_ids);
  node_cache_.push_back(
      std::make_unique<SumNode>(next_node_id_++, std::move(children)));
  return node_cache_.back().get();
}

Node *CircuitManager::NewTestNode(std::vector<Node *> children) {
  node_cache_.push_back(
      std::make_unique<TestNode>(next_node_id_++, std::move(children)));
  return node_cache_.back().get();
}

Node *CircuitManager::NewVariableTerminalNode(Variable *variable,
                                              DomainSize value) {
  const NodeSize variable_index = variable->variable_index();
  auto unique_variable_nodes_it = unique_variable_nodes_.find(variable_index);
  if (unique_variable_nodes_it == unique_variable_nodes_.end()) {
    node_cache_.push_back(std::make_unique<VariableTerminalNode>(
        next_node_id_++, variable, value));
    unique_variable_nodes_it =
        unique_variable_nodes_
            .insert({variable_index,
                     std::vector<Node *>(variable->domain_size(), nullptr)})
            .first;
    unique_variable_nodes_it->second[value] = node_cache_.back().get();
    return node_cache_.back().get();
  }
  if (unique_variable_nodes_it->second[value] == nullptr) {
    node_cache_.push_back(std::make_unique<VariableTerminalNode>(
        next_node_id_++, variable, value));
    unique_variable_nodes_it->second[value] = node_cache_.back().get();
    return node_cache_.back().get();
  }
  return unique_variable_nodes_it->second[value];
}

Node *CircuitManager::NewParameterTerminalNode(
    std::vector<Variable *> parent_variables, Variable *child_variable,
    std::vector<DomainSize> parent_configurations,
    DomainSize child_configuraiton) {
  node_cache_.push_back(std::make_unique<ParameterTerminalNode>(
      next_node_id_++, std::move(parent_variables), child_variable,
      std::move(parent_configurations), child_configuraiton));
  return node_cache_.back().get();
}

Node *CircuitManager::NewTestThresholdParameterTerminalNode(
    std::vector<Variable *> parent_variables, Variable *child_variable,
    std::vector<DomainSize> parent_configurations) {
  auto variable_it =
      unique_threshold_nodes_.find(child_variable->variable_index());
  if (variable_it == unique_threshold_nodes_.end()) {
    std::vector<Node *> factor(CircuitFactor::GetFactorSize(parent_variables),
                               nullptr);
    variable_it =
        unique_threshold_nodes_
            .insert(std::make_pair(child_variable->variable_index(),
                                   CircuitFactor(parent_variables, factor)))
            .first;
  }
  Node *result_node = variable_it->second.GetNodeFromVariableConfiguration(
      parent_configurations);
  if (result_node == nullptr) {
    node_cache_.push_back(std::make_unique<TestThresholdParameterTerminalNode>(
        next_node_id_++, std::move(parent_variables), child_variable,
        std::move(parent_configurations)));
    result_node = node_cache_.back().get();
    variable_it->second.SetNodeFromVariableConfiguration(
        result_node->get_parameter_terminal_node()->parent_configurations(),
        result_node);
  }
  return result_node;
}

Node *CircuitManager::NewTestProbabilityParameterTerminalNode(
    bool test_result, std::vector<Variable *> parent_variables,
    Variable *child_variable, std::vector<DomainSize> parent_configurations,
    DomainSize child_configuraiton) {
  node_cache_.push_back(std::make_unique<TestProbabilityParameterTerminalNode>(
      next_node_id_++, test_result, std::move(parent_variables), child_variable,
      std::move(parent_configurations), child_configuraiton));
  return node_cache_.back().get();
}

Node *CircuitManager::NewZNode(std::vector<Node *> children) {
  node_cache_.push_back(
      std::make_unique<ZNode>(next_node_id_++, std::move(children)));
  return node_cache_.back().get();
}

std::vector<Node *> CircuitManager::SerializeNodes(Node *root_node) {
  std::unordered_map<Node *, uintmax_t> parents_per_node;
  parents_per_node[root_node] = 0;
  std::unordered_set<Node *> explored_nodes;
  explored_nodes.insert(root_node);
  std::list<Node *> queue;
  queue.push_back(root_node);
  while (!queue.empty()) {
    Node *cur_node = queue.front();
    queue.pop_front();
    for (Node *child_node : cur_node->children()) {
      parents_per_node[child_node] += 1;
      if (explored_nodes.find(child_node) == explored_nodes.end()) {
        explored_nodes.insert(child_node);
        queue.push_back(child_node);
      }
    }
  }
  // create nodes parent before leaf.
  std::vector<Node *> result;
  queue.clear();
  queue.push_back(root_node);
  while (!queue.empty()) {
    Node *cur_node = queue.front();
    result.push_back(cur_node);
    queue.pop_front();
    for (Node *child_node : cur_node->children()) {
      auto &cur_number_of_parents = parents_per_node[child_node];
      cur_number_of_parents -= 1;
      if (cur_number_of_parents == 0) {
        queue.push_back(child_node);
      }
    }
  }
  assert(parents_per_node.size() == result.size());
  return result;
}

void CircuitManager::SaveAsTacFile(Node *root_node, const char *tac_filename,
                                   const char *lmap_filename) {
  std::ofstream output_file;
  output_file.open(tac_filename);
  std::vector<Node *> sequenced_nodes = SerializeNodes(root_node);
  size_t next_node_index = 0;
  std::vector<Node *> literal_node_map;
  literal_node_map.push_back(nullptr);
  std::unordered_map<Node *, size_t> node_to_index;
  // write nodes from leaf to root.
  for (auto it = sequenced_nodes.rbegin(); it != sequenced_nodes.rend(); ++it) {
    Node *cur_node = *it;
    node_to_index[cur_node] = next_node_index;
    if (!cur_node->is_leaf()) {
      output_file << next_node_index++ << " "
                  << static_cast<char>(cur_node->type()) << " "
                  << cur_node->children().size() << " ";
      for (Node *cur_child : cur_node->children()) {
        assert(node_to_index.find(cur_child) != node_to_index.end());
        output_file << node_to_index[cur_child] << " ";
      }
      output_file << std::endl;
    } else {
      output_file << next_node_index++ << " L " << literal_node_map.size()
                  << std::endl;
      literal_node_map.push_back(cur_node);
    }
  }
  output_file.close();
  output_file.open(lmap_filename);
  const size_t num_terminal_nodes = literal_node_map.size();
  for (size_t literal_index = 1; literal_index < num_terminal_nodes;
       ++literal_index) {
    Node *cur_terminal_node = literal_node_map[literal_index];
    if (cur_terminal_node->type() == node_type::variable) {
      // variable indicator node
      VariableTerminalNode *cur_variable_terminal_node =
          cur_terminal_node->get_variable_terminal_node();
      assert(cur_terminal_node != nullptr);
      const auto &variable_and_value =
          cur_variable_terminal_node->variable_and_value();
      const auto &variable_domain_names =
          variable_and_value.first->domain_names();
      output_file << literal_index << " i "
                  << variable_and_value.first->variable_name() << "="
                  << variable_domain_names[variable_and_value.second]
                  << std::endl;
    } else {
      // Parameter node
      assert(cur_terminal_node->get_parameter_terminal_node() != nullptr);
      output_file << literal_index << " p 0 "
                  << cur_terminal_node->get_parameter_terminal_node()->Label()
                  << std::endl;
    }
  }
  output_file.close();
}
} // namespace test_circuit
} // namespace test_bayesian_network
