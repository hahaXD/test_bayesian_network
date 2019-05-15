#include <algorithm>
#include <fstream>
#include <iostream>
#include <list>
#include <regex>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

#include <htd/main.hpp>
#include <test_bayesian_network/test_bayesian_network.h>

namespace test_bayesian_network {
namespace util {
std::vector<std::string> TopologicalSortNodes(
    const std::unordered_map<std::string, std::unordered_set<std::string>>
        &parents) {
  std::vector<std::string> results;
  std::unordered_map<std::string, NodeSize> num_children_per_node;
  std::list<std::string> processing_list; // contains leaf nodes
  for (auto it = parents.begin(); it != parents.end(); ++it) {
    for (const auto &cur_parent : it->second) {
      auto cur_parent_it = num_children_per_node.find(cur_parent);
      if (cur_parent_it == num_children_per_node.end()) {
        cur_parent_it = num_children_per_node.insert({cur_parent, 0}).first;
      }
      cur_parent_it->second += 1;
    }
  }
  for (auto it = parents.begin(); it != parents.end(); ++it) {
    auto cur_child_it = num_children_per_node.find(it->first);
    if (cur_child_it == num_children_per_node.end()) {
      processing_list.push_back(it->first);
    }
  }
  while (!processing_list.empty()) {
    std::string cur_node = *processing_list.begin();
    processing_list.pop_front();
    results.push_back(cur_node);
    const auto &cur_parents = parents.find(cur_node)->second;
    for (const auto &cur_parent_name : cur_parents) {
      auto cur_parent_it = num_children_per_node.find(cur_parent_name);
      cur_parent_it->second -= 1;
      if (cur_parent_it->second == 0) {
        processing_list.push_back(cur_parent_name);
      }
    }
  }
  std::reverse(results.begin(), results.end());
  return results;
}
} // namespace util
TestBayesianNetwork::TestBayesianNetwork(std::vector<Node *> nodes)
    : nodes_(std::move(nodes)) {}

TestBayesianNetwork::TestBayesianNetwork(
    std::vector<std::unique_ptr<Node>> unique_nodes)
    : unique_nodes_(std::move(unique_nodes)), nodes_() {
  for (const auto &cur_node : unique_nodes_) {
    nodes_.push_back(cur_node.get());
  }
}

std::unique_ptr<TestBayesianNetwork>
TestBayesianNetwork::PrunedNetworkForTesting(
    Node *test_node,
    std::unordered_set<Node *> unselected_testing_nodes) const {
  std::unordered_set<Node *> query_nodes(test_node->parents().begin(),
                                         test_node->parents().end());
  unselected_testing_nodes.insert(test_node);
  return PruneNetwork(query_nodes, std::move(unselected_testing_nodes));
}

const std::vector<Node *> TestBayesianNetwork::nodes() const { return nodes_; }

std::vector<Variable *> TestBayesianNetwork::GetEliminationOrder(
    const std::vector<Node *> &nodes_to_query) const {
  std::unordered_set<Node *> nodes_to_query_set(nodes_to_query.begin(),
                                                nodes_to_query.end());
  std::unordered_map<Node *, htd::vertex_t> node_map;
  std::unordered_map<htd::vertex_t, Node *> htd_map_index_to_node;
  htd::vertex_t next_index = 1;
  std::unique_ptr<htd::LibraryInstance> manager(
      htd::createManagementInstance(htd::Id::FIRST));
  htd::MultiHypergraph htd_graph(manager.get());
  for (Node *cur_node : nodes_) {
    node_map[cur_node] = next_index;
    htd_map_index_to_node[next_index++] = cur_node;
    htd_graph.addVertex();
    std::vector<Node *> node_list;
    node_list.push_back(cur_node);
    node_list.insert(node_list.end(), cur_node->parents().begin(),
                     cur_node->parents().end());
    const size_t node_list_size = node_list.size();
    for (size_t i = 0; i < node_list_size; ++i) {
      for (size_t j = i + 1; j < node_list_size; ++j) {
        htd_graph.addEdge(node_map[node_list[i]], node_map[node_list[j]]);
      }
    }
  }
  htd::MinFillOrderingAlgorithm minfill_manager(manager.get());
  auto ordering = minfill_manager.computeOrdering(htd_graph);
  std::vector<Variable *> elimination_order;
  for (auto var_index : ordering->sequence()) {
    Node *cur_node = htd_map_index_to_node[var_index];
    if (nodes_to_query_set.find(cur_node) != nodes_to_query_set.end()) {
      continue;
    } else {
      elimination_order.push_back(cur_node->variable());
    }
  }
  delete (ordering);
  return elimination_order;
}
std::unique_ptr<TestBayesianNetwork>
TestBayesianNetwork::PrunedNetworkForQuery(Node *query_node) const {
  std::unordered_set<Node *> query_nodes = {query_node};
  return PruneNetwork(query_nodes, {});
}

std::unique_ptr<TestBayesianNetwork> TestBayesianNetwork::PruneNetwork(
    const std::unordered_set<Node *> &query_nodes,
    std::unordered_set<Node *> unselected_testing_nodes) const {
  std::vector<Node *> network_wo_unselected_test_nodes;
  std::unordered_set<Node *> nodes_to_be_pruned =
      std::move(unselected_testing_nodes);
  for (Node *cur_node : nodes_) {
    for (Node *cur_parent : cur_node->parents()) {
      if (nodes_to_be_pruned.find(cur_parent) != nodes_to_be_pruned.end()) {
        nodes_to_be_pruned.insert(cur_node);
        break;
      }
    }
    if (nodes_to_be_pruned.find(cur_node) != nodes_to_be_pruned.end()) {
      continue;
    }
    // nodes that will not be pruned by the unselected testing nodes
    network_wo_unselected_test_nodes.push_back(cur_node);
  }
  // Prune according to evidence.
  std::unordered_set<Node *> internal_nodes;
  std::vector<Node *> network_with_evidence_leaf_reversed;
  for (auto i = network_wo_unselected_test_nodes.rbegin();
       i != network_wo_unselected_test_nodes.rend(); ++i) {
    Node *cur_node = *i;
    if (cur_node->variable_type() == variable_type::evidence ||
        query_nodes.find(cur_node) != query_nodes.end() ||
        internal_nodes.find(cur_node) != internal_nodes.end()) {
      // do not prune, and record its parent as internal node.
      for (Node *cur_parent : cur_node->parents()) {
        internal_nodes.insert(cur_parent);
      }
      network_with_evidence_leaf_reversed.push_back(cur_node);
      continue;
    }
  }
  std::reverse(network_with_evidence_leaf_reversed.begin(),
               network_with_evidence_leaf_reversed.end());
  return std::make_unique<TestBayesianNetwork>(
      std::move(network_with_evidence_leaf_reversed));
}

std::unique_ptr<TestBayesianNetwork>
TestBayesianNetwork::ParseTestBayesianNetworkFromNetFile(const char *filename) {
  std::ifstream input_file_stream;
  input_file_stream.open(filename);
  if (!input_file_stream) {
    std::cerr << "Unable to open file " << filename << std::endl;
    return nullptr;
  }
  std::string node_name = "";
  bool node_start = false;
  std::string line = "";
  std::smatch m;
  std::regex node_pattern("node[ ]*\\b([^ ]+)\\b");
  std::regex state_line_pattern("states = ([^;]*);");
  std::regex state_pattern("\\\"([^\\\" ]*)\\\"");
  std::regex diagnosistype_line_pattern("diagnosistype = \\\"([^\\\"]*)\\\";");
  std::regex decision_line_pattern("isdecisionvariable = \\\"([^\\\"]*)\\\";");
  std::regex potential_pattern("potential \\(([^\\(\\)]*)\\)");
  std::regex node_name_pattern("\\b([a-zA-Z0-9_]+)\\b");
  std::unordered_map<std::string, bool> test_node_per_variable;
  std::unordered_map<std::string, std::vector<std::string>>
      domain_names_per_variable;
  std::unordered_map<std::string, variable_type> variable_type_per_variable;
  std::unordered_map<std::string, std::unordered_set<std::string>>
      parents_per_variables;
  while (std::getline(input_file_stream, line)) {
    if (std::regex_search(line, m, node_pattern)) {
      node_name = m[1];
      test_node_per_variable[node_name] = false;
      variable_type_per_variable[node_name] = variable_type::hidden;
      continue;
    }
    if (line.find("}") != std::string::npos) {
      node_start = false;
      continue;
    }
    if (node_start) {
      if (std::regex_search(line, m, state_line_pattern)) {
        std::string states_content = m[1];
        std::vector<std::string> domain_names;
        while (std::regex_search(states_content, m, state_pattern)) {
          domain_names.push_back(m[1]);
          states_content = m.suffix().str();
        }
        domain_names_per_variable[node_name] = std::move(domain_names);
        continue;
      }
      if (std::regex_search(line, m, diagnosistype_line_pattern)) {
        if (m[1] == "AUXILIARY") {
          variable_type_per_variable[node_name] = variable_type::hidden;
        } else if (m[1] == "OBSERVATION") {
          variable_type_per_variable[node_name] = variable_type::evidence;
        } else {
          assert(m[1] == "TARGET"); variable_type_per_variable[node_name] = variable_type::query;}
        continue;
      }
      if (std::regex_search(line, m, decision_line_pattern)) {
        if (m[1] == "true") {
          test_node_per_variable[node_name] = true;
        } else {
          test_node_per_variable[node_name] = false;
        }
      }
    }
    if (line.find("{") != std::string::npos) {
      // set the start of the node def
      node_start = true;
    }
    if (std::regex_search(line, m, potential_pattern)) {
      std::string variable_content = m[1];
      std::string variable_name = "";
      std::unordered_set<std::string> parent_names;
      while (std::regex_search(variable_content, m, node_name_pattern)) {
        if (variable_name.empty()) {
          variable_name = m[1];
        } else {
          parent_names.insert(m[1]);
        }
        variable_content = m.suffix().str();
      }
      parents_per_variables[variable_name] = parent_names;
    }
  }
  std::vector<std::string> topological_ordered_nodes =
      util::TopologicalSortNodes(parents_per_variables);
  std::vector<std::unique_ptr<Node>> nodes;
  std::unordered_map<std::string, Node *> constructed_nodes;
  NodeSize next_variable_id = 0;
  for (const auto &cur_node_name : topological_ordered_nodes) {
    auto domain_names_it = domain_names_per_variable.find(cur_node_name);
    auto variable_type_it = variable_type_per_variable.find(cur_node_name);
    auto test_node_it = test_node_per_variable.find(cur_node_name);
    if (domain_names_it == domain_names_per_variable.end() ||
        variable_type_it == variable_type_per_variable.end() ||
        test_node_it == test_node_per_variable.end()) {
      std::cerr
          << "<Error at parsing .net file> Domain name is missing for variable "
          << cur_node_name << std::endl;
      return nullptr;
    }
    const auto &cur_domain_names = domain_names_it->second;
    const DomainSize cur_domain_size = cur_domain_names.size();
    variable_type cur_variable_type = variable_type_it->second;
    bool cur_test = test_node_it->second;
    auto new_variable = std::make_unique<Variable>(
        next_variable_id++, cur_domain_size, cur_variable_type);
    new_variable->set_names(cur_node_name, cur_domain_names);
    std::vector<Node *> parents;
    const std::unordered_set<std::string> &parent_names =
        parents_per_variables[cur_node_name];
    for (const auto &cur_parent_name : parent_names) {
      parents.push_back(constructed_nodes[cur_parent_name]);
    }
    std::sort(parents.begin(), parents.end(), [](Node *a, Node *b) -> bool {
      return a->variable()->variable_index() < b->variable()->variable_index();
    });
    nodes.emplace_back(std::make_unique<Node>(
        std::move(new_variable), std::move(parents),
        cur_test ? node_type::test : node_type::regular));
    constructed_nodes[cur_node_name] = nodes.back().get();
  }
  bool contains_query = false;
  bool contains_evidence = false;
  for (const auto &cur_node : nodes) {
    if (cur_node->variable()->type() == variable_type::evidence) {
      contains_evidence = true;
      continue;
    }
    if (cur_node->variable_type() == variable_type::query) {
      contains_query = true;
      continue;
    }
  }
  if (!contains_query || !contains_evidence) {
    std::cerr << "<Error at parsing .net file> The network does not contain "
                 "any query and/or any evidence nodes."
              << std::endl;
    return nullptr;
  }
  return std::make_unique<TestBayesianNetwork>(std::move(nodes));
}

} // namespace test_bayesian_network
