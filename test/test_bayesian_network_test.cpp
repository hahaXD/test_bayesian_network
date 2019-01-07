#include <fstream>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <test_bayesian_network/bucket_elimination.h>
#include <test_bayesian_network/node.h>

#include <test_bayesian_network/test_bayesian_network.h>
#include <test_bayesian_network/variable.h>

namespace test_bayesian_network {
namespace {
std::string test_net_file_content = R"(net
{
	propagationenginegenerator1791944048146838126L = "edu.ucla.belief.approx.BeliefPropagationSettings@4aa4e5a4";
	recoveryenginegenerator6944530267470113528l = "edu.ucla.util.SettingsImpl@7afd6854";
	node_size = (130.0 55.0);
}

node variable2
{
	states = ("state0" "state1" );
	position = (244 -326);
	isdecisionvariable = "false";
	diagnosistype = "TARGET";
	DSLxSUBMODEL = "Root Submodel";
	isimpactvariable = "false";
	ismapvariable = "false";
	label = "variable2";
	ID = "variable2";
	DSLxEXTRA_DEFINITIONxDIAGNOSIS_TYPE = "AUXILIARY";
	excludepolicy = "include whole CPT";
}
node variable1
{
	states = ("state0" "state1" );
	position = (82 -236);
	isdecisionvariable = "false";
	diagnosistype = "AUXILIARY";
	DSLxSUBMODEL = "Root Submodel";
	isqueryparticipant = "false";
	isimpactvariable = "true";
	ismapvariable = "false";
	label = "variable1";
	ID = "variable1";
	ishiddenvariable = "false";
	DSLxEXTRA_DEFINITIONxDIAGNOSIS_TYPE = "AUXILIARY";
	excludepolicy = "include whole CPT";
}
node variable0
{
	states = ("state0" "state1" );
	position = (148 -134);
	diagnosistype = "OBSERVATION";
	DSLxSUBMODEL = "Root Submodel";
	isqueryparticipant = "false";
	label = "variable0";
	ishiddenvariable = "false";
	DSLxEXTRA_DEFINITIONxDIAGNOSIS_TYPE = "OBSERVATION";
	excludepolicy = "include whole CPT";
	isdecisionvariable = "true";
	iscptvalid = "true";
	isimpactvariable = "false";
	ismapvariable = "false";
	ID = "variable0";
}
potential ( variable2 | variable0 variable1 )
{
	data = (((	0.5	0.5	)
		(	0.5	0.5	))
		((	0.5	0.5	)
		(	0.5	0.5	)));
}
potential ( variable1 | variable0 )
{
	data = ((	0.5	0.5	)
		(	0.5	0.5	));
}
potential ( variable0 | )
{
	data = (	0.5	0.5	);
})";

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
TEST(TEST_BAYESIAN_NETWORK_TEST, NETWORK_PRUNNING_TEST_SELECT_FIRST_TEST_NODE) {
  auto test_network = ConstructNetwork1();
  // Select a_1
  Node *node_a1 = test_network->nodes()[1];
  std::unordered_set<Node *> unselected_testing_nodes = {
      test_network->nodes()[3], test_network->nodes()[4],
      test_network->nodes()[6], test_network->nodes()[7]};
  auto pruned_network = test_network->PrunedNetworkForTesting(
      node_a1, std::move(unselected_testing_nodes));
  EXPECT_EQ(pruned_network->nodes().size(), (size_t)1);
}

TEST(TEST_BAYESIAN_NETWORK_TEST,
     NETWORK_PRUNNING_TEST_SELECT_SECOND_TEST_NODE) {
  auto test_network = ConstructNetwork1();
  // Select b1
  Node *node_b1 = test_network->nodes()[4];
  std::unordered_set<Node *> unselected_testing_nodes = {
      test_network->nodes()[3], test_network->nodes()[6],
      test_network->nodes()[7]};
  auto pruned_network = test_network->PrunedNetworkForTesting(
      node_b1, std::move(unselected_testing_nodes));
  EXPECT_EQ(pruned_network->nodes().size(), (size_t)1);
  // Select a3
  Node *node_a3 = test_network->nodes()[3];
  std::unordered_set<Node *> unselected_testing_nodes_for_a3 = {
      test_network->nodes()[4], test_network->nodes()[6],
      test_network->nodes()[7]};
  auto pruned_network_for_a3 = test_network->PrunedNetworkForTesting(
      node_a3, std::move(unselected_testing_nodes_for_a3));
  EXPECT_EQ(pruned_network_for_a3->nodes().size(), (size_t)3);
}
TEST(TEST_BAYESIAN_NETWORK_TEST, TOPOLOGICAL_SORT_TEST) {
  std::unordered_map<std::string, std::unordered_set<std::string>> parents;
  parents["b"] = {};
  parents["a"] = {"b"};
  parents["c"] = {"a", "b"};
  auto ordered_nodes =
      test_bayesian_network::util::TopologicalSortNodes(parents);
  EXPECT_THAT(ordered_nodes, testing::ElementsAre("b", "a", "c"));
}

TEST(TEST_BAYESIAN_NETWORK_TEST, LOAD_NET_FILE_TEST) {
  std::string tmp_filename = "/tmp/test_network.net";
  std::ofstream ofs(tmp_filename, std::ofstream::out);
  ASSERT_TRUE(ofs);
  ofs << test_net_file_content;
  ofs.close();
  auto tbn = TestBayesianNetwork::ParseTestBayesianNetworkFromNetFile(
      tmp_filename.c_str());
  EXPECT_EQ(tbn->nodes().size(), (size_t)3);
  EXPECT_EQ(tbn->nodes()[0]->variable()->variable_name(), "variable0");
  EXPECT_THAT(tbn->nodes()[0]->variable()->domain_names(),
              testing::ElementsAre("state0", "state1"));
  EXPECT_TRUE(tbn->nodes()[0]->parents().empty());
  EXPECT_EQ(tbn->nodes()[0]->variable()->type(), variable_type::evidence);
  EXPECT_EQ(tbn->nodes()[0]->type(), node_type::test);
  EXPECT_EQ(tbn->nodes()[1]->variable()->variable_name(), "variable1");
  EXPECT_THAT(tbn->nodes()[1]->variable()->domain_names(),
              testing::ElementsAre("state0", "state1"));
  EXPECT_THAT(tbn->nodes()[1]->parents(),
              testing::ElementsAre(tbn->nodes()[0]));
  EXPECT_EQ(tbn->nodes()[1]->variable()->type(), variable_type::hidden);
  EXPECT_EQ(tbn->nodes()[1]->type(), node_type::regular);
  EXPECT_EQ(tbn->nodes()[2]->variable()->variable_name(), "variable2");
  EXPECT_THAT(tbn->nodes()[2]->variable()->domain_names(),
              testing::ElementsAre("state0", "state1"));
  EXPECT_THAT(tbn->nodes()[2]->parents(),
              testing::ElementsAre(tbn->nodes()[0], tbn->nodes()[1]));
  EXPECT_EQ(tbn->nodes()[2]->variable()->type(), variable_type::query);
  EXPECT_EQ(tbn->nodes()[2]->type(), node_type::regular);
}
} // namespace test_bayesian_network
