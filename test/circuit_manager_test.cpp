#include <fstream>
#include <sstream>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <test_bayesian_network/circuit_manager.h>
#include <test_bayesian_network/variable.h>
namespace test_bayesian_network {
namespace test_circuit {
TEST(CIRCUIT_MANAGER_TEST, WRITE_TO_FILE_TEST) {
  auto child_variable = std::make_unique<Variable>(1, 3, variable_type::query);
  CircuitManager cm;
  Node *param_1 = cm.NewParameterTerminalNode({}, child_variable.get(), {}, 0);
  Node *param_2 = cm.NewParameterTerminalNode({}, child_variable.get(), {}, 1);
  Node *param_3 = cm.NewParameterTerminalNode({}, child_variable.get(), {}, 2);
  Node *product_node = cm.NewProductNode({param_1, param_2, param_3});
  const char *output_tac_filename = "/tmp/write_file_test_filename.tac";
  const char *output_lmap_filename = "/tmp/write_file_test_filename.lmap";
  CircuitManager::SaveAsTacFile(product_node, output_tac_filename,
                                output_lmap_filename);
  std::ifstream test_file;
  test_file.open(output_tac_filename);
  std::stringstream str_stream;
  str_stream << test_file.rdbuf();
  const std::string tac_content_read = str_stream.str();
  const std::string tac_content_expected =
      "0 L 1\n1 L 2\n2 L 3\n3 * 3 2 1 0 \n";
  EXPECT_EQ(tac_content_read, tac_content_expected);
  test_file.close();
  test_file.open(output_lmap_filename);
  str_stream.str("");
  str_stream.clear();
  str_stream << test_file.rdbuf();
  const std::string lmap_content_read = str_stream.str();
  const std::string lmap_content_expected =
      "1 p 0 1=2 | \n2 p 0 1=1 | \n3 p 0 1=0 | \n";
  EXPECT_EQ(lmap_content_read, lmap_content_expected);
}
} // namespace test_circuit
} // namespace test_bayesian_network
