#ifndef CIRCUIT_NODE_H
#define CIRCUIT_NODE_H
#include <test_bayesian_network/type.h>
#include <test_bayesian_network/variable.h>
#include <vector>

namespace test_bayesian_network {
namespace test_circuit {

enum node_type {
  sum = '+',
  product = '*',
  test = '?',
  variable = 'v',
  parameter = 'p',
  threshold_parameter = 'r',
  test_probability_parameter = 'q',
  z = 'Z'
};

class SumNode;
class ProductNode;
class TestNode;
class VariableTerminalNode;
class ParameterTerminalNode;
class TestThresholdParameterTerminalNode;
class TestProbabilityParameterTerminalNode;
class ZNode;

class Node {
public:
  Node(CircuitSize node_id, std::vector<Node *> children, node_type type)
      : node_id_(node_id), children_(std::move(children)), type_(type) {}
  virtual ~Node() = default;
  CircuitSize node_id() const { return node_id_; }
  const std::vector<Node *> &children() const { return children_; }
  node_type type() const { return type_; }
  virtual SumNode *get_sum_node() { return nullptr; }
  virtual ProductNode *get_product_node() { return nullptr; }
  virtual TestNode *get_test_node() { return nullptr; }
  virtual VariableTerminalNode *get_variable_terminal_node() { return nullptr; }
  virtual ParameterTerminalNode *get_parameter_terminal_node() {
    return nullptr;
  }
  virtual bool is_leaf() const { return false; }
  virtual TestThresholdParameterTerminalNode *
  get_test_threshold_parameter_terminal_node() {
    return nullptr;
  }
  virtual TestProbabilityParameterTerminalNode *
  get_test_probability_parameter_terminal_node() {
    return nullptr;
  }
  virtual ZNode *get_z_node() { return nullptr; }

private:
  CircuitSize node_id_;
  std::vector<Node *> children_;
  node_type type_;
};

class SumNode : public Node {
public:
  SumNode(CircuitSize node_id, std::vector<Node *> children)
      : Node(node_id, std::move(children), node_type::sum) {}
  SumNode *get_sum_node() override { return this; }

private:
};

class ProductNode : public Node {
public:
  ProductNode(CircuitSize node_id, std::vector<Node *> children)
      : Node(node_id, std::move(children), node_type::product) {}
  ProductNode *get_product_node() override { return this; }
};

// TestNode should has five children. The first child is the Pr(pa,e). The
// second child is the Pr(e). The third child is the test threshold. The last
// two children are two inputs that we want to select based on the test outcome
// and test threshold.
class TestNode : public Node {
public:
  TestNode(CircuitSize node_id, std::vector<Node *> children)
      : Node(node_id, std::move(children), node_type::test) {}

  TestNode *get_test_node() override { return this; }
};

class VariableTerminalNode : public Node {
public:
  VariableTerminalNode(CircuitSize node_id, Variable *variable,
                       DomainSize value)
      : Node(node_id, {}, node_type::variable), variable_(variable),
        value_(value) {}

  VariableTerminalNode *get_variable_terminal_node() override { return this; }
  std::pair<Variable *, DomainSize> variable_and_value() const {
    return std::make_pair(variable_, value_);
  }
  bool is_leaf() const override { return true; }

  std::string Label() const {
    return variable_->variable_name() + "=" + variable_->domain_names()[value_];
  }

private:
  Variable *variable_;
  DomainSize value_;
};

class ParameterTerminalNode : public Node {
public:
  ParameterTerminalNode(CircuitSize node_id,
                        std::vector<Variable *> parent_variables,
                        Variable *child_variable,
                        std::vector<DomainSize> parent_configurations,
                        DomainSize child_configuraiton,
                        node_type type = node_type::parameter)
      : Node(node_id, {}, type), parent_variables_(std::move(parent_variables)),
        child_variable_(child_variable),
        parent_configurations_(std::move(parent_configurations)),
        child_configuration_(child_configuraiton) {}

  ParameterTerminalNode *get_parameter_terminal_node() override { return this; }

  DomainSize child_configuration() const { return child_configuration_; }

  Variable *child_variable() const { return child_variable_; }

  bool is_leaf() const override { return true; }

  virtual std::string Label() const {
    std::string result = child_variable_->variable_name() + "=" +
                         child_variable_->domain_names()[child_configuration_] +
                         " | ";
    const NodeSize parent_size = parent_variables_.size();
    for (auto i = 0; i < parent_size; ++i) {
      result +=
          parent_variables_[i]->variable_name() + "=" +
          parent_variables_[i]->domain_names()[parent_configurations_[i]] + " ";
    }
    return result;
  }

protected:
  std::vector<Variable *> parent_variables_;
  Variable *child_variable_;
  std::vector<DomainSize> parent_configurations_;
  DomainSize child_configuration_;
};

class TestThresholdParameterTerminalNode : public ParameterTerminalNode {
public:
  TestThresholdParameterTerminalNode(
      CircuitSize node_id, std::vector<Variable *> parent_variables,
      Variable *child_variable, std::vector<DomainSize> parent_configurations)
      : ParameterTerminalNode(node_id, std::move(parent_variables),
                              child_variable, std::move(parent_configurations),
                              0, node_type::threshold_parameter) {}
  TestThresholdParameterTerminalNode *
  get_test_threshold_parameter_terminal_node() override {
    return this;
  }

  bool is_leaf() const override { return true; }

  std::string Label() const override {
    std::string result = child_variable_->variable_name() + " | ";
    const NodeSize parent_size = parent_variables_.size();
    for (auto i = 0; i < parent_size; ++i) {
      result +=
          parent_variables_[i]->variable_name() + "=" +
          parent_variables_[i]->domain_names()[parent_configurations_[i]] + " ";
    }
    result += "Thres";
    return result;
  }
};

class TestProbabilityParameterTerminalNode : public ParameterTerminalNode {
public:
  TestProbabilityParameterTerminalNode(
      CircuitSize node_id, bool test_result,
      std::vector<Variable *> parent_variables, Variable *child_variable,
      std::vector<DomainSize> parent_configurations,
      DomainSize child_configuraiton)
      : ParameterTerminalNode(node_id, std::move(parent_variables),
                              child_variable, std::move(parent_configurations),
                              child_configuraiton,
                              node_type::test_probability_parameter),
        test_result_(test_result) {}

  TestProbabilityParameterTerminalNode *
  get_test_probability_parameter_terminal_node() override {
    return this;
  }

  bool test_result() const { return test_result_; }

  bool is_leaf() const override { return true; }

  std::string Label() const override {
    std::string regular_label = ParameterTerminalNode::Label();
    return regular_label + " " + (test_result_ ? "+" : "-");
  }

private:
  bool test_result_;
};

class ZNode : public Node {
public:
  ZNode(CircuitSize node_id, std::vector<Node *> children)
      : Node(node_id, std::move(children), node_type::z) {}
  ZNode *get_z_node() { return this; }
};

} // namespace test_circuit
} // namespace test_bayesian_network
#endif
