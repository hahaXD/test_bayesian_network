#ifndef VARIABLE_H
#define VARIABLE_H
#include <string>
#include <vector>

#include <test_bayesian_network/type.h>

namespace test_bayesian_network {
enum variable_type { evidence = 'e', query = 'q', hidden = 'h' };

class Variable {
public:
  Variable(NodeSize variable_index, DomainSize domain_size, variable_type type)
      : variable_index_(variable_index), domain_size_(domain_size), type_(type),
        variable_name_(std::to_string(variable_index_)), domain_names_() {
    for (auto i = 0; i < domain_size_; ++i) {
      domain_names_.push_back(std::to_string(i));
    }
  }
  NodeSize variable_index() const { return variable_index_; }
  DomainSize domain_size() const { return domain_size_; }
  variable_type type() const { return type_; }

  void set_names(std::string variable_name,
                 std::vector<std::string> domain_names) {
    variable_name_ = std::move(variable_name);
    domain_names_ = std::move(domain_names);
  }

  const std::string variable_name() const { return variable_name_; }

  const std::vector<std::string> &domain_names() const { return domain_names_; }

private:
  NodeSize variable_index_;
  DomainSize domain_size_;
  variable_type type_;
  // name mapping
  std::string variable_name_;
  std::vector<std::string> domain_names_;
};
} // namespace test_bayesian_network
#endif
