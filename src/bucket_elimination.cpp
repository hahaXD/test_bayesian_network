#include <test_bayesian_network/bucket_elimination.h>
#include <test_bayesian_network/circuit_manager.h>
#include <unordered_map>

namespace {
using test_bayesian_network::NodeSize;
using test_bayesian_network::Variable;
std::vector<Variable *>
MergeOrderedVariables(const std::vector<Variable *> &first_variables,
                      const std::vector<Variable *> &second_variables,
                      const std::unordered_map<NodeSize, NodeSize>
                          &variable_index_to_elimination_order) {
  std::vector<Variable *> result;
  auto first_it = first_variables.begin();
  auto second_it = second_variables.begin();
  while (first_it != first_variables.end() ||
         second_it != second_variables.end()) {
    if (first_it == first_variables.end()) {
      result.push_back(*second_it);
      ++second_it;
      continue;
    }
    if (second_it == second_variables.end()) {
      result.push_back(*first_it);
      ++first_it;
      continue;
    }
    if ((*first_it) == (*second_it)) {
      result.push_back(*first_it);
      ++first_it;
      ++second_it;
      continue;
    }
    NodeSize first_variable_order =
        variable_index_to_elimination_order.find((*first_it)->variable_index())
            ->second;
    NodeSize second_variable_order = variable_index_to_elimination_order
                                         .find((*second_it)->variable_index())
                                         ->second;
    if (first_variable_order < second_variable_order) {
      result.push_back(*first_it);
      ++first_it;
      continue;
    }
    // first_variable_order > second_variable_order
    result.push_back(*second_it);
    ++second_it;
    continue;
  }
  return result;
}
} // namespace

namespace test_bayesian_network {
using test_circuit::CircuitManager;
FactorForElimination::FactorForElimination(
    std::unique_ptr<CircuitFactor> circuit_factor,
    std::vector<Variable *> variables_under_elimination_order)
    : circuit_factor_(std::move(circuit_factor)),
      variables_under_elimination_order_(
          std::move(variables_under_elimination_order)) {}

std::unique_ptr<FactorForElimination>
FactorForElimination::GetFactorForEliminationFromTotalEliminationOrder(
    std::unique_ptr<CircuitFactor> circuit_factor,
    const std::vector<Variable *> &total_elimination_order) {
  std::unordered_map<NodeSize, Variable *> variables;
  for (Variable *cur_variable : circuit_factor->variables()) {
    variables[cur_variable->variable_index()] = cur_variable;
  }
  std::vector<Variable *> variables_under_elimination_order;
  for (Variable *cur_variable : total_elimination_order) {
    auto cur_variable_it = variables.find(cur_variable->variable_index());
    if (cur_variable_it != variables.end()) {
      variables_under_elimination_order.push_back(cur_variable_it->second);
    }
  }
  return std::make_unique<FactorForElimination>(
      std::move(circuit_factor), std::move(variables_under_elimination_order));
}

std::unique_ptr<FactorForElimination>
FactorForElimination::SumOut(test_circuit::CircuitManager *circuit_manager) {
  if (variables_under_elimination_order_.empty()) {
    // returns nullptr if no variable to sum out.
    return nullptr;
  }
  auto result_factor = circuit_factor_->SumOut(
      variables_under_elimination_order_[0], circuit_manager);
  std::vector<Variable *> result_variables_under_elimination_order(
      variables_under_elimination_order_.begin() + 1,
      variables_under_elimination_order_.end());
  return std::make_unique<FactorForElimination>(
      std::move(result_factor),
      std::move(result_variables_under_elimination_order));
}

std::unique_ptr<FactorForElimination>
FactorForElimination::Multiply(const FactorForElimination &other,
                               test_circuit::CircuitManager *circuit_manager,
                               const std::unordered_map<NodeSize, NodeSize>
                                   &variable_index_to_elimination_order) {
  std::vector<Variable *> result_variables_under_elimination_order =
      MergeOrderedVariables(variables_under_elimination_order_,
                            other.variables_under_elimination_order_,
                            variable_index_to_elimination_order);
  auto multiplied_factor =
      circuit_factor_->Multiply(*other.circuit_factor(), circuit_manager);
  return std::make_unique<FactorForElimination>(
      std::move(multiplied_factor),
      std::move(result_variables_under_elimination_order));
}

Variable *FactorForElimination::NextVariableToEliminate() const {
  if (variables_under_elimination_order_.empty()) {
    return nullptr;
  }
  return variables_under_elimination_order_[0];
}

BucketElimination::BucketElimination(
    std::vector<CircuitFactor *> circuit_factors,
    std::vector<Variable *> elimination_order, CircuitManager *circuit_manager)
    : variable_index_to_elimination_order_(),
      elimination_order_(std::move(elimination_order)),
      circuit_factors_(std::move(circuit_factors)),
      circuit_manager_(circuit_manager) {
  const size_t elimination_size = elimination_order_.size();
  for (size_t i = 0; i < elimination_size; ++i) {
    variable_index_to_elimination_order_[elimination_order_[i]
                                             ->variable_index()] = i;
  }
} // namespace test_bayesian_network

std::unique_ptr<CircuitFactor> BucketElimination::Run() {
  const NodeSize num_variables_to_eliminate =
      variable_index_to_elimination_order_.size();
  std::vector<std::vector<std::unique_ptr<FactorForElimination>>> buckets(
      num_variables_to_eliminate + 1);
  for (CircuitFactor *cur_circuit_factor : circuit_factors_) {
    std::unique_ptr<CircuitFactor> cur_copy = std::make_unique<CircuitFactor>(
        cur_circuit_factor->variables(), cur_circuit_factor->factor_nodes());
    auto cur_factor_for_elimination =
        FactorForElimination::GetFactorForEliminationFromTotalEliminationOrder(
            std::move(cur_copy), elimination_order_);
    AddFactorToBucket(std::move(cur_factor_for_elimination), &buckets);
  }
  for (NodeSize i = 0; i < num_variables_to_eliminate; ++i) {
    const auto &factors_to_be_multiplied = buckets[i];
    if (factors_to_be_multiplied.empty())
      continue;
    const NodeSize factors_to_be_multiplied_size =
        factors_to_be_multiplied.size();
    if (factors_to_be_multiplied_size == 1) {
      auto next_factor = factors_to_be_multiplied[0]->SumOut(circuit_manager_);
      AddFactorToBucket(std::move(next_factor), &buckets);
      continue;
    }
    auto factors_multiplied = factors_to_be_multiplied[0]->Multiply(
        *factors_to_be_multiplied[1].get(), circuit_manager_,
        variable_index_to_elimination_order_);
    for (auto j = 2; j < factors_to_be_multiplied_size; ++j) {
      factors_multiplied = factors_multiplied->Multiply(
          *factors_to_be_multiplied[j].get(), circuit_manager_,
          variable_index_to_elimination_order_);
    }
    auto next_factor = factors_multiplied->SumOut(circuit_manager_);
    AddFactorToBucket(std::move(next_factor), &buckets);
  }
  if (buckets.back().empty()) {
    return nullptr;
  }
  if (buckets.back().size() == 1) {
    std::unique_ptr<FactorForElimination> result = std::move(buckets.back()[0]);
    std::unique_ptr<CircuitFactor> circuit_result =
        result->ReleaseCircuitFactor();
    return circuit_result;
  }
  auto multiplied_circuit =
      buckets.back()[0]->Multiply(*buckets.back()[1].get(), circuit_manager_,
                                  variable_index_to_elimination_order_);
  for (NodeSize i = 2; i < (NodeSize)buckets.back().size(); ++i) {
    multiplied_circuit =
        multiplied_circuit->Multiply(*buckets.back()[i].get(), circuit_manager_,
                                     variable_index_to_elimination_order_);
  }
  return multiplied_circuit->ReleaseCircuitFactor();
}

void BucketElimination::AddFactorToBucket(
    std::unique_ptr<FactorForElimination> factor_to_add,
    std::vector<std::vector<std::unique_ptr<FactorForElimination>>> *buckets) {
  const NodeSize num_variables_to_eliminate =
      variable_index_to_elimination_order_.size();
  Variable *next_varaible_to_eliminate =
      factor_to_add->NextVariableToEliminate();
  if (next_varaible_to_eliminate == nullptr) {
    (*buckets)[num_variables_to_eliminate].push_back(std::move(factor_to_add));
  } else {
    (*buckets)[variable_index_to_elimination_order_
                   .find(next_varaible_to_eliminate->variable_index())
                   ->second]
        .push_back(std::move(factor_to_add));
  }
}
} // namespace test_bayesian_network
