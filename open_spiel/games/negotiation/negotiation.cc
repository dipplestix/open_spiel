// Copyright 2019 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "open_spiel/games/negotiation/negotiation.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <utility>

#include "open_spiel/abseil-cpp/absl/random/poisson_distribution.h"
#include "open_spiel/abseil-cpp/absl/random/uniform_int_distribution.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace negotiation {

namespace {

// Facts about the game
const GameType kGameType{
    /*short_name=*/"negotiation",
    /*long_name=*/"Negotiation",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kSampledStochastic,
    GameType::Information::kImperfectInformation,
    GameType::Utility::kGeneralSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/2,
    /*min_num_players=*/2,
    /*provides_information_state_string=*/false,
    /*provides_information_state_tensor=*/false,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/
    {{"enable_proposals", GameParameter(kDefaultEnableProposals)},
     {"enable_utterances", GameParameter(kDefaultEnableUtterances)},
     {"num_items", GameParameter(kDefaultNumItems)},
     {"num_symbols", GameParameter(kDefaultNumSymbols)},
     {"rng_seed", GameParameter(kDefaultSeed)},
     {"utterance_dim", GameParameter(kDefaultUtteranceDim)},
     {"min_quantity", GameParameter(kDefaultMinQuantity)},
     {"max_quantity", GameParameter(kDefaultMaxQuantity)},
     {"min_value", GameParameter(kDefaultMinValue)},
     {"max_value", GameParameter(kDefaultMaxValue)},
     {"quantity_mean", GameParameter(kDefaultQuantityMean)},
     {"max_rounds", GameParameter(kDefaultMaxRounds)},
     {"discount", GameParameter(kDefaultDiscount)}}};

static std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new NegotiationGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

RegisterSingleTensorObserver single_tensor(kGameType.short_name);

std::string TurnTypeToString(TurnType turn_type) {
  if (turn_type == TurnType::kProposal) {
    return "Proposal";
  } else if (turn_type == TurnType::kUtterance) {
    return "Utterance";
  } else {
    SpielFatalError("Unrecognized turn type");
  }
}
}  // namespace

std::string NegotiationState::ActionToString(Player player,
                                             Action move_id) const {
  if (player == kChancePlayerId) {
    return absl::StrCat("chance outcome ", move_id);
  }
  
  // Check if this is a walk away action
  if (move_id == parent_game_.NumDistinctActions() - 1) {
    return absl::StrCat("Walk away (get ", walk_away_values_[player], " points)");
  }
  
  std::string action_string = "";
  if (turn_type_ == TurnType::kProposal) {
    if (move_id == parent_game_.NumDistinctProposals() - 1) {
      absl::StrAppend(&action_string, "Proposal: Agreement reached!");
    } else {
      std::vector<int> proposal = DecodeProposal(move_id);
      std::string prop_str = absl::StrJoin(proposal, ", ");
      absl::StrAppend(&action_string, "Proposal: [", prop_str, "]");
    }
  } else {
    std::vector<int> utterance = DecodeUtterance(move_id);
    std::string utt_str = absl::StrJoin(utterance, ", ");
    absl::StrAppend(&action_string, ", Utterance: [", utt_str, "]");
  }
  return action_string;
}

bool NegotiationState::IsTerminal() const {
  // If utterances are enabled, force the agent to utter something even when
  // they accept the proposal or run out of steps (i.e. on ther last turn).
  bool utterance_check =
      (enable_utterances_ ? utterances_.size() == proposals_.size() : true);
  return (agreement_reached_ || proposals_.size() >= max_steps_) &&
         utterance_check;
}

std::vector<double> NegotiationState::Returns() const {
  if (!IsTerminal() || !agreement_reached_) {
    return std::vector<double>(num_players_, 0.0);
  }

  // Check if the last action was a walk away action
  if (walk_away_) {
    std::vector<double> returns;
    returns.reserve(walk_away_values_.size());
    for (int value : walk_away_values_) {
      returns.push_back(static_cast<double>(value));
    }
    // Apply discount based on complete rounds
    if (num_steps_ > 2) {
      // Calculate complete rounds - it should be (num_steps_ - 2) / 2 integer division
      int complete_rounds = (num_steps_ - 2) / 2;  // Integer division, floors the result
      double discount = std::pow(parent_game_.discount(), complete_rounds);
      for (int i = 0; i < num_players_; ++i) {
        returns[i] *= discount;
      }
    }
    return returns;
  }

  // Calculate rewards from accepted proposal
  int proposing_player = proposals_.size() % 2 == 1 ? 0 : 1;
  int other_player = 1 - proposing_player;
  const std::vector<int>& final_proposal = proposals_.back();

  // Calculate utilities for both players
  std::vector<double> returns(num_players_);
  for (int i = 0; i < num_items_; ++i) {
    returns[proposing_player] += final_proposal[i] * agent_utils_[proposing_player][i];
    returns[other_player] += (item_pool_[i] - final_proposal[i]) * agent_utils_[other_player][i];
  }

  // Apply discount based on complete rounds
  if (num_steps_ > 2) {
    // Calculate complete rounds - it should be (num_steps_ - 2) / 2 integer division
    int complete_rounds = (num_steps_ - 2) / 2;  // Integer division, floors the result
    double discount = std::pow(parent_game_.discount(), complete_rounds);
    for (int i = 0; i < num_players_; ++i) {
      returns[i] *= discount;
    }
  }

  return returns;
}

std::string NegotiationState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  if (IsChanceNode()) {
    return "ChanceNode -- no observation";
  }

  std::string str = absl::StrCat("Max steps: ", max_steps_, "\n");
  absl::StrAppend(&str, "Item pool: ", absl::StrJoin(item_pool_, " "), "\n");

  if (!agent_utils_.empty()) {
    absl::StrAppend(&str, "Agent ", player,
                    " util vec: ", absl::StrJoin(agent_utils_[player], " "),
                    "\n");
  }

  // Add walk away value to observation
  absl::StrAppend(&str, "Walk away value: ", walk_away_values_[player], "\n");

  absl::StrAppend(&str, "Current player: ", CurrentPlayer(), "\n");
  absl::StrAppend(&str, "Turn Type: ", TurnTypeToString(turn_type_), "\n");

  if (!proposals_.empty()) {
    absl::StrAppend(&str, "Most recent proposal: [",
                    absl::StrJoin(proposals_.back(), ", "), "]\n");
  }

  if (!utterances_.empty()) {
    absl::StrAppend(&str, "Most recent utterance: [",
                    absl::StrJoin(utterances_.back(), ", "), "]\n");
  }

  return str;
}

// New structure:
// [Current player (2) | Turn type (2) | Terminal status (2) | 
//  Round number (1) | Base discount factor (1) | Current round discount (1) | 
//  Item pool (num_items) | Utilities (num_items) | 
//  Walk away value (1) | // Only include the observing player's walk away value
//  Proposal history (max_rounds * 2 * num_items)]
std::vector<int> NegotiationGame::ObservationTensorShape() const {
  // New structure:
  // [Current player (2) | Turn type (2) | Terminal status (2) | 
  //  Round number (1) | Base discount factor (1) | Current round discount (1) | 
  //  Item pool (num_items) | Utilities (num_items) | 
  //  Walk away value (1) | // Only include the observing player's walk away value
  //  Proposal history (max_rounds * 2 * num_items)]
  
  // Calculate max_steps based on max_rounds
  int max_steps = max_rounds_ * 2;
  
  return {kNumPlayers + 2 + 2 + 
          1 + 1 + 1 + 
          num_items_ + num_items_ + 
          1 +  // Only include the observing player's walk away value
          max_steps * num_items_};
}

void NegotiationState::ObservationTensor(Player player,
                                       absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  SPIEL_CHECK_EQ(values.size(), parent_game_.ObservationTensorSize());
  std::fill(values.begin(), values.end(), 0);

  // No observations at chance nodes.
  if (IsChanceNode()) {
    return;
  }

  int offset = 0;

  // Current player - still using one-hot encoding (2 values)
  if (!IsTerminal()) {
    values[offset + CurrentPlayer()] = 1;
  }
  offset += kNumPlayers;

  // Current turn type - still using one-hot encoding (2 values)
  if (turn_type_ == TurnType::kProposal) {
    values[offset] = 1;
  } else {
    values[offset + 1] = 1;
  }
  offset += 2;

  // Terminal status - still using one-hot encoding (2 values)
  values[offset] = IsTerminal() ? 1 : 0;
  values[offset + 1] = agreement_reached_ ? 1 : 0;
  offset += 2;

  // Current round number (1 value)
  int current_round = (num_steps_ > 0) ? (num_steps_ - 1) / 2 : 0;
  values[offset] = static_cast<float>(current_round);
  offset += 1;

  // Base discount factor of the game (1 value)
  values[offset] = static_cast<float>(parent_game_.discount());
  offset += 1;
  
  // Current round's applied discount factor (1 value)
  double current_discount = 1.0;
  if (num_steps_ > 2) {
    int complete_rounds = (num_steps_ - 2) / 2;
    current_discount = std::pow(parent_game_.discount(), complete_rounds);
  }
  // Ensure the discount factor is stored properly as a float
  values[offset] = static_cast<float>(current_discount);
  offset += 1;

  // Item pool - direct values (num_items values)
  for (int i = 0; i < num_items_; ++i) {
    values[offset + i] = static_cast<float>(item_pool_[i]);
  }
  offset += num_items_;

  // Player utilities - direct values (num_items values)
  for (int i = 0; i < num_items_; ++i) {
    values[offset + i] = static_cast<float>(agent_utils_[player][i]);
  }
  offset += num_items_;

  // Walk away value for the observing player only (1 value)
  values[offset] = static_cast<float>(walk_away_values_[player]);
  offset += 1;

  // Proposal history (max_steps_ * num_items values)
  // Initialize all to -1 (indicating no proposal)
  for (int i = 0; i < max_steps_ * num_items_; ++i) {
    values[offset + i] = -1;
  }
  
  // Fill in proposals that have been made
  for (int p = 0; p < proposals_.size() && p < max_steps_; ++p) {
    for (int i = 0; i < num_items_; ++i) {
      values[offset + p * num_items_ + i] = static_cast<float>(proposals_[p][i]);
    }
  }
  
  SPIEL_CHECK_EQ(offset + max_steps_ * num_items_, values.size());
}

NegotiationState::NegotiationState(std::shared_ptr<const Game> game)
    : State(game),
      parent_game_(static_cast<const NegotiationGame&>(*game)),
      enable_proposals_(parent_game_.EnableProposals()),
      enable_utterances_(parent_game_.EnableUtterances()),
      num_items_(parent_game_.NumItems()),
      num_symbols_(parent_game_.NumSymbols()),
      utterance_dim_(parent_game_.UtteranceDim()),
      num_steps_(0),
      max_steps_(-1),
      agreement_reached_(false),
      cur_player_(kChancePlayerId),
      turn_type_(TurnType::kProposal),
      discount_(1.0),  // Initialize discount to 1.0
      item_pool_({}),
      agent_utils_({}),
      proposals_({}),
      utterances_({}),
      walk_away_values_({}) {}

int NegotiationState::CurrentPlayer() const {
  return IsTerminal() ? kTerminalPlayerId : cur_player_;
}

// From Sec 2.1 of the paper: "At each round (i) an item pool is sampled
// uniformly, instantiating a quantity (between 0 and 5) for each of the types
// and represented as a vector i \in {0...5}^3 and (ii) each agent j receives a
// utility function sampled uniformly, which specifies how rewarding one unit of
// each item is (with item rewards between 0 and 10, and with the constraint
// that there is at least one item with non-zero utility), represented as a
// vector u_j \in {0...10}^3".
void NegotiationState::DetermineItemPoolAndUtilities() {
  // Clear existing values
  item_pool_.clear();
  agent_utils_.clear();
  walk_away_values_.clear();

  // Use the configured max_rounds if available, otherwise sample it
  if (parent_game_.MaxRounds() > 0) {
    // Set directly from the parameter
    max_steps_ = parent_game_.MaxRounds() * 2;  // Each round has 2 steps (player 0, player 1)
  } else {
    // Generate max number of rounds (max number of steps for the episode): we
    // sample N between 4 and 10 at the start of each episode, according to a
    // truncated Poissondistribution with mean 7, as done in the Cao et al. '18
    // paper.
    max_steps_ = -1;
    absl::poisson_distribution<int> steps_dist(7.0);
    while (!(max_steps_ >= 4 && max_steps_ <= 10)) {
      max_steps_ = steps_dist(*parent_game_.RNG());
    }
  }

  // Generate the pool of items using Poisson distribution with configurable mean
  absl::poisson_distribution<int> quantity_dist(parent_game_.QuantityMean());
  for (int i = 0; i < num_items_; ++i) {
    // Generate the quantity with Poisson distribution without clamping
    int quantity = quantity_dist(*parent_game_.RNG());
    item_pool_.push_back(quantity);
  }

  // Generate agent utilities.
  absl::uniform_int_distribution<int> util_dist(parent_game_.MinValue(), parent_game_.MaxValue());
  for (int i = 0; i < num_players_; ++i) {
    agent_utils_.push_back({});
    int sum_util = 0;
    while (sum_util == 0) {
      agent_utils_[i].clear();
      for (int j = 0; j < num_items_; ++j) {
        agent_utils_[i].push_back(util_dist(*parent_game_.RNG()));
        sum_util += agent_utils_[i].back();
      }
    }
  }

  // Generate walk away values
  walk_away_values_.resize(num_players_);
  for (int i = 0; i < num_players_; ++i) {
    // Calculate maximum possible utility for this player
    int max_utility = 0;
    for (int j = 0; j < num_items_; ++j) {
      max_utility += agent_utils_[i][j] * item_pool_[j];
    }
    // Generate random walk away value between 1 and max_utility
    if (max_utility > 1) {
      absl::uniform_int_distribution<int> walk_away_dist(1, max_utility);
      walk_away_values_[i] = walk_away_dist(*parent_game_.RNG());
    } else {
      walk_away_values_[i] = 1;  // If max_utility is 1, just use 1
    }
  }
}

void NegotiationState::InitializeEpisode() {
  num_steps_ = 0;
  agreement_reached_ = false;
  walk_away_ = false;  // Reset walk away flag
  cur_player_ = 0;
  turn_type_ = TurnType::kProposal;
  proposals_.clear();
  utterances_.clear();
  
  // Generate new item pool and utilities
  DetermineItemPoolAndUtilities();
}

void NegotiationState::DoApplyAction(Action move_id) {
  if (IsChanceNode()) {
    DetermineItemPoolAndUtilities();
    cur_player_ = 0;
    turn_type_ = TurnType::kProposal;
  } else {
    // Check if this is a walk away action
    if (move_id == parent_game_.NumDistinctActions() - 1) {
      walk_away_ = true;
      agreement_reached_ = true;
    } else if (move_id == parent_game_.NumDistinctProposals() - 1) {
      walk_away_ = false;  // Explicitly set walk away to false for accept
      agreement_reached_ = true;
    } else {
      if (turn_type_ == TurnType::kProposal) {
        proposals_.push_back(DecodeProposal(move_id));
      } else {
        utterances_.push_back(DecodeUtterance(move_id));
      }
    }

    // Switch players and turn types
    if (turn_type_ == TurnType::kProposal) {
      if (enable_utterances_) {
        turn_type_ = TurnType::kUtterance;
      } else {
        cur_player_ = 1 - cur_player_;
        turn_type_ = TurnType::kProposal;
      }
    } else {
      cur_player_ = 1 - cur_player_;
      turn_type_ = TurnType::kProposal;
    }
  }
  num_steps_++;  // Increment step counter
}

bool NegotiationState::NextProposal(std::vector<int>* proposal) const {
  // Starting from the right, move left trying to increase the value. When
  // successful, increment the value and set all the right digits back to 0.
  for (int i = num_items_ - 1; i >= 0; --i) {
    if ((*proposal)[i] + 1 <= item_pool_[i]) {
      // Success!
      (*proposal)[i]++;
      for (int j = i + 1; j < num_items_; ++j) {
        (*proposal)[j] = 0;
      }
      return true;
    }
  }

  return false;
}

std::vector<int> NegotiationState::DecodeInteger(int encoded_value,
                                                 int dimensions,
                                                 int num_digit_values) const {
  std::vector<int> decoded(dimensions, 0);
  int i = dimensions - 1;
  while (encoded_value > 0) {
    SPIEL_CHECK_GE(i, 0);
    SPIEL_CHECK_LT(i, dimensions);
    decoded[i] = encoded_value % num_digit_values;
    encoded_value /= num_digit_values;
    i--;
  }
  return decoded;
}

int NegotiationState::EncodeInteger(const std::vector<int>& container,
                                    int num_digit_values) const {
  int encoded_value = 0;
  for (int digit : container) {
    encoded_value = encoded_value * num_digit_values + digit;
  }
  return encoded_value;
}

Action NegotiationState::EncodeProposal(
    const std::vector<int>& proposal) const {
  SPIEL_CHECK_EQ(proposal.size(), num_items_);
  return EncodeInteger(proposal, parent_game_.MaxQuantity() + 1);
}

Action NegotiationState::EncodeUtterance(
    const std::vector<int>& utterance) const {
  SPIEL_CHECK_EQ(utterance.size(), utterance_dim_);
  // Utterance ids are offset from zero (starting at NumDistinctProposals()).
  return parent_game_.NumDistinctProposals() +
         EncodeInteger(utterance, num_symbols_);
}

std::vector<int> NegotiationState::DecodeProposal(int encoded_proposal) const {
  return DecodeInteger(encoded_proposal, num_items_, parent_game_.MaxQuantity() + 1);
}

std::vector<int> NegotiationState::DecodeUtterance(
    int encoded_utterance) const {
  // Utterance ids are offset from zero (starting at NumDistinctProposals()).
  return DecodeInteger(encoded_utterance - parent_game_.NumDistinctProposals(),
                       utterance_dim_, num_symbols_);
}

std::vector<Action> NegotiationState::LegalActions() const {
  if (IsChanceNode()) {
    return LegalChanceOutcomes();
  } else if (IsTerminal()) {
    return {};
  } else if (turn_type_ == TurnType::kProposal) {
    std::vector<Action> legal_actions;

    // Proposals are always enabled, so first contruct them.
    std::vector<int> proposal(num_items_, 0);
    legal_actions.push_back(EncodeProposal(proposal));

    while (NextProposal(&proposal)) {
      legal_actions.push_back(EncodeProposal(proposal));
    }

    if (!proposals_.empty()) {
      // Add the agreement action only if there's been at least one proposal.
      legal_actions.push_back(parent_game_.NumDistinctProposals() - 1);
    }

    // Add walk away action
    legal_actions.push_back(parent_game_.NumDistinctActions() - 1);

    return legal_actions;
  } else {
    SPIEL_CHECK_TRUE(enable_utterances_);
    SPIEL_CHECK_FALSE(parent_game_.LegalUtterances().empty());
    return parent_game_.LegalUtterances();
  }
}

std::vector<std::pair<Action, double>> NegotiationState::ChanceOutcomes()
    const {
  SPIEL_CHECK_TRUE(IsChanceNode());
  // The game has chance mode kSampledStochastic, so there is only a single
  // outcome, and it's all randomized in the ApplyAction.
  std::vector<std::pair<Action, double>> outcomes = {std::make_pair(0, 1.0)};
  return outcomes;
}

std::string NegotiationState::ToString() const {
  if (IsChanceNode()) {
    return "Initial chance node";
  }

  std::string str = absl::StrCat("Max steps: ", max_steps_, "\n");
  absl::StrAppend(&str, "Item pool: ", absl::StrJoin(item_pool_, " "), "\n");

  if (!agent_utils_.empty()) {
    for (int i = 0; i < num_players_; ++i) {
      absl::StrAppend(&str, "Agent ", i,
                      " util vec: ", absl::StrJoin(agent_utils_[i], " "), "\n");
    }
  }

  absl::StrAppend(&str, "Current player: ", cur_player_, "\n");
  absl::StrAppend(&str, "Turn Type: ", TurnTypeToString(turn_type_), "\n");

  for (int i = 0; i < proposals_.size(); ++i) {
    absl::StrAppend(&str, "Player ", i % 2, " proposes: [",
                    absl::StrJoin(proposals_[i], ", "), "]");
    if (enable_utterances_ && i < utterances_.size()) {
      absl::StrAppend(&str, " utters: [", absl::StrJoin(utterances_[i], ", "),
                      "]");
    }
    absl::StrAppend(&str, "\n");
  }

  if (agreement_reached_) {
    absl::StrAppend(&str, "Agreement reached!\n");
  }

  return str;
}

std::unique_ptr<State> NegotiationState::Clone() const {
  return std::unique_ptr<State>(new NegotiationState(*this));
}

NegotiationGame::NegotiationGame(const GameParameters& params)
    : Game(kGameType, params),
      enable_proposals_(
          ParameterValue<bool>("enable_proposals", kDefaultEnableProposals)),
      enable_utterances_(
          ParameterValue<bool>("enable_utterances", kDefaultEnableUtterances)),
      num_items_(ParameterValue<int>("num_items", kDefaultNumItems)),
      num_symbols_(ParameterValue<int>("num_symbols", kDefaultNumSymbols)),
      utterance_dim_(
          ParameterValue<int>("utterance_dim", kDefaultUtteranceDim)),
      seed_(ParameterValue<int>("rng_seed", kDefaultSeed)),
      discount_(ParameterValue<double>("discount", kDefaultDiscount)),
      min_quantity_(ParameterValue<int>("min_quantity", kDefaultMinQuantity)),
      max_quantity_(ParameterValue<int>("max_quantity", kDefaultMaxQuantity)),
      min_value_(ParameterValue<int>("min_value", kDefaultMinValue)),
      max_value_(ParameterValue<int>("max_value", kDefaultMaxValue)),
      quantity_mean_(ParameterValue<double>("quantity_mean", kDefaultQuantityMean)),
      max_rounds_(ParameterValue<int>("max_rounds", kDefaultMaxRounds)),
      legal_utterances_({}) {
  // Use a time-based random seed when none is provided
  if (seed_ < 0) {
    // Use current time as seed for true randomness when no seed is provided
    rng_ = std::make_unique<std::mt19937>(std::random_device{}());
  } else {
    // Use the provided seed for deterministic behavior
    rng_ = std::make_unique<std::mt19937>(seed_);
  }
  ConstructLegalUtterances();
}

void NegotiationGame::ConstructLegalUtterances() {
  if (enable_utterances_) {
    legal_utterances_.resize(NumDistinctUtterances());
    for (int i = 0; i < NumDistinctUtterances(); ++i) {
      legal_utterances_[i] = NumDistinctProposals() + i;
    }
  }
}

int NegotiationGame::MaxGameLength() const {
  if (enable_utterances_) {
    return 2 * kMaxSteps;  // Every step is two turns: proposal, then utterance.
  } else {
    return kMaxSteps;
  }
}

int NegotiationGame::NumDistinctUtterances() const {
  return static_cast<int>(std::pow(num_symbols_, utterance_dim_));
}

int NegotiationGame::NumDistinctProposals() const {
  // Every slot can hold { 0, 1, ..., MaxQuantity }, and there is an extra
  // one at the end for the special "agreement reached" action.
  return static_cast<int>(std::pow(max_quantity_ + 1, num_items_)) + 1;
}

std::string NegotiationState::Serialize() const {
  if (IsChanceNode()) {
    return "chance";
  } else {
    std::string state_str = "";
    absl::StrAppend(&state_str, MaxSteps(), "\n");
    absl::StrAppend(&state_str, absl::StrJoin(ItemPool(), " "), "\n");
    for (int p = 0; p < NumPlayers(); ++p) {
      absl::StrAppend(&state_str, absl::StrJoin(AgentUtils()[p], " "), "\n");
    }
    absl::StrAppend(&state_str, HistoryString(), "\n");
    return state_str;
  }
}

std::unique_ptr<State> NegotiationGame::DeserializeState(
    const std::string& str) const {
  if (str == "chance") {
    return NewInitialState();
  } else {
    std::vector<std::string> lines = absl::StrSplit(str, '\n');
    std::unique_ptr<State> state = NewInitialState();
    SPIEL_CHECK_EQ(lines.size(), 5);
    NegotiationState& nstate = static_cast<NegotiationState&>(*state);
    // Take the chance action, but then reset the quantities. Make sure game's
    // RNG state is not advanced during deserialization so copy it beforehand
    // in order to be able to restore after the chance action.
    std::unique_ptr<std::mt19937> rng = std::make_unique<std::mt19937>(*rng_);
    nstate.ApplyAction(0);
    rng_ = std::move(rng);
    nstate.ItemPool().clear();
    nstate.AgentUtils().clear();
    // Max steps
    nstate.SetMaxSteps(std::stoi(lines[0]));
    // Item pool.
    std::vector<std::string> parts = absl::StrSplit(lines[1], ' ');
    for (const auto& part : parts) {
      nstate.ItemPool().push_back(std::stoi(part));
    }
    // Agent utilities.
    for (Player player : {0, 1}) {
      parts = absl::StrSplit(lines[2 + player], ' ');
      nstate.AgentUtils().push_back({});
      for (const auto& part : parts) {
        nstate.AgentUtils()[player].push_back(std::stoi(part));
      }
    }
    nstate.SetCurrentPlayer(0);
    // Actions.
    if (lines.size() == 5) {
      parts = absl::StrSplit(lines[4], ' ');
      // Skip the first one since it is the chance node.
      for (int i = 1; i < parts.size(); ++i) {
        Action action = static_cast<Action>(std::stoi(parts[i]));
        nstate.ApplyAction(action);
      }
    }
    return state;
  }
}

std::string NegotiationGame::GetRNGState() const {
  std::ostringstream rng_stream;
  rng_stream << *rng_;
  return rng_stream.str();
}

void NegotiationGame::SetRNGState(const std::string& rng_state) const {
  if (rng_state.empty()) return;
  std::istringstream rng_stream(rng_state);
  rng_stream >> *rng_;
}

std::unique_ptr<State> NegotiationGame::NewInitialState() const {
  // If a seed was provided, reset the RNG to ensure consistent game configurations
  if (seed_ >= 0) {
    rng_ = std::make_unique<std::mt19937>(seed_);
  }
  return std::unique_ptr<State>(new NegotiationState(shared_from_this()));
}

}  // namespace negotiation
}  // namespace open_spiel
