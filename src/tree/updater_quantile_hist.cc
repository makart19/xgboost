/*!
 * Copyright 2017-2022 by XGBoost Contributors
 * \file updater_quantile_hist.cc
 * \brief use quantized feature values to construct a tree
 * \author Philip Cho, Tianqi Checn, Egor Smirnov
 */
#include <dmlc/timer.h>
#include <rabit/rabit.h>

#include <algorithm>
#include <unordered_map>
#include <cmath>
#include <iomanip>
#include <memory>
#include <numeric>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "xgboost/logging.h"
#include "xgboost/tree_updater.h"

#include "constraints.h"
#include "param.h"
#include "./updater_quantile_hist.h"
#include "./split_evaluator.h"
#include "../common/random.h"
#include "../common/hist_util.h"
#include "../common/row_set.h"
#include "../common/column_matrix.h"
#include "../common/threading_utils.h"

namespace xgboost {
namespace tree {

DMLC_REGISTRY_FILE_TAG(updater_quantile_hist);

DMLC_REGISTER_PARAMETER(CPUHistMakerTrainParam);

void QuantileHistMaker::Configure(const Args& args) {
  // initialize pruner
  if (!pruner_) {
    pruner_.reset(TreeUpdater::Create("prune", ctx_, task_));
  }
  pruner_->Configure(args);
  param_.UpdateAllowUnknown(args);
  hist_maker_param_.UpdateAllowUnknown(args);
}

template <typename GradientSumT>
void QuantileHistMaker::SetBuilder(const size_t n_trees,
                                   std::unique_ptr<Builder<GradientSumT>>* builder, DMatrix* dmat) {
  builder->reset(
      new Builder<GradientSumT>(n_trees, param_, std::move(pruner_), dmat, task_, ctx_));
}

template<typename GradientSumT>
void QuantileHistMaker::CallBuilderUpdate(const std::unique_ptr<Builder<GradientSumT>>& builder,
                                          HostDeviceVector<GradientPair> *gpair,
                                          DMatrix *dmat,
                                          GHistIndexMatrix const& gmat,
                                          const std::vector<RegTree *> &trees) {
  for (auto tree : trees) {
    builder->Update(gmat, column_matrix_, gpair, dmat, tree);
  }
}

void QuantileHistMaker::Update(HostDeviceVector<GradientPair> *gpair,
                               DMatrix *dmat,
                               const std::vector<RegTree *> &trees) {
  auto it = dmat->GetBatches<GHistIndexMatrix>(HistBatch(param_)).begin();
  auto p_gmat = it.Page();
  if (dmat != p_last_dmat_ || is_gmat_initialized_ == false) {
    updater_monitor_.Start("GmatInitialization");
    column_matrix_.Init(*p_gmat, param_.sparse_threshold, ctx_->Threads());
    updater_monitor_.Stop("GmatInitialization");
    // A proper solution is puting cut matrix in DMatrix, see:
    // https://github.com/dmlc/xgboost/issues/5143
    is_gmat_initialized_ = true;
  }
  // rescale learning rate according to size of trees
  float lr = param_.learning_rate;
  param_.learning_rate = lr / trees.size();

  // build tree
  const size_t n_trees = trees.size();
  if (hist_maker_param_.single_precision_histogram) {
    if (!float_builder_) {
      this->SetBuilder(n_trees, &float_builder_, dmat);
    }
    CallBuilderUpdate(float_builder_, gpair, dmat, *p_gmat, trees);
  } else {
    if (!double_builder_) {
      SetBuilder(n_trees, &double_builder_, dmat);
    }
    CallBuilderUpdate(double_builder_, gpair, dmat, *p_gmat, trees);
  }

  param_.learning_rate = lr;

  p_last_dmat_ = dmat;
}

bool QuantileHistMaker::UpdatePredictionCache(
    const DMatrix* data, linalg::VectorView<float> out_preds) {
  if (hist_maker_param_.single_precision_histogram && float_builder_) {
      return float_builder_->UpdatePredictionCache(data, out_preds);
  } else if (double_builder_) {
      return double_builder_->UpdatePredictionCache(data, out_preds);
  } else {
      return false;
  }
}


template <typename GradientSumT>
template <typename BinIdxType, bool any_missing>
void QuantileHistMaker::Builder<GradientSumT>::InitRoot(
    const GHistIndexMatrix &gmat,
    DMatrix *p_fmat, RegTree *p_tree, const std::vector<GradientPair> &gpair_h,
    int *num_leaves, std::vector<CPUExpandEntry> *expand) {
  CPUExpandEntry node(RegTree::kRoot, p_tree->GetDepth(0), 0.0f);

  nodes_for_explicit_hist_build_.clear();
  nodes_for_subtraction_trick_.clear();
  nodes_for_explicit_hist_build_.push_back(node);

  size_t page_id = 0;
  for (auto const &gidx : p_fmat->GetBatches<GHistIndexMatrix>(HistBatch(param_))) {
      RowPartitioner &partitioner = this->partitioner_.front();

    this->histogram_builder_->template BuildHist<BinIdxType, true>(
        page_id, gidx, p_tree,
        nodes_for_explicit_hist_build_, nodes_for_subtraction_trick_, gpair_h,
        &(partitioner.GetOptPartition()),
        &(partitioner.GetNodeAssignments()));
    ++page_id;
  }

  {
    auto nid = RegTree::kRoot;
    auto hist = this->histogram_builder_->Histogram()[nid];
    GradientPairT grad_stat;
    if (data_layout_ == DataLayout::kDenseDataZeroBased ||
        data_layout_ == DataLayout::kDenseDataOneBased) {
      auto const& gmat = *(p_fmat->GetBatches<GHistIndexMatrix>(HistBatch(param_)).begin());
      const std::vector<uint32_t> &row_ptr = gmat.cut.Ptrs();
      const uint32_t ibegin = row_ptr[fid_least_bins_];
      const uint32_t iend = row_ptr[fid_least_bins_ + 1];
      auto begin = hist.data();
      for (uint32_t i = ibegin; i < iend; ++i) {
        const GradientPairT et = begin[i];
        grad_stat.Add(et.GetGrad(), et.GetHess());
      }
    } else {
      for (const GradientPair& gh : gpair_h) {
        grad_stat.Add(gh.GetGrad(), gh.GetHess());
      }
      rabit::Allreduce<rabit::op::Sum, GradientSumT>(
          reinterpret_cast<GradientSumT *>(&grad_stat), 2);
    }

    auto weight = evaluator_->InitRoot(GradStats{grad_stat});
    p_tree->Stat(RegTree::kRoot).sum_hess = grad_stat.GetHess();
    p_tree->Stat(RegTree::kRoot).base_weight = weight;
    (*p_tree)[RegTree::kRoot].SetLeaf(param_.learning_rate * weight);

    std::vector<CPUExpandEntry> entries{node};
    builder_monitor_.Start("EvaluateSplits");
    auto ft = p_fmat->Info().feature_types.ConstHostSpan();
    for (auto const& gmat : p_fmat->GetBatches<GHistIndexMatrix>(HistBatch(param_))) {
      evaluator_->EvaluateSplits(histogram_builder_->Histogram(), gmat.cut, ft,
                                 *p_tree, &entries);
      break;
    }
    builder_monitor_.Stop("EvaluateSplits");
    node = entries.front();
  }

  expand->push_back(node);
  ++(*num_leaves);
}

template<typename GradientSumT>
void QuantileHistMaker::Builder<GradientSumT>::AddSplitsToTree(
          const std::vector<CPUExpandEntry>& expand,
          RegTree *p_tree,
          int *num_leaves,
          std::vector<CPUExpandEntry>* nodes_for_apply_split,
          std::unordered_map<uint32_t, bool>* smalest_nodes_mask_ptr, size_t depth) {
  std::unordered_map<uint32_t, bool>& smalest_nodes_mask = *smalest_nodes_mask_ptr;
  const bool is_loss_guided = static_cast<TrainParam::TreeGrowPolicy>(param_.grow_policy)
                              != TrainParam::kDepthWise;
  std::vector<uint16_t> complete_node_ids;
  if (param_.max_depth == 0) {
    size_t max_nid = 0;
    int max_nid_child = 0;
    size_t it = 0;
    for (auto const& entry : expand) {
      max_nid = std::max(max_nid, static_cast<size_t>(2*entry.nid + 2));
      if (entry.IsValid(param_, *num_leaves)) {
        nodes_for_apply_split->push_back(entry);
        evaluator_->ApplyTreeSplit(entry, p_tree);
        ++(*num_leaves);
        ++it;
        max_nid_child = std::max(max_nid_child,
                                static_cast<int>(std::max((*p_tree)[entry.nid].LeftChild(),
                                (*p_tree)[entry.nid].RightChild())));
      }
    }
    (*num_leaves) -= it;
    curr_level_nodes_.clear();
    for (auto const& entry : expand) {
      if (entry.IsValid(param_, *num_leaves)) {
        (*num_leaves)++;
        curr_level_nodes_[2*entry.nid] = (*p_tree)[entry.nid].LeftChild();
        curr_level_nodes_[2*entry.nid + 1] = (*p_tree)[entry.nid].RightChild();
        complete_node_ids.push_back((*p_tree)[entry.nid].LeftChild());
        complete_node_ids.push_back((*p_tree)[entry.nid].RightChild());
        if (entry.split.left_sum.GetHess() <= entry.split.right_sum.GetHess() || is_loss_guided) {
          smalest_nodes_mask[curr_level_nodes_[2*entry.nid]] = true;
        } else {
          smalest_nodes_mask[curr_level_nodes_[2*entry.nid + 1]] = true;
        }
      } else {
        curr_level_nodes_[2*entry.nid] = static_cast<uint16_t>(1) << 15 |
                                        static_cast<uint16_t>(entry.nid);
        curr_level_nodes_[2*entry.nid + 1] = curr_level_nodes_[2*entry.nid];
      }
    }

  } else {
    for (auto const& entry : expand) {
      if (entry.IsValid(param_, *num_leaves)) {
        nodes_for_apply_split->push_back(entry);
        evaluator_->ApplyTreeSplit(entry, p_tree);
        (*num_leaves)++;
        curr_level_nodes_[2*entry.nid] = (*p_tree)[entry.nid].LeftChild();
        curr_level_nodes_[2*entry.nid + 1] = (*p_tree)[entry.nid].RightChild();
        complete_node_ids.push_back((*p_tree)[entry.nid].LeftChild());
        complete_node_ids.push_back((*p_tree)[entry.nid].RightChild());
        if (entry.split.left_sum.GetHess() <= entry.split.right_sum.GetHess() || is_loss_guided) {
          smalest_nodes_mask[curr_level_nodes_[2*entry.nid]] = true;
          smalest_nodes_mask[curr_level_nodes_[2*entry.nid + 1]] = false;
        } else {
          smalest_nodes_mask[curr_level_nodes_[2*entry.nid + 1]] = true;
          smalest_nodes_mask[curr_level_nodes_[2*entry.nid]] = false;
        }
      } else {
        curr_level_nodes_[2*entry.nid] = static_cast<uint16_t>(1) << 15 |
                                        static_cast<uint16_t>(entry.nid);
        curr_level_nodes_[2*entry.nid + 1] = curr_level_nodes_[2*entry.nid];
      }
    }
  }
  complete_trees_depth_wise_ = complete_node_ids;
}

// Split nodes to 2 sets depending on amount of rows in each node
// Histograms for small nodes will be built explicitly
// Histograms for big nodes will be built by 'Subtraction Trick'
// Exception: in distributed setting, we always build the histogram for the left child node
//    and use 'Subtraction Trick' to built the histogram for the right child node.
//    This ensures that the workers operate on the same set of tree nodes.
template <typename GradientSumT>
void QuantileHistMaker::Builder<GradientSumT>::SplitSiblings(
    const std::vector<CPUExpandEntry> &nodes_for_apply_split,
    std::vector<CPUExpandEntry> *nodes_to_evaluate, RegTree *p_tree) {
  builder_monitor_.Start("SplitSiblings");
  RowPartitioner &partitioner = this->partitioner_.front();

  // auto const& row_set_collection = this->partitioner_.front().Partitions();
  for (auto const& entry : nodes_for_apply_split) {
    int nid = entry.nid;

    const int cleft = (*p_tree)[nid].LeftChild();
    const int cright = (*p_tree)[nid].RightChild();
    const CPUExpandEntry left_node = CPUExpandEntry(cleft, p_tree->GetDepth(cleft), 0.0);
    const CPUExpandEntry right_node = CPUExpandEntry(cright, p_tree->GetDepth(cright), 0.0);
    nodes_to_evaluate->push_back(left_node);
    nodes_to_evaluate->push_back(right_node);
    bool is_loss_guide =  static_cast<TrainParam::TreeGrowPolicy>(param_.grow_policy) ==
                          TrainParam::kDepthWise ? false : true;
    if (is_loss_guide) {
      if (partitioner.GetOptPartition().GetPartitionSize(cleft) <=
          partitioner.GetOptPartition().GetPartitionSize(cright)) {
        nodes_for_explicit_hist_build_.push_back(left_node);
        nodes_for_subtraction_trick_.push_back(right_node);
      } else {
        nodes_for_explicit_hist_build_.push_back(right_node);
        nodes_for_subtraction_trick_.push_back(left_node);
      }
    } else {
      if (entry.split.left_sum.GetHess() <= entry.split.right_sum.GetHess()) {
        nodes_for_explicit_hist_build_.push_back(left_node);
        nodes_for_subtraction_trick_.push_back(right_node);
      } else {
        nodes_for_explicit_hist_build_.push_back(right_node);
        nodes_for_subtraction_trick_.push_back(left_node);
      }
    }
  }
  builder_monitor_.Stop("SplitSiblings");
}

template<typename GradientSumT>
template <typename BinIdxType, bool any_missing>
void QuantileHistMaker::Builder<GradientSumT>::ExpandTree(
    const GHistIndexMatrix& gmat,
    const common::ColumnMatrix& column_matrix,
    DMatrix* p_fmat,
    RegTree* p_tree,
    const std::vector<GradientPair>& gpair_h) {
  builder_monitor_.Start("ExpandTree");
  int num_leaves = 0;
  split_conditions_.clear();
  split_ind_.clear();
  Driver<CPUExpandEntry> driver(static_cast<TrainParam::TreeGrowPolicy>(param_.grow_policy));
  std::vector<CPUExpandEntry> expand;
  std::vector<size_t>& row_indices = *row_set_collection_.Data();
  const size_t size_threads = row_indices.size() == 0 ?
                              (gmat.row_ptr.size() - 1) : row_indices.size();
  RowPartitioner &partitioner = this->partitioner_.front();

  const_cast<common::OptPartitionBuilder&>(partitioner.GetOptPartition()).SetSlice(0,
                                           0, size_threads);
  node_ids_.resize(size_threads, 0);
  bool is_loss_guide = static_cast<TrainParam::TreeGrowPolicy>(param_.grow_policy) ==
                       TrainParam::kDepthWise ? false : true;
  curr_level_nodes_.clear();

  InitRoot<BinIdxType, any_missing>(gmat, p_fmat, p_tree, gpair_h, &num_leaves, &expand);
  driver.Push(expand[0]);
  complete_trees_depth_wise_.clear();
  complete_trees_depth_wise_.emplace_back(0);
  int32_t depth = 0;
  while (!driver.IsEmpty()) {
    std::unordered_map<uint32_t, bool> smalest_nodes_mask;
    expand = driver.Pop();
    depth = expand[0].depth + 1;
    std::vector<CPUExpandEntry> nodes_for_apply_split;
    std::vector<CPUExpandEntry> nodes_to_evaluate;
    nodes_for_explicit_hist_build_.clear();
    nodes_for_subtraction_trick_.clear();
    AddSplitsToTree(expand, p_tree, &num_leaves, &nodes_for_apply_split,
                    &smalest_nodes_mask, depth);

    if (nodes_for_apply_split.size() != 0) {
      builder_monitor_.Start("ApplySplit");
      RowPartitioner &partitioner = this->partitioner_.front();
      if (is_loss_guide) {
        if (gmat.cut.HasCategorical()) {
          partitioner.UpdatePosition<any_missing, BinIdxType, true, true>(this->ctx_, gmat,
                      column_matrix,
                      nodes_for_apply_split, p_tree,
                      depth,
                      &smalest_nodes_mask,
                      is_loss_guide,
                      &split_conditions_,
                      &split_ind_, param_.max_depth,
                      &complete_trees_depth_wise_,
                      &curr_level_nodes_);
        } else {
          partitioner.UpdatePosition<any_missing, BinIdxType, true, false>(this->ctx_, gmat,
                      column_matrix,
                      nodes_for_apply_split, p_tree,
                      depth,
                      &smalest_nodes_mask,
                      is_loss_guide,
                      &split_conditions_,
                      &split_ind_, param_.max_depth,
                      &complete_trees_depth_wise_,
                      &curr_level_nodes_);
        }
      } else {
        if (gmat.cut.HasCategorical()) {
          partitioner.UpdatePosition<any_missing, BinIdxType, false, true>(this->ctx_, gmat,
                      column_matrix,
                      nodes_for_apply_split, p_tree,
                      depth,
                      &smalest_nodes_mask,
                      is_loss_guide,
                      &split_conditions_,
                      &split_ind_, param_.max_depth,
                      &complete_trees_depth_wise_,
                      &curr_level_nodes_);
        } else {
          partitioner.UpdatePosition<any_missing, BinIdxType, false, false>(this->ctx_, gmat,
                      column_matrix,
                      nodes_for_apply_split, p_tree,
                      depth,
                      &smalest_nodes_mask,
                      is_loss_guide,
                      &split_conditions_,
                      &split_ind_, param_.max_depth,
                      &complete_trees_depth_wise_,
                      &curr_level_nodes_);
        }
      }
      builder_monitor_.Stop("ApplySplit");
      SplitSiblings(nodes_for_apply_split, &nodes_to_evaluate, p_tree);
      if (param_.max_depth == 0 || depth < param_.max_depth) {
        size_t i = 0;
        RowPartitioner &partitioner = this->partitioner_.front();
        builder_monitor_.Start("BuildHist");
        for (auto const &gidx : p_fmat->GetBatches<GHistIndexMatrix>(HistBatch(param_))) {
          this->histogram_builder_->template BuildHist<BinIdxType, false>(
              i, gidx, p_tree,
              nodes_for_explicit_hist_build_, nodes_for_subtraction_trick_,
              gpair_h, &(partitioner.GetOptPartition()), &(partitioner.GetNodeAssignments()));
          ++i;
        }
        builder_monitor_.Stop("BuildHist");
        builder_monitor_.Start("EvaluateSplits");
        auto ft = p_fmat->Info().feature_types.ConstHostSpan();
        evaluator_->EvaluateSplits(this->histogram_builder_->Histogram(),
                                  gmat.cut, ft, *p_tree, &nodes_to_evaluate);
        builder_monitor_.Stop("EvaluateSplits");
      }
      for (size_t i = 0; i < nodes_for_apply_split.size(); ++i) {
        CPUExpandEntry left_node = nodes_to_evaluate.at(i * 2 + 0);
        CPUExpandEntry right_node = nodes_to_evaluate.at(i * 2 + 1);
        driver.Push(left_node);
        driver.Push(right_node);
      }
    }
  }
  builder_monitor_.Stop("ExpandTree");
}

template <typename GradientSumT>
void QuantileHistMaker::Builder<GradientSumT>::Update(
    const GHistIndexMatrix &gmat,
    const common::ColumnMatrix &column_matrix,
    HostDeviceVector<GradientPair> *gpair,
    DMatrix *p_fmat, RegTree *p_tree) {
  builder_monitor_.Start("Update");

  std::vector<GradientPair>* gpair_ptr = &(gpair->HostVector());
  // in case 'num_parallel_trees != 1' no posibility to change initial gpair
  if (GetNumberOfTrees() != 1) {
    gpair_local_.resize(gpair_ptr->size());
    gpair_local_ = *gpair_ptr;
    gpair_ptr = &gpair_local_;
  }
  p_last_fmat_mutable_ = p_fmat;

  switch (column_matrix.GetTypeSize()) {
    case common::kUint8BinsTypeSize:
      this->InitData<uint8_t>(gmat, column_matrix, *p_fmat, *p_tree, gpair_ptr);
      if (column_matrix.AnyMissing()) {
        ExpandTree<uint8_t, true>(gmat, column_matrix, p_fmat, p_tree, *gpair_ptr);
      } else {
        ExpandTree<uint8_t, false>(gmat, column_matrix, p_fmat, p_tree, *gpair_ptr);
      }
      break;
    case common::kUint16BinsTypeSize:
      this->InitData<uint16_t>(gmat, column_matrix, *p_fmat, *p_tree, gpair_ptr);
      if (column_matrix.AnyMissing()) {
        ExpandTree<uint16_t, true>(gmat, column_matrix, p_fmat, p_tree, *gpair_ptr);
      } else {
        ExpandTree<uint16_t, false>(gmat, column_matrix, p_fmat, p_tree, *gpair_ptr);
      }
      break;
    case common::kUint32BinsTypeSize:
      this->InitData<uint32_t>(gmat, column_matrix, *p_fmat, *p_tree, gpair_ptr);
      if (column_matrix.AnyMissing()) {
        ExpandTree<uint32_t, true>(gmat, column_matrix, p_fmat, p_tree, *gpair_ptr);
      } else {
        ExpandTree<uint32_t, false>(gmat, column_matrix, p_fmat, p_tree, *gpair_ptr);
      }
      break;
    default:
      CHECK(false);  // no default behavior
  }

  pruner_->Update(gpair, p_fmat, std::vector<RegTree*>{p_tree});

  builder_monitor_.Stop("Update");
}

template<typename GradientSumT>
bool QuantileHistMaker::Builder<GradientSumT>::UpdatePredictionCache(
    const DMatrix* data,
    linalg::VectorView<float> out_preds) {
  // p_last_fmat_ is a valid pointer as long as UpdatePredictionCache() is called in
  // conjunction with Update().
  if (!p_last_fmat_ || !p_last_tree_ || data != p_last_fmat_ ||
      p_last_fmat_ != p_last_fmat_mutable_) {
    return false;
  }
  builder_monitor_.Start("UpdatePredictionCache");
  CHECK_GT(out_preds.Size(), 0U);
  RowPartitioner &partitioner = this->partitioner_.front();
  common::BlockedSpace2d space(1, [&](size_t node) {
    return partitioner.GetNodeAssignments().size();
  }, 1024);
    common::ParallelFor2d(space, this->ctx_->Threads(), [&](size_t node, common::Range1d r) {
      int tid = omp_get_thread_num();
      for (size_t it = r.begin(); it <  r.end(); ++it) {
        bst_float leaf_value;
        // if a node is marked as deleted by the pruner, traverse upward to locate
        // a non-deleted leaf.
        int nid = (~(static_cast<uint16_t>(1) << 15)) & partitioner.GetNodeAssignments()[it];
        if ((*p_last_tree_)[nid].IsDeleted()) {
          while ((*p_last_tree_)[nid].IsDeleted()) {
            nid = (*p_last_tree_)[nid].Parent();
          }
          CHECK((*p_last_tree_)[nid].IsLeaf());
        }
        leaf_value = (*p_last_tree_)[nid].LeafValue();
        out_preds(it) += leaf_value;
      }
    });
  builder_monitor_.Stop("UpdatePredictionCache");
  return true;
}

template <typename GradientSumT>
void QuantileHistMaker::Builder<GradientSumT>::InitSampling(const DMatrix& fmat,
                                                            std::vector<GradientPair>* gpair) {
  const auto& info = fmat.Info();
  auto& rnd = common::GlobalRandom();
  std::vector<GradientPair>& gpair_ref = *gpair;

#if XGBOOST_CUSTOMIZE_GLOBAL_PRNG
  std::bernoulli_distribution coin_flip(param_.subsample);
  for (size_t i = 0; i < info.num_row_; ++i) {
    if (!(gpair_ref[i].GetHess() >= 0.0f && coin_flip(rnd)) || gpair_ref[i].GetGrad() == 0.0f) {
      gpair_ref[i] = GradientPair(0);
    }
  }
#else
  uint64_t initial_seed = rnd();

  auto n_threads = static_cast<size_t>(ctx_->Threads());
  const size_t discard_size = info.num_row_ / n_threads;
  std::bernoulli_distribution coin_flip(param_.subsample);

  dmlc::OMPException exc;
  #pragma omp parallel num_threads(n_threads)
  {
    exc.Run([&]() {
      const size_t tid = omp_get_thread_num();
      const size_t ibegin = tid * discard_size;
      const size_t iend = (tid == (n_threads - 1)) ? info.num_row_ : ibegin + discard_size;
      RandomReplace::MakeIf([&](size_t i, RandomReplace::EngineT& eng) {
        return !(gpair_ref[i].GetHess() >= 0.0f && coin_flip(eng));
      }, GradientPair(0), initial_seed, ibegin, iend, &gpair_ref);
    });
  }
  exc.Rethrow();
#endif  // XGBOOST_CUSTOMIZE_GLOBAL_PRNG
}
template<typename GradientSumT>
size_t QuantileHistMaker::Builder<GradientSumT>::GetNumberOfTrees() {
  return n_trees_;
}

template <typename GradientSumT>
template <typename BinIdxType>
void QuantileHistMaker::Builder<GradientSumT>::InitData(const GHistIndexMatrix& gmat,
                                          const common::ColumnMatrix& column_matrix,
                                          const DMatrix& fmat,
                                          const RegTree& tree,
                                          std::vector<GradientPair>* gpair) {
  builder_monitor_.Start("InitData");
  const auto& info = fmat.Info();

  {
    // initialize histogram collection
    uint32_t nbins = gmat.cut.Ptrs().back();
    // initialize histogram builder
    dmlc::OMPException exc;
    exc.Rethrow();
    this->histogram_builder_->Reset(nbins, param_.max_bin, this->ctx_->Threads(), 1,
                                    param_.max_depth, rabit::IsDistributed());

    if (param_.subsample < 1.0f) {
      CHECK_EQ(param_.sampling_method, TrainParam::kUniform)
        << "Only uniform sampling is supported, "
        << "gradient-based sampling is only support by GPU Hist.";
      builder_monitor_.Start("InitSampling");
      InitSampling(fmat, gpair);
      builder_monitor_.Stop("InitSampling");
      // We should check that the partitioning was done correctly
      // and each row of the dataset fell into exactly one of the categories
    }
    const bool is_lossguide = static_cast<TrainParam::TreeGrowPolicy>(param_.grow_policy) !=
                              TrainParam::kDepthWise;

    if (!partition_is_initiated_) {
      partitioner_.clear();
      partitioner_.emplace_back(this->ctx_, gmat, column_matrix, &tree,
                                param_.max_depth, is_lossguide);
      partition_is_initiated_ = true;
    } else {
      partitioner_.front().Reset(this->ctx_, gmat, column_matrix, &tree,
                                param_.max_depth, is_lossguide);
    }

    const size_t block_size = common::GetBlockSize(info.num_row_, this->ctx_->Threads());
  }
  {
    /* determine layout of data */
    const size_t nrow = info.num_row_;
    const size_t ncol = info.num_col_;
    const size_t nnz = info.num_nonzero_;
    // number of discrete bins for feature 0
    const uint32_t nbins_f0 = gmat.cut.Ptrs()[1] - gmat.cut.Ptrs()[0];
    if (nrow * ncol == nnz) {
      // dense data with zero-based indexing
      data_layout_ = DataLayout::kDenseDataZeroBased;
    } else if (nbins_f0 == 0 && nrow * (ncol - 1) == nnz) {
      // dense data with one-based indexing
      data_layout_ = DataLayout::kDenseDataOneBased;
    } else {
      // sparse data
      data_layout_ = DataLayout::kSparseData;
    }
  }
  // store a pointer to the tree
  p_last_tree_ = &tree;
  if (data_layout_ == DataLayout::kDenseDataOneBased) {
    evaluator_.reset(new HistEvaluator<GradientSumT, CPUExpandEntry>{
        param_, info, this->ctx_->Threads(), column_sampler_, task_, true});
  } else {
    evaluator_.reset(new HistEvaluator<GradientSumT, CPUExpandEntry>{
        param_, info, this->ctx_->Threads(), column_sampler_, task_, false});
  }

  if (data_layout_ == DataLayout::kDenseDataZeroBased
      || data_layout_ == DataLayout::kDenseDataOneBased) {
    /* specialized code for dense data:
       choose the column that has a least positive number of discrete bins.
       For dense data (with no missing value),
       the sum of gradient histogram is equal to snode[nid] */
    const std::vector<uint32_t>& row_ptr = gmat.cut.Ptrs();
    const auto nfeature = static_cast<bst_uint>(row_ptr.size() - 1);
    uint32_t min_nbins_per_feature = 0;
    for (bst_uint i = 0; i < nfeature; ++i) {
      const uint32_t nbins = row_ptr[i + 1] - row_ptr[i];
      if (nbins > 0) {
        if (min_nbins_per_feature == 0 || min_nbins_per_feature > nbins) {
          min_nbins_per_feature = nbins;
          fid_least_bins_ = i;
        }
      }
    }
    CHECK_GT(min_nbins_per_feature, 0U);
  }

  builder_monitor_.Stop("InitData");
}

template struct QuantileHistMaker::Builder<float>;
template struct QuantileHistMaker::Builder<double>;

XGBOOST_REGISTER_TREE_UPDATER(FastHistMaker, "grow_fast_histmaker")
.describe("(Deprecated, use grow_quantile_histmaker instead.)"
          " Grow tree using quantized histogram.")
.set_body(
    [](ObjInfo task) {
      LOG(WARNING) << "grow_fast_histmaker is deprecated, "
                   << "use grow_quantile_histmaker instead.";
      return new QuantileHistMaker(task);
    });

XGBOOST_REGISTER_TREE_UPDATER(QuantileHistMaker, "grow_quantile_histmaker")
.describe("Grow tree using quantized histogram.")
.set_body(
    [](ObjInfo task) {
      return new QuantileHistMaker(task);
    });
}  // namespace tree
}  // namespace xgboost
