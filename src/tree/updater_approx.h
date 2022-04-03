/*!
 * Copyright 2021 XGBoost contributors
 *
 * \brief Implementation for the approx tree method.
 */
#ifndef XGBOOST_TREE_UPDATER_APPROX_H_
#define XGBOOST_TREE_UPDATER_APPROX_H_

#include <limits>
#include <utility>
#include <vector>
#include <algorithm>
#include <unordered_map>

#include "../common/partition_builder.h"
#include "../common/opt_partition_builder.h"
#include "../common/column_matrix.h"
#include "../common/random.h"
#include "constraints.h"
#include "driver.h"
#include "hist/evaluate_splits.h"
#include "hist/expand_entry.h"
#include "hist/param.h"
#include "param.h"
#include "xgboost/json.h"
#include "xgboost/tree_updater.h"

namespace xgboost {
namespace tree {
class RowPartitioner {
  common::OptPartitionBuilder opt_partition_builder_;
  std::vector<uint16_t> node_ids_;

 public:
  bst_row_t base_rowid = 0;
  bool is_loss_guided = false;

 public:
  /**
   * \brief Turn split values into discrete bin indices.
   */
  void FindSplitConditions(const std::vector<CPUExpandEntry> &nodes,
                           const RegTree &tree, const GHistIndexMatrix &gmat,
                           std::unordered_map<uint32_t, int32_t> *split_conditions) {
    for (const auto& node : nodes) {
      const int32_t nid = node.nid;
      const bst_uint fid = tree[nid].SplitIndex();
      const bst_float split_pt = tree[nid].SplitCond();
      const uint32_t lower_bound = gmat.cut.Ptrs()[fid];
      const uint32_t upper_bound = gmat.cut.Ptrs()[fid + 1];
      int32_t split_cond = -1;
      // convert floating-point split_pt into corresponding bin_id
      // split_cond = -1 indicates that split_pt is less than all known cut points
      CHECK_LT(upper_bound, static_cast<uint32_t>(std::numeric_limits<int32_t>::max()));
      for (uint32_t bound = lower_bound; bound < upper_bound; ++bound) {
        if (split_pt == gmat.cut.Values()[bound]) {
          split_cond = static_cast<int32_t>(bound);
        }
      }
      (*split_conditions)[nid] = split_cond;
    }
  }

  template <bool any_missing, typename BinIdxType,
            bool is_loss_guided, bool any_cat>
  void UpdatePosition(GenericParameter const* ctx, GHistIndexMatrix const& gmat,
                      common::ColumnMatrix const& column_matrix,
                      std::vector<CPUExpandEntry> const& nodes, RegTree const* p_tree,
                      int depth,
                      std::unordered_map<uint32_t, bool>* smalest_nodes_mask_ptr,
                      const bool loss_guide,
                      std::unordered_map<uint32_t, int32_t>* split_conditions_,
                      std::unordered_map<uint32_t, uint64_t>* split_ind_, const size_t max_depth,
                      std::vector<uint16_t>* complete_trees_depth_wise_,
                      std::unordered_map<uint32_t, uint16_t>* curr_level_nodes_,
                      bool is_left_small = true,
                      bool check_is_left_small = false) {
    // 1. Find split condition for each split
    const size_t n_nodes = nodes.size();
    FindSplitConditions(nodes, *p_tree, gmat, split_conditions_);
    // 2.1 Create a blocked space of size SUM(samples in each node)
    const uint32_t* offsets = gmat.index.Offset();
    const uint64_t rows_offset = gmat.row_ptr.size() - 1;
    std::vector<uint32_t> split_nodes(n_nodes, 0);
    for (size_t i = 0; i < n_nodes; ++i) {
        const int32_t nid = nodes[i].nid;
        split_nodes[i] = nid;
        const uint64_t fid = (*p_tree)[nid].SplitIndex();
        (*split_ind_)[nid] = fid*((gmat.IsDense() ? rows_offset : 1));
        (*split_conditions_)[nid] = (*split_conditions_)[nid] - gmat.cut.Ptrs()[fid];
    }
    std::vector<uint16_t> curr_level_nodes_data_vec;
    std::vector<uint64_t> split_ind_data_vec;
    std::vector<int32_t> split_conditions_data_vec;
    std::vector<bool> smalest_nodes_mask_vec;
    if (max_depth != 0) {
      curr_level_nodes_data_vec.resize((1 << (max_depth + 2)), 0);
      split_ind_data_vec.resize((1 << (max_depth + 2)), 0);
      split_conditions_data_vec.resize((1 << (max_depth + 2)), 0);
      smalest_nodes_mask_vec.resize((1 << (max_depth + 2)), 0);
      for (size_t nid = 0; nid < (1 << (max_depth + 2)); ++nid) {
        curr_level_nodes_data_vec[nid] = (*curr_level_nodes_)[nid];
        split_ind_data_vec[nid] = (*split_ind_)[nid];
        split_conditions_data_vec[nid] = (*split_conditions_)[nid];
        smalest_nodes_mask_vec[nid] = (*smalest_nodes_mask_ptr)[nid];
      }
    }
    const size_t n_features = gmat.cut.Ptrs().size() - 1;
    int nthreads = ctx->Threads();
    nthreads = std::max(nthreads, 1);
    const size_t depth_begin = opt_partition_builder_.DepthBegin(*complete_trees_depth_wise_,
                                                                p_tree, loss_guide, depth);
    const size_t depth_size = opt_partition_builder_.DepthSize(gmat, *complete_trees_depth_wise_,
                                                              p_tree, loss_guide, depth);

    auto const& index = gmat.index;
    auto const& cut_values = gmat.cut.Values();
    auto const& cut_ptrs = gmat.cut.Ptrs();
    RegTree const tree = *p_tree;
    auto pred = [&](auto ridx, auto bin_id, auto nid, auto split_cond) {
      if (!any_cat) {
        return bin_id <= split_cond;
      }
      bool is_cat = tree.GetSplitTypes()[nid] == FeatureType::kCategorical;
      if (any_cat && is_cat) {
        const bst_uint fid = tree[nid].SplitIndex();
        const bool default_left = tree[nid].DefaultLeft();
        // const auto column_ptr = column_matrix.GetColumn<BinIdxType, any_missing>(fid);
        auto node_cats = tree.NodeCats(nid);
        auto begin = gmat.RowIdx(ridx);
        auto end = gmat.RowIdx(ridx + 1);
        auto f_begin = cut_ptrs[fid];
        auto f_end = cut_ptrs[fid + 1];
        // bypassing the column matrix as we need the cut value instead of bin idx for categorical
        // features.
        auto gidx = BinarySearchBin(begin, end, index, f_begin, f_end);
        bool go_left;
        if (gidx == -1) {
          go_left = default_left;
        } else {
          go_left = Decision(node_cats, cut_values[gidx], default_left);
        }
        return go_left;
      } else {
        return bin_id <= split_cond;
      }
    };
    if (max_depth != 0) {
    #pragma omp parallel num_threads(nthreads)
      {
        size_t tid = omp_get_thread_num();
        const BinIdxType* numa = tid < nthreads/2 ?
          reinterpret_cast<const BinIdxType*>(column_matrix.GetIndexData()) :
          reinterpret_cast<const BinIdxType*>(column_matrix.GetIndexData());
          // reinterpret_cast<const BinIdxType*>(column_matrix.GetIndexSecondData());
        size_t chunck_size = common::GetBlockSize(depth_size, nthreads);
        size_t thread_size = chunck_size;
        size_t begin = thread_size * tid;
        size_t end = std::min(begin + thread_size, depth_size);
        begin += depth_begin;
        end += depth_begin;
        opt_partition_builder_.template CommonPartition<BinIdxType,
                                                        is_loss_guided,
                                                        !any_missing,
                                                        any_cat>(tid, begin, end, numa,
                                                                node_ids_.data(),
                                                                &split_conditions_data_vec,
                                                                &split_ind_data_vec,
                                                                &smalest_nodes_mask_vec,
                                                                &curr_level_nodes_data_vec,
                                                                column_matrix,
                                                                split_nodes, pred, depth);
      }
    } else {
    #pragma omp parallel num_threads(nthreads)
      {
        size_t tid = omp_get_thread_num();
        const BinIdxType* numa = tid < nthreads/2 ?
          reinterpret_cast<const BinIdxType*>(column_matrix.GetIndexData()) :
          reinterpret_cast<const BinIdxType*>(column_matrix.GetIndexData());
          // reinterpret_cast<const BinIdxType*>(column_matrix.GetIndexSecondData());
        size_t chunck_size = common::GetBlockSize(depth_size, nthreads);
        size_t thread_size = chunck_size;
        size_t begin = thread_size * tid;
        size_t end = std::min(begin + thread_size, depth_size);
        begin += depth_begin;
        end += depth_begin;
        opt_partition_builder_.template CommonPartition<BinIdxType,
                                                        is_loss_guided,
                                                        !any_missing,
                                                        any_cat>(tid, begin, end, numa,
                                                                 node_ids_.data(),
                                                                 split_conditions_,
                                                                 split_ind_,
                                                                 smalest_nodes_mask_ptr,
                                                                 curr_level_nodes_,
                                                                 column_matrix,
                                                                 split_nodes, pred, depth);
      }
    }

    if (depth != max_depth || loss_guide) {
      opt_partition_builder_.UpdateRowBuffer(*complete_trees_depth_wise_,
                                            p_tree, gmat, n_features, depth,
                                            node_ids_, is_loss_guided);
      opt_partition_builder_.UpdateThreadsWork(*complete_trees_depth_wise_, gmat,
                                              n_features, depth, is_loss_guided,
                                              is_left_small, check_is_left_small);
    }
  }

  std::vector<uint16_t> &GetNodeAssignments() { return node_ids_; }

  auto const &GetThreadTasks(const size_t tid) const {
    return opt_partition_builder_.GetSlices(tid);
  }

  auto const &GetOptPartition() const {
    return opt_partition_builder_;
  }

  RowPartitioner() = default;
  explicit RowPartitioner(GenericParameter const *ctx,
                                GHistIndexMatrix const &gmat,
                                common::ColumnMatrix const &column_matrix,
                                const RegTree* p_tree_local,
                                size_t max_depth,
                                bool is_loss_guide) {
    is_loss_guided = is_loss_guide;
    const size_t block_size = common::GetBlockSize(gmat.row_ptr.size() - 1, ctx->Threads());

    if (is_loss_guided) {
      opt_partition_builder_.ResizeRowsBuffer(gmat.row_ptr.size() - 1);
      uint32_t* row_set_collection_vec_p = opt_partition_builder_.GetRowsBuffer();
      #pragma omp parallel num_threads(ctx->Threads())
      {
        const size_t tid = omp_get_thread_num();
        const size_t ibegin = tid * block_size;
        const size_t iend = std::min(static_cast<size_t>(ibegin + block_size),
            static_cast<size_t>(gmat.row_ptr.size() - 1));
        for (size_t i = ibegin; i < iend; ++i) {
          row_set_collection_vec_p[i] = i;
        }
      }
    }
    switch (column_matrix.GetTypeSize()) {
      case common::kUint8BinsTypeSize:
        opt_partition_builder_.Init<uint8_t>(gmat, column_matrix, p_tree_local,
                                                ctx->Threads(), max_depth,
                                                is_loss_guide);
        break;
      case common::kUint16BinsTypeSize:
        opt_partition_builder_.Init<uint16_t>(gmat, column_matrix, p_tree_local,
                                                ctx->Threads(), max_depth,
                                                is_loss_guide);
        break;
      case common::kUint32BinsTypeSize:
        opt_partition_builder_.Init<uint32_t>(gmat, column_matrix, p_tree_local,
                                                ctx->Threads(), max_depth,
                                                is_loss_guide);
        break;
      default:
        CHECK(false);  // no default behavior
    }
    opt_partition_builder_.SetSlice(0, 0, gmat.row_ptr.size() - 1);
    node_ids_.resize(gmat.row_ptr.size() - 1, 0);
  }

  void Reset(GenericParameter const *ctx,
                     GHistIndexMatrix const &gmat,
                     common::ColumnMatrix const & column_matrix,
                     const RegTree* p_tree_local,
                     size_t max_depth,
                     bool is_loss_guide) {
    const size_t block_size = common::GetBlockSize(gmat.row_ptr.size() - 1, ctx->Threads());

    if (is_loss_guide) {
      opt_partition_builder_.ResizeRowsBuffer(gmat.row_ptr.size() - 1);
      uint32_t* row_set_collection_vec_p = opt_partition_builder_.GetRowsBuffer();
      #pragma omp parallel num_threads(ctx->Threads())
      {
        const size_t tid = omp_get_thread_num();
        const size_t ibegin = tid * block_size;
        const size_t iend = std::min(static_cast<size_t>(ibegin + block_size),
            static_cast<size_t>(gmat.row_ptr.size() - 1));
        for (size_t i = ibegin; i < iend; ++i) {
          row_set_collection_vec_p[i] = i;
        }
      }
    }
    switch (column_matrix.GetTypeSize()) {
      case common::kUint8BinsTypeSize:
        opt_partition_builder_.Init<uint8_t>(gmat, column_matrix, p_tree_local,
                                                ctx->Threads(), max_depth,
                                                is_loss_guide);
        break;
      case common::kUint16BinsTypeSize:
        opt_partition_builder_.Init<uint16_t>(gmat, column_matrix, p_tree_local,
                                                ctx->Threads(), max_depth,
                                                is_loss_guide);
        break;
      case common::kUint32BinsTypeSize:
        opt_partition_builder_.Init<uint32_t>(gmat, column_matrix, p_tree_local,
                                                ctx->Threads(), max_depth,
                                                is_loss_guide);
        break;
      default:
        CHECK(false);  // no default behavior
    }
    opt_partition_builder_.SetSlice(0, 0, gmat.row_ptr.size() - 1);
    node_ids_.resize(gmat.row_ptr.size() - 1, 0);
  }
};
}  // namespace tree
}  // namespace xgboost
#endif  // XGBOOST_TREE_UPDATER_APPROX_H_
