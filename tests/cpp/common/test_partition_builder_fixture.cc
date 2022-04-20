/*!
 * Copyright 2022 by XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <utility>

#include "../../../src/common/opt_partition_builder.h"
#include "../helpers.h"

namespace xgboost {
namespace common {

class OptPartitionBuilderTestFixture :
                public testing::TestWithParam<std::tuple<size_t, size_t, int32_t>> {
using DMatrixP = std::shared_ptr<DMatrix>;

 public:
  OptPartitionBuilderTestFixture() {
    auto param = GetParam();
    row_count_ = std::get<0>(param);
    column_count_ = std::get<1>(param);
    max_bin_count_ = std::get<2>(param);
  }

  DMatrixP GenDMatrix(std::uint64_t seed) {
    return RandomDataGenerator(row_count_, column_count_, 0).Seed(seed).GenerateDMatrix();
  }

  const GHistIndexMatrix& GetIndexMatrix(const DMatrixP& dmatrix_ptr) const {
    const auto& res = *(dmatrix_ptr->GetBatches<GHistIndexMatrix>(
              BatchParam{GenericParameter::kCpuId, max_bin_count_}).begin());
    return res;
  }

  ColumnMatrix GetColumnMatrix(const GHistIndexMatrix& gmat) const {
    ColumnMatrix column_matrix;
    column_matrix.Init(gmat, 0, 1);

    return std::move(column_matrix);
  }

  void CommonPartitionCheck(const GHistIndexMatrix& gmat,
                            const ColumnMatrix& column_matrix,
                            const RegTree& tree) {
    switch (column_matrix.GetTypeSize()) {
    case kUint8BinsTypeSize:
          CommonPartitionCheckImpl<typename BinTypeMap<kUint8BinsTypeSize>::type>(gmat,
                                                                    column_matrix, tree);
          break;
        case kUint16BinsTypeSize:
          CommonPartitionCheckImpl<typename BinTypeMap<kUint16BinsTypeSize>::type>(gmat,
                                                                    column_matrix, tree);
          break;
        default:
          CommonPartitionCheckImpl<typename BinTypeMap<kUint32BinsTypeSize>::type>(gmat,
                                                                    column_matrix, tree);
      }
  }

  std::tuple<size_t, size_t> GetRef(const GHistIndexMatrix& gmat, size_t split_bin_id) {
    size_t left_cnt = 0;
    size_t right_cnt = 0;

    const size_t bin_id_min = gmat.cut.Ptrs()[0];
    const size_t bin_id_max = gmat.cut.Ptrs()[1];

    for (size_t rid = 0; rid < row_count_; ++rid) {
    for (size_t offset = gmat.row_ptr[rid]; offset < gmat.row_ptr[rid + 1]; ++offset) {
      const size_t bin_id = gmat.index[offset];
        if (bin_id >= bin_id_min && bin_id < bin_id_max) {
        if (bin_id <= split_bin_id) {
        left_cnt++;
        } else {
        right_cnt++;
        }
      }
    }
    }
    return std::make_tuple(left_cnt, right_cnt);
  }

 private:
  template <typename BinType>
  void CommonPartitionCheckImpl(const GHistIndexMatrix& gmat,
                                const ColumnMatrix& column_matrix,
                                const RegTree& tree) {
    OptPartitionBuilder opt_partition_builder;

    opt_partition_builder.template Init<BinType>(gmat, column_matrix, &tree,
    1, 3, false);
    const BinType* data = reinterpret_cast<const BinType*>(column_matrix.GetIndexData());
    const size_t fid = 0;
    const size_t split = 0;
    std::unordered_map<uint32_t, int32_t> split_conditions;
    std::unordered_map<uint32_t, uint64_t> split_ind;
    std::vector<uint16_t> node_ids(row_count_, 0);
    std::unordered_map<uint32_t, bool> smalest_nodes_mask;
    smalest_nodes_mask[1] = true;
    std::unordered_map<uint32_t, uint16_t> nodes;  // (1, 0);
    std::vector<uint32_t> split_nodes(1, 0);
    auto pred = [&](auto ridx, auto bin_id, auto nid, auto split_cond) {
    return false;
    };

    opt_partition_builder.template CommonPartition<
    BinType, false, true, false>(0, 0, row_count_, data,
                node_ids.data(),
                &split_conditions,
                &split_ind,
                &smalest_nodes_mask,  // row_gpairs,
                column_matrix, split_nodes, pred, 1);
    opt_partition_builder.UpdateRowBuffer(node_ids,
                      gmat, gmat.cut.Ptrs().size() - 1,
                      0, node_ids, false);
    size_t split_bin_id = 0;
    auto [left_cnt, right_cnt] = GetRef(gmat, split_bin_id);
      ASSERT_EQ(opt_partition_builder.summ_size, left_cnt);
    ASSERT_EQ(row_count_ - opt_partition_builder.summ_size, right_cnt);
  }

 private:
  size_t row_count_;
  size_t column_count_;
  int32_t max_bin_count_;
};

TEST_P(OptPartitionBuilderTestFixture, TestCommonPartitionCheck) {
  std::uint64_t seed = 3;

  auto dmatrix_ptr = this->GenDMatrix(seed);
  const auto& bin_matrix = this->GetIndexMatrix(dmatrix_ptr);
  auto column_matrix = this->GetColumnMatrix(bin_matrix);
  RegTree tree;
  tree.ExpandNode(0, 0, 0, true, 0, 0, 0, 0, 0, 0, 0);

  this->CommonPartitionCheck(bin_matrix, column_matrix, tree);
}

INSTANTIATE_TEST_CASE_P(
    OptPartitionBuilderValueParametrized,
    OptPartitionBuilderTestFixture,
    testing::Combine(testing::Values(8),
                     testing::Values(16),
                     testing::Values(4, 512)));

}  // namespace common
}  // namespace xgboost
