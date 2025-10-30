#include "openvino/openvino.hpp"
#include "openvino/opsets/opset13.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/pass/serialize.hpp"

#include <openvino/runtime/intel_npu/remote_properties.hpp>
#include <openvino/runtime/properties.hpp>

#include <iostream>

template<typename T, typename presentationType, ov::element::Type_t ovElemType>
void stridedEltwiseTest(ov::Core& core) {
    const ov::Shape inputShapeSlice({1, 4, 6});
    const ov::element::Type_t elemType = ovElemType;

    auto remoteCtx = core.get_default_context("NPU");

    auto param1 = std::make_shared<ov::op::v0::Parameter>(elemType, inputShapeSlice);
    auto param2 = std::make_shared<ov::op::v0::Parameter>(elemType, inputShapeSlice);
    auto multiply = std::make_shared<ov::op::v1::Multiply>(param1, param2);
    const auto results = ov::ResultVector{std::make_shared<ov::opset1::Result>(multiply->output(0))};
    auto multiplyModel = std::make_shared<ov::Model>(results, ov::ParameterVector{param1, param2}, "EltwiseMultiply");

    std::vector<int> inputs_with_strides{0, 1};
    std::vector<int> outputs_with_strides{0};
    ov::AnyMap compilation_params { ov::inputs_with_dynamic_strides(inputs_with_strides),
                                    ov::outputs_with_dynamic_strides(outputs_with_strides) };

    ov::CompiledModel compiled_model = core.compile_model(multiplyModel, "NPU", compilation_params);
    ov::InferRequest infer_request = compiled_model.create_infer_request();

    auto outputTensor = infer_request.get_output_tensor(0);
    auto outData = outputTensor.data<presentationType>();

    auto runU8EltwiseModelOnSlice = [&](ov::Tensor& in1, ov::Tensor& in2, char* tileName) -> void {
        infer_request.set_input_tensor(0, in1);
        infer_request.set_input_tensor(1, in2);
        infer_request.infer();
        size_t dataIdx = 0;
        std::cout << tileName << std::endl;
        dataIdx = 0;
        for (size_t idx = 0; idx < 4; idx++) {
            for (size_t idx2 = 0; idx2 < 6; idx2++) {
                std::cout << static_cast<int>(outData[dataIdx]) << " ";
                dataIdx++;
            }
        std::cout << "\n";
        }
    };

{
    T sanityData[] = {1, 2, 3, 4, 5, 6,
                        2, 2, 3, 4, 5, 6,
                        3, 2, 3, 4, 5, 6,
                        4, 2, 3, 4, 5, 6};

    ov::Tensor sanity_input1(elemType, inputShapeSlice, sanityData);

    runU8EltwiseModelOnSlice(sanity_input1, sanity_input1, "sanity");
}

{
    T data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                3, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                4, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                5, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                6, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                7, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};


    ov::AnyMap main_params = {{ov::intel_npu::mem_type.name(), ov::intel_npu::MemType::L0_INTERNAL_BUF},
                         {ov::intel_npu::tensor_type.name(), {ov::intel_npu::TensorType::INPUT}},
                         {ov::intel_npu::mem_handle.name(), reinterpret_cast<void*>(data)}};
    const ov::Shape inputShape({1, 8, 12});
    auto zeroMainRemoteTensor = remoteCtx.create_tensor(elemType, inputShape, main_params);

    ov::RemoteTensor input1_roi1(zeroMainRemoteTensor, {0, 0, 0}, {1, 4, 6});
    ov::RemoteTensor input2_roi1(zeroMainRemoteTensor, {0, 0, 0}, {1, 4, 6});
    ov::RemoteTensor input1_roi2(zeroMainRemoteTensor, {0, 4, 0}, {1, 8, 6});
    ov::RemoteTensor input2_roi2(zeroMainRemoteTensor, {0, 4, 0}, {1, 8, 6});
    ov::RemoteTensor input1_roi3(zeroMainRemoteTensor, {0, 0, 6}, {1, 4, 12});
    ov::RemoteTensor input2_roi3(zeroMainRemoteTensor, {0, 0, 6}, {1, 4, 12});
    ov::RemoteTensor input1_roi4(zeroMainRemoteTensor, {0, 4, 6}, {1, 8, 12});
    ov::RemoteTensor input2_roi4(zeroMainRemoteTensor, {0, 4, 6}, {1, 8, 12});

    runU8EltwiseModelOnSlice(input1_roi1, input2_roi1, "HW tile1");
    runU8EltwiseModelOnSlice(input1_roi2, input2_roi2, "HW tile2");
    runU8EltwiseModelOnSlice(input1_roi3, input2_roi3, "HW tile3");
    runU8EltwiseModelOnSlice(input1_roi4, input2_roi4, "HW tile4");
}

 {
    T data[] = {1, 2, 3, 4, 5, 6,
                2, 2, 3, 4, 5, 6,
                3, 2, 3, 4, 5, 6,
                4, 2, 3, 4, 5, 6,
                5, 2, 3, 4, 5, 6,
                6, 2, 3, 4, 5, 6,
                7, 2, 3, 4, 5, 6,
                8, 2, 3, 4, 5, 6};

    const ov::Shape inputShape({1, 8, 6});

    ov::AnyMap main_params_h_tiling = {{ov::intel_npu::mem_type.name(), ov::intel_npu::MemType::L0_INTERNAL_BUF},
                         {ov::intel_npu::tensor_type.name(), {ov::intel_npu::TensorType::INPUT}},
                         {ov::intel_npu::mem_handle.name(), reinterpret_cast<void*>(data)}};

    auto zeroMainRemoteTEnsorForWTiling = remoteCtx.create_tensor(elemType, inputShape, main_params_h_tiling);

    std::cout << "tile only over H\n";
    ov::RemoteTensor input1_roi1(zeroMainRemoteTEnsorForWTiling, {0, 0, 0}, {1, 4, 6});
    ov::RemoteTensor input2_roi1(zeroMainRemoteTEnsorForWTiling, {0, 0, 0}, {1, 4, 6});
    ov::RemoteTensor input1_roi2(zeroMainRemoteTEnsorForWTiling, {0, 4, 0}, {1, 8, 6});
    ov::RemoteTensor input2_roi2(zeroMainRemoteTEnsorForWTiling, {0, 4, 0}, {1, 8, 6});

    runU8EltwiseModelOnSlice(input1_roi1, input2_roi1, "H tile1");
    runU8EltwiseModelOnSlice(input1_roi2, input2_roi2, "H tile2");
 }


 {
    T data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                3, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                4, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

    const ov::Shape inputShape({1, 4, 12});

    ov::AnyMap main_params_w_tiling = {{ov::intel_npu::mem_type.name(), ov::intel_npu::MemType::L0_INTERNAL_BUF},
                         {ov::intel_npu::tensor_type.name(), {ov::intel_npu::TensorType::INPUT}},
                         {ov::intel_npu::mem_handle.name(), reinterpret_cast<void*>(data)}};

    auto zeroMainRemoteTEnsorForWTiling = remoteCtx.create_tensor(elemType, inputShape, main_params_w_tiling);

    std::cout << "tile only over W\n";
    ov::RemoteTensor input1_roi1(zeroMainRemoteTEnsorForWTiling, {0, 0, 0}, {1, 4, 6});
    ov::RemoteTensor input2_roi1(zeroMainRemoteTEnsorForWTiling, {0, 0, 0}, {1, 4, 6});
    ov::RemoteTensor input1_roi2(zeroMainRemoteTEnsorForWTiling, {0, 0, 6}, {1, 4, 12});
    ov::RemoteTensor input2_roi2(zeroMainRemoteTEnsorForWTiling, {0, 0, 6}, {1, 4, 12});

    runU8EltwiseModelOnSlice(input1_roi1, input2_roi1, "W tile1");
    runU8EltwiseModelOnSlice(input1_roi2, input2_roi2, "W tile2");
 }

 {
    std::cout << "Split over C\n";

    T data[] = {1, 2, 3, 4, 5, 6,
                2, 2, 3, 4, 5, 6,
                3, 2, 3, 4, 5, 6,
                4, 2, 3, 4, 5, 6,
                7, 8, 9, 10, 11, 12,
                8, 9, 10, 11, 12, 13,
                9, 10, 11, 12, 13, 14,
                10, 11, 12, 13, 14, 15};

    const ov::Shape inputShape({2, 4, 6});

    ov::AnyMap main_params_c_tiling = {{ov::intel_npu::mem_type.name(), ov::intel_npu::MemType::L0_INTERNAL_BUF},
                         {ov::intel_npu::tensor_type.name(), {ov::intel_npu::TensorType::INPUT}},
                         {ov::intel_npu::mem_handle.name(), reinterpret_cast<void*>(data)}};

    auto zeroMainRemoteTEnsorForCTiling = remoteCtx.create_tensor(elemType, inputShape, main_params_c_tiling);

    ov::RemoteTensor input1_roi1(zeroMainRemoteTEnsorForCTiling, {0, 0, 0}, {1, 4, 6});
    ov::RemoteTensor input2_roi1(zeroMainRemoteTEnsorForCTiling, {0, 0, 0}, {1, 4, 6});
    ov::RemoteTensor input1_roi2(zeroMainRemoteTEnsorForCTiling, {1, 0, 0}, {2, 4, 6});
    ov::RemoteTensor input2_roi2(zeroMainRemoteTEnsorForCTiling, {1, 0, 0}, {2, 4, 6});

    runU8EltwiseModelOnSlice(input1_roi1, input2_roi1, "C tile1");
    runU8EltwiseModelOnSlice(input1_roi2, input2_roi2, "C tile2");
 }

  {
    std::cout << "Split over CH\n";

    T data[] = {1, 2, 3, 4, 5, 6,
                      2, 2, 3, 4, 5, 6,
                      3, 2, 3, 4, 5, 6,
                      4, 2, 3, 4, 5, 6,
                      5, 2, 3, 4, 5, 6,
                      6, 2, 3, 4, 5, 6,
                      7, 2, 3, 4, 5, 6,
                      8, 2, 3, 4, 5, 6,
                      7, 8, 9, 10, 11, 12,
                      8, 9, 10, 11, 12, 13,
                      9, 10, 11, 12, 13, 14,
                      10, 11, 12, 13, 14, 15,
                      11, 12, 13, 14, 15, 10,
                      12, 13, 14, 15, 10, 11,
                      13, 14, 15, 10, 11, 12,
                      14, 15, 10, 11, 12, 13};

    const ov::Shape inputShape({2, 8, 6});

    ov::AnyMap main_params_ch_tiling = {{ov::intel_npu::mem_type.name(), ov::intel_npu::MemType::L0_INTERNAL_BUF},
                         {ov::intel_npu::tensor_type.name(), {ov::intel_npu::TensorType::INPUT}},
                         {ov::intel_npu::mem_handle.name(), reinterpret_cast<void*>(data)}};

    auto zeroMainRemoteTEnsorForCHTiling = remoteCtx.create_tensor(elemType, inputShape, main_params_ch_tiling);

    ov::RemoteTensor input1_roi1(zeroMainRemoteTEnsorForCHTiling, {0, 0, 0}, {1, 4, 6});
    ov::RemoteTensor input2_roi1(zeroMainRemoteTEnsorForCHTiling, {0, 0, 0}, {1, 4, 6});
    ov::RemoteTensor input1_roi2(zeroMainRemoteTEnsorForCHTiling, {0, 4, 0}, {1, 8, 6});
    ov::RemoteTensor input2_roi2(zeroMainRemoteTEnsorForCHTiling, {0, 4, 0}, {1, 8, 6});
    ov::RemoteTensor input1_roi3(zeroMainRemoteTEnsorForCHTiling, {1, 0, 0}, {2, 4, 6});
    ov::RemoteTensor input2_roi3(zeroMainRemoteTEnsorForCHTiling, {1, 0, 0}, {2, 4, 6});
    ov::RemoteTensor input1_roi4(zeroMainRemoteTEnsorForCHTiling, {1, 4, 0}, {2, 8, 6});
    ov::RemoteTensor input2_roi4(zeroMainRemoteTEnsorForCHTiling, {1, 4, 0}, {2, 8, 6});

    runU8EltwiseModelOnSlice(input1_roi1, input2_roi1, "CH tile1");
    runU8EltwiseModelOnSlice(input1_roi2, input2_roi2, "CH tile2");
    runU8EltwiseModelOnSlice(input1_roi3, input2_roi3, "CH tile3");
    runU8EltwiseModelOnSlice(input1_roi4, input2_roi4, "CH tile4");
 }

 {
    std::cout << "Split over CHW\n";

    T data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                      2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                      3, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                      4, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                      5, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                      6, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                      7, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                      8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                      7, 8, 9, 10, 11, 12, 13, 14, 15, 1, 2, 3,
                      8, 9, 10, 11, 12, 13, 14, 15, 1, 2, 3, 4,
                      9, 10, 11, 12, 13, 14, 15, 1, 2, 3, 4, 5,
                      10, 11, 12, 13, 14, 15, 1, 2, 3, 4, 5, 6,
                      11, 12, 13, 14, 15, 1, 2, 3, 4, 5, 6, 7,
                      12, 13, 14, 15, 1, 2, 3, 4, 5, 6, 7, 8,
                      13, 14, 15, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                      14, 15, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    const ov::Shape inputShape({2, 8, 12});

    ov::AnyMap main_params_chw_tiling = {{ov::intel_npu::mem_type.name(), ov::intel_npu::MemType::L0_INTERNAL_BUF},
                         {ov::intel_npu::tensor_type.name(), {ov::intel_npu::TensorType::INPUT}},
                         {ov::intel_npu::mem_handle.name(), reinterpret_cast<void*>(data)}};

    auto zeroMainRemoteTEnsorForCHWTiling = remoteCtx.create_tensor(elemType, inputShape, main_params_chw_tiling);

    ov::RemoteTensor input1_roi1(zeroMainRemoteTEnsorForCHWTiling, {0, 0, 0}, {1, 4, 6});
    ov::RemoteTensor input2_roi1(zeroMainRemoteTEnsorForCHWTiling, {0, 0, 0}, {1, 4, 6});
    ov::RemoteTensor input1_roi2(zeroMainRemoteTEnsorForCHWTiling, {0, 0, 6}, {1, 4, 12});
    ov::RemoteTensor input2_roi2(zeroMainRemoteTEnsorForCHWTiling, {0, 0, 6}, {1, 4, 12});
    ov::RemoteTensor input1_roi3(zeroMainRemoteTEnsorForCHWTiling, {0, 4, 0}, {1, 8, 6});
    ov::RemoteTensor input2_roi3(zeroMainRemoteTEnsorForCHWTiling, {0, 4, 0}, {1, 8, 6});
    ov::RemoteTensor input1_roi4(zeroMainRemoteTEnsorForCHWTiling, {0, 4, 6}, {1, 8, 12});
    ov::RemoteTensor input2_roi4(zeroMainRemoteTEnsorForCHWTiling, {0, 4, 6}, {1, 8, 12});
    ov::RemoteTensor input1_roi5(zeroMainRemoteTEnsorForCHWTiling, {1, 0, 0}, {2, 4, 6});
    ov::RemoteTensor input2_roi5(zeroMainRemoteTEnsorForCHWTiling, {1, 0, 0}, {2, 4, 6});
    ov::RemoteTensor input1_roi6(zeroMainRemoteTEnsorForCHWTiling, {1, 0, 6}, {2, 4, 12});
    ov::RemoteTensor input2_roi6(zeroMainRemoteTEnsorForCHWTiling, {1, 0, 6}, {2, 4, 12});
    ov::RemoteTensor input1_roi7(zeroMainRemoteTEnsorForCHWTiling, {1, 4, 0}, {2, 8, 6});
    ov::RemoteTensor input2_roi7(zeroMainRemoteTEnsorForCHWTiling, {1, 4, 0}, {2, 8, 6});
    ov::RemoteTensor input1_roi8(zeroMainRemoteTEnsorForCHWTiling, {1, 4, 6}, {2, 8, 12});
    ov::RemoteTensor input2_roi8(zeroMainRemoteTEnsorForCHWTiling, {1, 4, 6}, {2, 8, 12});

    runU8EltwiseModelOnSlice(input1_roi1, input2_roi1, "CHW tile1");
    runU8EltwiseModelOnSlice(input1_roi2, input2_roi2, "CHW tile2");
    runU8EltwiseModelOnSlice(input1_roi3, input2_roi3, "CHW tile3");
    runU8EltwiseModelOnSlice(input1_roi4, input2_roi4, "CHW tile4");
    runU8EltwiseModelOnSlice(input1_roi5, input2_roi5, "CHW tile1");
    runU8EltwiseModelOnSlice(input1_roi6, input2_roi6, "CHW tile2");
    runU8EltwiseModelOnSlice(input1_roi7, input2_roi7, "CHW tile3");
    runU8EltwiseModelOnSlice(input1_roi8, input2_roi8, "CHW tile4");
 }

  {
    std::cout << "Split over CW\n";

    T data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                      2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                      3, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                      4, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                      5, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                      6, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                      7, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                      8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

    const ov::Shape inputShape({2, 4, 12});

    ov::AnyMap main_params_cw_tiling = {{ov::intel_npu::mem_type.name(), ov::intel_npu::MemType::L0_INTERNAL_BUF},
                         {ov::intel_npu::tensor_type.name(), {ov::intel_npu::TensorType::INPUT}},
                         {ov::intel_npu::mem_handle.name(), reinterpret_cast<void*>(data)}};

    auto zeroMainRemoteTEnsorForCWTiling = remoteCtx.create_tensor(elemType, inputShape, main_params_cw_tiling);

    ov::RemoteTensor input1_roi1(zeroMainRemoteTEnsorForCWTiling, {0, 0, 0}, {1, 4, 6});
    ov::RemoteTensor input2_roi1(zeroMainRemoteTEnsorForCWTiling, {0, 0, 0}, {1, 4, 6});
    ov::RemoteTensor input1_roi2(zeroMainRemoteTEnsorForCWTiling, {0, 0, 6}, {1, 4, 12});
    ov::RemoteTensor input2_roi2(zeroMainRemoteTEnsorForCWTiling, {0, 0, 6}, {1, 4, 12});
    ov::RemoteTensor input1_roi3(zeroMainRemoteTEnsorForCWTiling, {1, 0, 0}, {2, 4, 6});
    ov::RemoteTensor input2_roi3(zeroMainRemoteTEnsorForCWTiling, {1, 0, 0}, {2, 4, 6});
    ov::RemoteTensor input1_roi4(zeroMainRemoteTEnsorForCWTiling, {1, 0, 6}, {2, 4, 12});
    ov::RemoteTensor input2_roi4(zeroMainRemoteTEnsorForCWTiling, {1, 0, 6}, {2, 4, 12});
       
    runU8EltwiseModelOnSlice(input1_roi1, input2_roi1, "CW tile1");
    runU8EltwiseModelOnSlice(input1_roi2, input2_roi2, "CW tile2");
    runU8EltwiseModelOnSlice(input1_roi3, input2_roi3, "CW tile3");
    runU8EltwiseModelOnSlice(input1_roi4, input2_roi4, "CW tile4");
 }

    auto runU8EltwiseModelOnSliceWithOutputSlice = [&](ov::Tensor& in1, ov::Tensor& in2, ov::Tensor& out) -> void {
        infer_request.set_input_tensor(0, in1);
        infer_request.set_input_tensor(1, in2);
        infer_request.set_output_tensor(0, out);
        infer_request.infer();
    };

{
    T data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                3, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                4, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                5, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                6, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                7, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};


    ov::AnyMap main_params = {{ov::intel_npu::mem_type.name(), ov::intel_npu::MemType::L0_INTERNAL_BUF},
                         {ov::intel_npu::tensor_type.name(), {ov::intel_npu::TensorType::INPUT}},
                         {ov::intel_npu::mem_handle.name(), reinterpret_cast<void*>(data)}};

    ov::AnyMap main_output_params = {{ov::intel_npu::mem_type.name(), ov::intel_npu::MemType::L0_INTERNAL_BUF},
                         {ov::intel_npu::tensor_type.name(), {ov::intel_npu::TensorType::OUTPUT}},
                         {ov::intel_npu::mem_handle.name(), reinterpret_cast<void*>(data)}};

    const ov::Shape inputShape({1, 8, 12});
    auto zeroMainRemoteTensor = remoteCtx.create_tensor(elemType, inputShape, main_params);
    auto zeroMainOutputTensor = remoteCtx.create_tensor(elemType, inputShape, main_params);

    ov::RemoteTensor input1_roi1(zeroMainRemoteTensor, {0, 0, 0}, {1, 4, 6});
    ov::RemoteTensor input2_roi1(zeroMainRemoteTensor, {0, 0, 0}, {1, 4, 6});
    ov::RemoteTensor input1_roi2(zeroMainRemoteTensor, {0, 4, 0}, {1, 8, 6});
    ov::RemoteTensor input2_roi2(zeroMainRemoteTensor, {0, 4, 0}, {1, 8, 6});
    ov::RemoteTensor input1_roi3(zeroMainRemoteTensor, {0, 0, 6}, {1, 4, 12});
    ov::RemoteTensor input2_roi3(zeroMainRemoteTensor, {0, 0, 6}, {1, 4, 12});
    ov::RemoteTensor input1_roi4(zeroMainRemoteTensor, {0, 4, 6}, {1, 8, 12});
    ov::RemoteTensor input2_roi4(zeroMainRemoteTensor, {0, 4, 6}, {1, 8, 12});

    ov::RemoteTensor output_roi1(zeroMainOutputTensor, {0, 0, 0}, {1, 4, 6});
    ov::RemoteTensor output_roi2(zeroMainOutputTensor, {0, 4, 0}, {1, 8, 6});
    ov::RemoteTensor output_roi3(zeroMainOutputTensor, {0, 0, 6}, {1, 4, 12});
    ov::RemoteTensor output_roi4(zeroMainOutputTensor, {0, 4, 6}, {1, 8, 12});

    runU8EltwiseModelOnSliceWithOutputSlice(input1_roi1, input2_roi1, output_roi1);
    runU8EltwiseModelOnSliceWithOutputSlice(input1_roi2, input2_roi2, output_roi2);
    runU8EltwiseModelOnSliceWithOutputSlice(input1_roi3, input2_roi3, output_roi3);
    runU8EltwiseModelOnSliceWithOutputSlice(input1_roi4, input2_roi4, output_roi4);

    ov::Tensor output(elemType, inputShape, data);
    std::cerr << "copying data\n";
    zeroMainOutputTensor.copy_to(output);
    std::cerr << "getting data\n";
    auto outData = output.data<presentationType>();
    std::cerr << "displaying data\n";
    for (size_t idx = 0; idx < 8; idx++) {
        for (size_t w_idx = 0; w_idx < 12; w_idx++) {
            std::cout << static_cast<int>(outData[(idx * 12) + w_idx]) << " ";
        }
        std::cout << std::endl;
    }

}

}

void maxPoolTest(ov::Core& core) {
    std::cout << "MaxPool test\n";

    auto remoteCtx = core.get_default_context("NPU");

    const ov::Shape maxPoolInputShape({1, 16, 108, 1280});
    const ov::Shape maxPoolSliceInputShape({1, 16, 36, 1280});
    const ov::element::Type_t maxPoolElemType = ov::element::Type_t::f32;

    float* maxPoolData = new float[2211840];

    for (int idx = 0; idx < 2211840; idx++) {
        maxPoolData[idx] = static_cast<float>(idx % 256);
    }

    ov::AnyMap max_pool_params = {{ov::intel_npu::mem_type.name(), ov::intel_npu::MemType::L0_INTERNAL_BUF},
                         {ov::intel_npu::tensor_type.name(), {ov::intel_npu::TensorType::INPUT}},
                         {ov::intel_npu::mem_handle.name(), reinterpret_cast<void*>(maxPoolData)}};

    auto zeroMaxPoolMainTensor = remoteCtx.create_tensor(maxPoolElemType, maxPoolInputShape, max_pool_params);

    ov::RemoteTensor max_pool_roi1(zeroMaxPoolMainTensor, {0, 0, 0, 0}, {1, 16, 36, 1280});
    ov::RemoteTensor max_pool_roi2(zeroMaxPoolMainTensor, {0, 0, 36, 0}, {1, 16, 72, 1280});
    ov::RemoteTensor max_pool_roi3(zeroMaxPoolMainTensor, {0, 0, 72, 0}, {1, 16, 108, 1280});

    auto max_pool_param = std::make_shared<ov::op::v0::Parameter>(maxPoolElemType, maxPoolSliceInputShape);

    auto max_pool = std::make_shared<ov::op::v1::MaxPool>(max_pool_param, ov::Strides{2, 2}, 
                                                            ov::Shape{0, 0}, ov::Shape{0, 0}, ov::Shape{4, 4});
    max_pool->get_output_tensor(0).set_names({"MaxPool_Results"});

    const auto max_pool_results = ov::ResultVector{std::make_shared<ov::opset1::Result>(max_pool->output(0))};
    auto max_pool_model = std::make_shared<ov::Model>(max_pool_results, ov::ParameterVector{max_pool_param}, "MaxPool");

    ov::serialize(max_pool_model, "maxpool.xml", "maxpool.bin");

    ov::CompiledModel compiled_max_pool = core.compile_model(max_pool_model, "NPU");
    ov::InferRequest max_pool_infer_request = compiled_max_pool.create_infer_request();

    max_pool_infer_request.set_input_tensor(0, max_pool_roi1);
    max_pool_infer_request.infer();

    auto outputMaxPoolTensor = max_pool_infer_request.get_output_tensor(0);
    auto outMaxPoolData = outputMaxPoolTensor.data<float>();

    std::cout << "printing max pool output 1\n";
    size_t dataIdx = 0;
    for (size_t idx = 0; idx < 16; idx++) {
        for (size_t idx2 = 0; idx2 < 36; idx2++) {
            for (size_t idx3 = 0; idx3 < 1280; idx3++) {
                std::cout << static_cast<int>(outMaxPoolData[dataIdx]) << " ";
                dataIdx++;
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }

    max_pool_infer_request.set_input_tensor(0, max_pool_roi2);
    max_pool_infer_request.infer();

    std::cout << "printing max pool output 2\n";
    dataIdx = 0;
    for (size_t idx = 0; idx < 16; idx++) {
        for (size_t idx2 = 0; idx2 < 36; idx2++) {
            for (size_t idx3 = 0; idx3 < 1280; idx3++) {
                std::cout << static_cast<int>(outMaxPoolData[dataIdx]) << " ";
                dataIdx++;
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }

    max_pool_infer_request.set_input_tensor(0, max_pool_roi3);
    max_pool_infer_request.infer();

    std::cout << "printing max pool output 3\n";
    dataIdx = 0;
    for (size_t idx = 0; idx < 16; idx++) {
        for (size_t idx2 = 0; idx2 < 36; idx2++) {
            for (size_t idx3 = 0; idx3 < 1280; idx3++) {
                std::cout << static_cast<int>(outMaxPoolData[dataIdx]) << " ";
                dataIdx++;
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
}

int main(int, char**) {
try {
    std::cout << "starting strided sample\n";

    ov::Core core;

    std::cout << "u8 test\n";
    stridedEltwiseTest<uint8_t, uint8_t, ov::element::Type_t::u8>(core);

    std::cout << "u32 test\n";
    stridedEltwiseTest<uint32_t, uint32_t, ov::element::Type_t::u32>(core);

    std::cout << "u16 test\n";
    stridedEltwiseTest<uint16_t, uint16_t, ov::element::Type_t::u16>(core);

    std::cout << "f32 test\n";
    stridedEltwiseTest<float, float, ov::element::Type_t::f32>(core);

    // u64 seems to be broken even on sanity.
    //std::cout << "u64 test\n";
    //stridedEltwiseTest<uint64_t, uint64_t, ov::element::Type_t::u64>(core);

    //maxPoolTest(core);

} catch (const std::exception& ex) {
    std::cout << ex.what() << std::endl;
    return EXIT_FAILURE;
}
    return EXIT_SUCCESS;
}