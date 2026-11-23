// -----// IR Dump After HIVMMapForallToBlocks (hivm-map-forall-to-blocks) ('func.func' operation: @_gqa_fwd_kernel_mix_aiv) //----- //
#map = affine_map<()[s0, s1] -> (s0 + s1)>
#map1 = affine_map<()[s0, s1, s2] -> (s0 + s1 + s2 * 8192)>
#map2 = affine_map<()[s0] -> (s0 * 64)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 20 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 20 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 40 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>, hacc.hivmc_compatible_print = false, hacc.hivmc_version = #hacc.hivmc_version<"0.2.0">, hivm.module_core_type = #hivm.module_core_type<MIX>, memref.memref_as_ptr} {
  func.func @_gqa_fwd_kernel_infer_workspace_shape_function() -> index attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_workspace_shape_function>} {
    %c163840 = arith.constant 163840 : index
    return %c163840 : index
  }
  func.func @_gqa_fwd_kernel_infer_task_type_function() -> i8 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_task_type_function>} {
    %c32_i8 = arith.constant 32 : i8
    return %c32_i8 : i8
  }
  func.func @_gqa_fwd_kernel_mix_aic(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, mix_mode = "mix", parallel_mode = "simd"} {
    %c2097152_i32 = arith.constant 2097152 : i32
    %c2_i32 = arith.constant 2 : i32
    %c128_i32 = arith.constant 128 : i32
    %c262144_i32 = arith.constant 262144 : i32
    %c8_i32 = arith.constant 8 : i32
    %c16_i32 = arith.constant 16 : i32
    %c20_i32 = arith.constant 20 : i32
    %c512_i32 = arith.constant 512 : i32
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    %c0 = arith.constant 0 : index
    %c65536 = arith.constant 65536 : index
    %c131072 = arith.constant 131072 : index
    hivm.hir.set_ffts_base_addr %arg0
    hivm.hir.set_mask_norm
    %0 = arith.muli %arg7, %arg8 : i32
    %1 = arith.muli %0, %arg9 : i32
    annotation.mark %1 {logical_block_num} : i32
    %2 = hivm.hir.get_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = arith.muli %arg9, %arg8 : i32
    %5 = arith.divsi %3, %4 : i32
    %6 = arith.remsi %5, %arg7 : i32
    %7 = arith.muli %6, %c512_i32 : i32
    %8 = arith.divsi %7, %c20_i32 : i32
    %9 = arith.addi %6, %c1_i32 : i32
    %10 = arith.muli %9, %c512_i32 : i32
    %11 = arith.divsi %10, %c20_i32 : i32
    %12 = arith.cmpi sgt, %11, %8 : i32
    scf.if %12 {
      %13 = arith.divsi %8, %c16_i32 : i32
      %14 = arith.subi %11, %c1_i32 : i32
      %15 = arith.divsi %14, %c16_i32 : i32
      %16 = arith.addi %15, %c1_i32 : i32
      hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 3
      scf.for %arg10 = %13 to %16 step %c1_i32  : i32 {
        %17 = arith.divsi %arg10, %c8_i32 : i32
        %18 = arith.remsi %arg10, %c8_i32 : i32
        %19 = arith.muli %17, %c262144_i32 : i32
        %20 = arith.index_cast %19 : i32 to index
        %21 = arith.muli %18, %c128_i32 : i32
        %22 = arith.index_cast %21 : i32 to index
        %23 = affine.apply #map()[%20, %22]
        %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%23], sizes: [256, 128], strides: [1024, 1] : memref<?xbf16> to memref<256x128xbf16, strided<[1024, 1], offset: ?>>
        %alloc = memref.alloc() : memref<256x128xbf16>
        hivm.hir.load ins(%reinterpret_cast : memref<256x128xbf16, strided<[1024, 1], offset: ?>>) outs(%alloc : memref<256x128xbf16>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
        %reinterpret_cast_0 = memref.reinterpret_cast %arg5 to offset: [%23], sizes: [256, 128], strides: [1024, 1] : memref<?xbf16> to memref<256x128xbf16, strided<[1024, 1], offset: ?>>
        %alloc_1 = memref.alloc() : memref<256x128xbf16>
        hivm.hir.load ins(%reinterpret_cast_0 : memref<256x128xbf16, strided<[1024, 1], offset: ?>>) outs(%alloc_1 : memref<256x128xbf16>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
        %24 = arith.muli %arg10, %c16_i32 : i32
        %25 = arith.maxsi %8, %24 : i32
        %26 = arith.subi %25, %24 : i32
        %27 = arith.addi %arg10, %c1_i32 : i32
        %28 = arith.muli %27, %c16_i32 : i32
        %29 = arith.minsi %11, %28 : i32
        %30 = arith.subi %29, %24 : i32
        %31 = arith.muli %18, %c8_i32 : i32
        %32 = arith.muli %17, %c2097152_i32 : i32
        %33 = arith.index_cast %32 : i32 to index
        scf.for %arg11 = %26 to %30 step %c1_i32  : i32 {
          %34 = arith.divsi %arg11, %c2_i32 : i32
          %35 = arith.remsi %arg11, %c2_i32 : i32
          %36 = arith.addi %31, %34 : i32
          %37 = arith.muli %35, %c128_i32 : i32
          %38 = arith.muli %36, %c128_i32 : i32
          %39 = arith.index_cast %38 : i32 to index
          %40 = arith.index_cast %37 : i32 to index
          %41 = affine.apply #map1()[%33, %39, %40]
          %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [%41], sizes: [128, 128], strides: [8192, 1] : memref<?xbf16> to memref<128x128xbf16, strided<[8192, 1], offset: ?>>
          %alloc_3 = memref.alloc() : memref<128x128xbf16>
          hivm.hir.load ins(%reinterpret_cast_2 : memref<128x128xbf16, strided<[8192, 1], offset: ?>>) outs(%alloc_3 : memref<128x128xbf16>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
          %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<128x256xf32>
          hivm.hir.mmadL1 {b_transpose, fixpipe_already_inserted = true} ins(%alloc_3, %alloc, %true, %c128, %c128, %c256 : memref<128x128xbf16>, memref<256x128xbf16>, i1, index, index, index) outs(%alloc_4 : memref<128x256xf32>)
          %42 = memref_ext.alloc_workspace() from %arg2 offset = [%c0] : from memref<?xi8> to memref<128x256xf16>
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 2
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, pre_quant = #hivm.fixpipe_pre_quant_mode<F322F16>} ins(%alloc_4 : memref<128x256xf32>) outs(%42 : memref<128x256xf16>)
          annotation.mark %42 : memref<128x256xf16>
          hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 0
          %43 = memref_ext.alloc_workspace() from %arg2 offset = [%c65536] : from memref<?xi8> to memref<128x256xbf16>
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 1
          %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<128x256xbf16>
          hivm.hir.load ins(%43 : memref<128x256xbf16>) outs(%alloc_5 : memref<128x256xbf16>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
          hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 3
          %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<128x128xf32>
          hivm.hir.mmadL1 {fixpipe_already_inserted = true} ins(%alloc_5, %alloc_1, %true, %c128, %c256, %c128 : memref<128x256xbf16>, memref<256x128xbf16>, i1, index, index, index) outs(%alloc_6 : memref<128x128xf32>)
          %44 = memref_ext.alloc_workspace() from %arg2 offset = [%c131072] : from memref<?xi8> to memref<128x128xf16>
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 4
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, pre_quant = #hivm.fixpipe_pre_quant_mode<F322F16>} ins(%alloc_6 : memref<128x128xf32>) outs(%44 : memref<128x128xf16>)
          annotation.mark %44 : memref<128x128xf16>
          hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 0
        }
      }
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 2
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 4
    }
    return
  }
  func.func @_gqa_fwd_kernel_mix_aiv(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, mix_mode = "mix", parallel_mode = "simd"} {
    %c131072 = arith.constant 131072 : index
    %c65536 = arith.constant 65536 : index
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant -1.000000e+00 : f32
    %cst_0 = arith.constant 2.000000e+00 : f32
    %cst_1 = arith.constant 0.0883883461 : f32
    %c512_i32 = arith.constant 512 : i32
    %c20_i32 = arith.constant 20 : i32
    %c16_i32 = arith.constant 16 : i32
    %c8_i32 = arith.constant 8 : i32
    %c128_i32 = arith.constant 128 : i32
    %c2_i32 = arith.constant 2 : i32
    %c2097152_i32 = arith.constant 2097152 : i32
    %c0 = arith.constant 0 : index
    %0 = arith.muli %arg7, %arg8 : i32
    %1 = arith.muli %0, %arg9 : i32
    %2 = hivm.hir.get_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = arith.muli %arg9, %arg8 : i32
    %5 = hivm.hir.get_sub_block_idx -> i64
    %6 = arith.index_cast %5 : i64 to index
    %7 = affine.apply #map2()[%6]
    hivm.hir.set_ffts_base_addr %arg0
    hivm.hir.set_mask_norm
    annotation.mark %1 {logical_block_num} : i32
    %8 = arith.divsi %3, %4 : i32
    %9 = arith.remsi %8, %arg7 : i32
    %10 = arith.muli %9, %c512_i32 : i32
    %11 = arith.divsi %10, %c20_i32 : i32
    %12 = arith.addi %9, %c1_i32 : i32
    %13 = arith.muli %12, %c512_i32 : i32
    %14 = arith.divsi %13, %c20_i32 : i32
    %15 = arith.cmpi sgt, %14, %11 : i32
    scf.if %15 {
      %16 = arith.divsi %11, %c16_i32 : i32
      %17 = arith.subi %14, %c1_i32 : i32
      %18 = arith.divsi %17, %c16_i32 : i32
      %19 = arith.addi %18, %c1_i32 : i32
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 2
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 4
      scf.for %arg10 = %16 to %19 step %c1_i32  : i32 {
        %20 = arith.divsi %arg10, %c8_i32 : i32
        %21 = arith.remsi %arg10, %c8_i32 : i32
        %22 = arith.muli %arg10, %c16_i32 : i32
        %23 = arith.maxsi %11, %22 : i32
        %24 = arith.subi %23, %22 : i32
        %25 = arith.addi %arg10, %c1_i32 : i32
        %26 = arith.muli %25, %c16_i32 : i32
        %27 = arith.minsi %14, %26 : i32
        %28 = arith.subi %27, %22 : i32
        %29 = arith.muli %21, %c8_i32 : i32
        %30 = arith.muli %20, %c2097152_i32 : i32
        %31 = arith.index_cast %30 : i32 to index
        scf.for %arg11 = %24 to %28 step %c1_i32  : i32 {
          %32 = arith.divsi %arg11, %c2_i32 : i32
          %33 = arith.remsi %arg11, %c2_i32 : i32
          %34 = arith.addi %29, %32 : i32
          %35 = arith.muli %33, %c128_i32 : i32
          %36 = arith.muli %34, %c128_i32 : i32
          %37 = arith.index_cast %36 : i32 to index
          %38 = arith.index_cast %35 : i32 to index
          %39 = affine.apply #map1()[%31, %37, %38]
          %40 = memref_ext.alloc_workspace() from %arg2 offset = [%c0] : from memref<?xi8> to memref<128x256xf16>
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 0
          %subview = memref.subview %40[%7, 0] [64, 256] [1, 1] : memref<128x256xf16> to memref<64x256xf16, strided<[256, 1], offset: ?>>
          %alloc = memref.alloc() {alignment = 64 : i64} : memref<64x256xf16>
          hivm.hir.load ins(%subview : memref<64x256xf16, strided<[256, 1], offset: ?>>) outs(%alloc : memref<64x256xf16>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 2
          %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<64x256xf32>
          hivm.hir.vcast ins(%alloc : memref<64x256xf16>) outs(%alloc_2 : memref<64x256xf32>)
          %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<64x256xf32>
          hivm.hir.vmul ins(%alloc_2, %cst_1 : memref<64x256xf32>, f32) outs(%alloc_3 : memref<64x256xf32>)
          %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<64x256xf32>
          hivm.hir.vexp ins(%alloc_3 : memref<64x256xf32>) outs(%alloc_4 : memref<64x256xf32>)
          %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<64x256xbf16>
          hivm.hir.vcast ins(%alloc_4 : memref<64x256xf32>) outs(%alloc_5 : memref<64x256xbf16>)
          %41 = memref_ext.alloc_workspace() from %arg2 offset = [%c65536] : from memref<?xi8> to memref<128x256xbf16>
          %subview_6 = memref.subview %41[%7, 0] [64, 256] [1, 1] : memref<128x256xbf16> to memref<64x256xbf16, strided<[256, 1], offset: ?>>
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 3
          hivm.hir.store ins(%alloc_5 : memref<64x256xbf16>) outs(%subview_6 : memref<64x256xbf16, strided<[256, 1], offset: ?>>) {tiled_op}
          annotation.mark %subview_6 : memref<64x256xbf16, strided<[256, 1], offset: ?>>
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 1
          %42 = memref_ext.alloc_workspace() from %arg2 offset = [%c131072] : from memref<?xi8> to memref<128x128xf16>
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 0
          %subview_7 = memref.subview %42[%7, 0] [64, 128] [1, 1] : memref<128x128xf16> to memref<64x128xf16, strided<[128, 1], offset: ?>>
          %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<64x128xf16>
          hivm.hir.load ins(%subview_7 : memref<64x128xf16, strided<[128, 1], offset: ?>>) outs(%alloc_8 : memref<64x128xf16>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 4
          %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<64x128xf32>
          hivm.hir.vcast ins(%alloc_8 : memref<64x128xf16>) outs(%alloc_9 : memref<64x128xf32>)
          %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<64x1xf32>
          hivm.hir.vreduce <sum> ins(%alloc_4 : memref<64x256xf32>) outs(%alloc_10 : memref<64x1xf32>) reduce_dims = [1]
          %collapse_shape = memref.collapse_shape %alloc_10 [[0, 1]] : memref<64x1xf32> into memref<64xf32>
          %alloc_11 = memref.alloc() {alignment = 64 : i64} : memref<64xf32>
          hivm.hir.vrec ins(%collapse_shape : memref<64xf32>) outs(%alloc_11 : memref<64xf32>)
          %alloc_12 = memref.alloc() {alignment = 64 : i64} : memref<64xf32>
          hivm.hir.vmul ins(%collapse_shape, %alloc_11 : memref<64xf32>, memref<64xf32>) outs(%alloc_12 : memref<64xf32>)
          %alloc_13 = memref.alloc() {alignment = 64 : i64} : memref<64xf32>
          hivm.hir.vmul ins(%alloc_12, %cst : memref<64xf32>, f32) outs(%alloc_13 : memref<64xf32>)
          %alloc_14 = memref.alloc() {alignment = 64 : i64} : memref<64xf32>
          hivm.hir.vadd ins(%alloc_13, %cst_0 : memref<64xf32>, f32) outs(%alloc_14 : memref<64xf32>)
          %alloc_15 = memref.alloc() {alignment = 64 : i64} : memref<64xf32>
          hivm.hir.vmul ins(%alloc_11, %alloc_14 : memref<64xf32>, memref<64xf32>) outs(%alloc_15 : memref<64xf32>)
          %expand_shape = memref.expand_shape %alloc_15 [[0, 1]] output_shape [64, 1] : memref<64xf32> into memref<64x1xf32>
          %alloc_16 = memref.alloc() {alignment = 64 : i64} : memref<64x128xf32>
          hivm.hir.vmul ins(%alloc_9, %expand_shape : memref<64x128xf32>, memref<64x1xf32>) outs(%alloc_16 : memref<64x128xf32>) broadcast = [1]
          %reinterpret_cast = memref.reinterpret_cast %arg6 to offset: [%39], sizes: [128, 128], strides: [8192, 1] : memref<?xbf16> to memref<128x128xbf16, strided<[8192, 1], offset: ?>>
          %subview_17 = memref.subview %reinterpret_cast[%7, 0] [64, 128] [1, 1] {to_be_bubbled_slice} : memref<128x128xbf16, strided<[8192, 1], offset: ?>> to memref<64x128xbf16, strided<[8192, 1], offset: ?>>
          %alloc_18 = memref.alloc() {alignment = 64 : i64} : memref<64x128xbf16>
          hivm.hir.vcast ins(%alloc_16 : memref<64x128xf32>) outs(%alloc_18 : memref<64x128xbf16>)
          hivm.hir.store ins(%alloc_18 : memref<64x128xbf16>) outs(%subview_17 : memref<64x128xbf16, strided<[8192, 1], offset: ?>>) {tiled_op}
        }
      }
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 3
    }
    return
  }
}


