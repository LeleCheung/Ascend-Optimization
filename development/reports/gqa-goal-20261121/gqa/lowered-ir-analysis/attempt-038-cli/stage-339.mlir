#map = affine_map<()[s0, s1] -> (s0 + s1)>
#map1 = affine_map<()[s0, s1, s2] -> (s0 + s1 + s2 * 8192)>
#map2 = affine_map<()[s0] -> (s0 * 64)>
#map3 = affine_map<()[s0] -> (s0 * 128)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 20 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 20 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 40 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>, hacc.hivmc_compatible_print = false, hacc.hivmc_version = #hacc.hivmc_version<"0.2.0">, hivm.module_core_type = #hivm.module_core_type<MIX>, memref.memref_as_ptr} {
  func.func @_gqa_fwd_kernel_infer_workspace_shape_function() -> index attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_workspace_shape_function>} {
    %c262144 = arith.constant 262144 : index
    return %c262144 : index
  }
  func.func @_gqa_fwd_kernel_infer_task_type_function() -> i8 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_task_type_function>} {
    %c32_i8 = arith.constant 32 : i8
    return %c32_i8 : i8
  }
  func.func @_gqa_fwd_kernel_mix_aic(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, hivm.storage_aligned, mix_mode = "mix", parallel_mode = "simd"} {
    %c163840_i64 = arith.constant 163840 : i64
    %c131072_i64 = arith.constant 131072 : i64
    %c65536_i64 = arith.constant 65536 : i64
    %c0_i64 = arith.constant 0 : i64
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
    %c196608 = arith.constant 196608 : index
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
      hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 5
      scf.for %arg10 = %13 to %16 step %c1_i32  : i32 {
        %17 = arith.divsi %arg10, %c8_i32 : i32
        %18 = arith.remsi %arg10, %c8_i32 : i32
        %19 = arith.muli %17, %c262144_i32 : i32
        %20 = arith.index_cast %19 : i32 to index
        %21 = arith.muli %18, %c128_i32 : i32
        %22 = arith.index_cast %21 : i32 to index
        %23 = affine.apply #map()[%20, %22]
        %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%23], sizes: [256, 128], strides: [1024, 1] : memref<?xbf16, #hivm.address_space<gm>> to memref<256x128xbf16, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>>
        %24 = hivm.hir.pointer_cast(%c0_i64) : memref<8x16x16x16xbf16, #hivm.address_space<cbuf>>
        %cast = memref.cast %24 : memref<8x16x16x16xbf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>
        hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast : memref<256x128xbf16, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast : memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>) init_out_buffer = false
        %25 = memref_ext.alloc_workspace() from %arg2 offset = [%c0] : from memref<?xi8, #hivm.address_space<gm>> to memref<256x128xf16, #hivm.address_space<gm>>
        hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 0
        %26 = hivm.hir.pointer_cast(%c65536_i64) : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>
        %cast_0 = memref.cast %26 : memref<8x16x16x16xf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xf16, #hivm.address_space<cbuf>>
        hivm.hir.nd2nz {dst_continuous} ins(%25 : memref<256x128xf16, #hivm.address_space<gm>>) outs(%cast_0 : memref<?x?x?x?xf16, #hivm.address_space<cbuf>>) init_out_buffer = false
        hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 5
        %27 = arith.muli %arg10, %c16_i32 : i32
        %28 = arith.maxsi %8, %27 : i32
        %29 = arith.subi %28, %27 : i32
        %30 = arith.addi %arg10, %c1_i32 : i32
        %31 = arith.muli %30, %c16_i32 : i32
        %32 = arith.minsi %11, %31 : i32
        %33 = arith.subi %32, %27 : i32
        %34 = arith.muli %18, %c8_i32 : i32
        %35 = arith.muli %17, %c2097152_i32 : i32
        %36 = arith.index_cast %35 : i32 to index
        scf.for %arg11 = %29 to %33 step %c1_i32  : i32 {
          %37 = arith.divsi %arg11, %c2_i32 : i32
          %38 = arith.remsi %arg11, %c2_i32 : i32
          %39 = arith.addi %34, %37 : i32
          %40 = arith.muli %38, %c128_i32 : i32
          %41 = arith.muli %39, %c128_i32 : i32
          %42 = arith.index_cast %41 : i32 to index
          %43 = arith.index_cast %40 : i32 to index
          %44 = affine.apply #map1()[%36, %42, %43]
          %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [%44], sizes: [128, 128], strides: [8192, 1] : memref<?xbf16, #hivm.address_space<gm>> to memref<128x128xbf16, strided<[8192, 1], offset: ?>, #hivm.address_space<gm>>
          %45 = hivm.hir.pointer_cast(%c131072_i64) : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>
          %cast_2 = memref.cast %45 : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>
          hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast_1 : memref<128x128xbf16, strided<[8192, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_2 : memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>) init_out_buffer = false
          %46 = hivm.hir.pointer_cast(%c0_i64) : memref<16x8x16x16xf32, #hivm.address_space<cc>>
          %cast_3 = memref.cast %46 : memref<16x8x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
          hivm.hir.mmadL1 {b_transpose, fixpipe_already_inserted = true} ins(%cast_2, %cast, %true, %c128, %c128, %c256 : memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>, memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_3 : memref<?x?x?x?xf32, #hivm.address_space<cc>>)
          %47 = memref_ext.alloc_workspace() from %arg2 offset = [%c65536] : from memref<?xi8, #hivm.address_space<gm>> to memref<128x256xf16, #hivm.address_space<gm>>
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 2
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, pre_quant = #hivm.fixpipe_pre_quant_mode<F322F16>} ins(%cast_3 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) outs(%47 : memref<128x256xf16, #hivm.address_space<gm>>)
          annotation.mark %47 : memref<128x256xf16, #hivm.address_space<gm>>
          hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 1
          %48 = memref_ext.alloc_workspace() from %arg2 offset = [%c131072] : from memref<?xi8, #hivm.address_space<gm>> to memref<128x256xf16, #hivm.address_space<gm>>
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 0
          %49 = hivm.hir.pointer_cast(%c163840_i64) : memref<16x8x16x16xf16, #hivm.address_space<cbuf>>
          %cast_4 = memref.cast %49 : memref<16x8x16x16xf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xf16, #hivm.address_space<cbuf>>
          hivm.hir.nd2nz {dst_continuous} ins(%48 : memref<128x256xf16, #hivm.address_space<gm>>) outs(%cast_4 : memref<?x?x?x?xf16, #hivm.address_space<cbuf>>) init_out_buffer = false
          hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 3
          %50 = hivm.hir.pointer_cast(%c0_i64) : memref<8x8x16x16xf32, #hivm.address_space<cc>>
          %cast_5 = memref.cast %50 : memref<8x8x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
          hivm.hir.mmadL1 {fixpipe_already_inserted = true} ins(%cast_4, %cast_0, %true, %c128, %c256, %c128 : memref<?x?x?x?xf16, #hivm.address_space<cbuf>>, memref<?x?x?x?xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_5 : memref<?x?x?x?xf32, #hivm.address_space<cc>>)
          %51 = memref_ext.alloc_workspace() from %arg2 offset = [%c196608] : from memref<?xi8, #hivm.address_space<gm>> to memref<128x128xf32, #hivm.address_space<gm>>
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 4
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%cast_5 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) outs(%51 : memref<128x128xf32, #hivm.address_space<gm>>)
          annotation.mark %51 : memref<128x128xf32, #hivm.address_space<gm>>
          hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 1
        }
      }
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 2
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 4
    }
    return
  }
  func.func @_gqa_fwd_kernel_mix_aiv(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.storage_aligned, mix_mode = "mix", parallel_mode = "simd"} {
    %cst = arith.constant -1.32223117 : f32
    %c196608 = arith.constant 196608 : index
    %c131072 = arith.constant 131072 : index
    %c65536 = arith.constant 65536 : index
    %c1_i32 = arith.constant 1 : i32
    %cst_0 = arith.constant 0.000000e+00 : f16
    %cst_1 = arith.constant 0.693147182 : f32
    %cst_2 = arith.constant 1.000000e+00 : f32
    %cst_3 = arith.constant 0.127517432 : f32
    %c512_i32 = arith.constant 512 : i32
    %c20_i32 = arith.constant 20 : i32
    %c16_i32 = arith.constant 16 : i32
    %c8_i32 = arith.constant 8 : i32
    %c262144_i32 = arith.constant 262144 : i32
    %c128_i32 = arith.constant 128 : i32
    %c2_i32 = arith.constant 2 : i32
    %c2097152_i32 = arith.constant 2097152 : i32
    %cst_4 = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = arith.muli %arg7, %arg8 : i32
    %1 = arith.muli %0, %arg9 : i32
    %2 = hivm.hir.get_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = arith.muli %arg9, %arg8 : i32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<64xf32, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%cst_4 : f32) outs(%alloc : memref<64xf32, #hivm.address_space<ub>>)
    %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<64x1xf32, #hivm.address_space<ub>>
    %collapse_shape = memref.collapse_shape %alloc_5 [[0, 1]] : memref<64x1xf32, #hivm.address_space<ub>> into memref<64xf32, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%cst_4 : f32) outs(%collapse_shape : memref<64xf32, #hivm.address_space<ub>>)
    %5 = hivm.hir.get_sub_block_idx -> i64
    %6 = arith.index_cast %5 : i64 to index
    %7 = affine.apply #map2()[%6]
    %8 = affine.apply #map3()[%6]
    hivm.hir.set_ffts_base_addr %arg0
    hivm.hir.set_mask_norm
    annotation.mark %1 {logical_block_num} : i32
    %9 = arith.divsi %3, %4 : i32
    %10 = arith.remsi %9, %arg7 : i32
    %11 = arith.muli %10, %c512_i32 : i32
    %12 = arith.divsi %11, %c20_i32 : i32
    %13 = arith.addi %10, %c1_i32 : i32
    %14 = arith.muli %13, %c512_i32 : i32
    %15 = arith.divsi %14, %c20_i32 : i32
    %16 = arith.cmpi sgt, %15, %12 : i32
    scf.if %16 {
      %17 = arith.divsi %12, %c16_i32 : i32
      %18 = arith.subi %15, %c1_i32 : i32
      %19 = arith.divsi %18, %c16_i32 : i32
      %20 = arith.addi %19, %c1_i32 : i32
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 2
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 4
      scf.for %arg10 = %17 to %20 step %c1_i32  : i32 {
        %21 = arith.divsi %arg10, %c8_i32 : i32
        %22 = arith.remsi %arg10, %c8_i32 : i32
        %23 = arith.muli %21, %c262144_i32 : i32
        %24 = arith.index_cast %23 : i32 to index
        %25 = arith.muli %22, %c128_i32 : i32
        %26 = arith.index_cast %25 : i32 to index
        %27 = affine.apply #map()[%24, %26]
        %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%27], sizes: [256, 128], strides: [1024, 1] : memref<?xbf16, #hivm.address_space<gm>> to memref<256x128xbf16, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>>
        %alloc_6 = memref.alloc() : memref<128x128xbf16, #hivm.address_space<ub>>
        %subview = memref.subview %reinterpret_cast[%8, 0] [128, 128] [1, 1] : memref<256x128xbf16, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>> to memref<128x128xbf16, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.load ins(%subview : memref<128x128xbf16, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_6 : memref<128x128xbf16, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
        %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<128x128xf32, #hivm.address_space<ub>>
        %collapse_shape_8 = memref.collapse_shape %alloc_6 [[0, 1]] : memref<128x128xbf16, #hivm.address_space<ub>> into memref<16384xbf16, #hivm.address_space<ub>>
        %collapse_shape_9 = memref.collapse_shape %alloc_7 [[0, 1]] : memref<128x128xf32, #hivm.address_space<ub>> into memref<16384xf32, #hivm.address_space<ub>>
        hivm.hir.vcast ins(%collapse_shape_8 : memref<16384xbf16, #hivm.address_space<ub>>) outs(%collapse_shape_9 : memref<16384xf32, #hivm.address_space<ub>>)
        %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<128x128xf16, #hivm.address_space<ub>>
        %collapse_shape_11 = memref.collapse_shape %alloc_10 [[0, 1]] : memref<128x128xf16, #hivm.address_space<ub>> into memref<16384xf16, #hivm.address_space<ub>>
        hivm.hir.vcast ins(%collapse_shape_9 : memref<16384xf32, #hivm.address_space<ub>>) outs(%collapse_shape_11 : memref<16384xf16, #hivm.address_space<ub>>)
        %28 = memref_ext.alloc_workspace() from %arg2 offset = [%c0] : from memref<?xi8, #hivm.address_space<gm>> to memref<256x128xf16, #hivm.address_space<gm>>
        %subview_12 = memref.subview %28[%8, 0] [128, 128] [1, 1] : memref<256x128xf16, #hivm.address_space<gm>> to memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 5
        %collapse_shape_13 = memref.collapse_shape %subview_12 [[0, 1]] : memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> into memref<16384xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.store ins(%collapse_shape_11 : memref<16384xf16, #hivm.address_space<ub>>) outs(%collapse_shape_13 : memref<16384xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>) {tiled_op}
        annotation.mark %subview_12 : memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 0
        %29 = arith.muli %arg10, %c16_i32 : i32
        %30 = arith.maxsi %12, %29 : i32
        %31 = arith.subi %30, %29 : i32
        %32 = arith.addi %arg10, %c1_i32 : i32
        %33 = arith.muli %32, %c16_i32 : i32
        %34 = arith.minsi %15, %33 : i32
        %35 = arith.subi %34, %29 : i32
        %36 = arith.muli %22, %c8_i32 : i32
        %37 = arith.muli %21, %c2097152_i32 : i32
        %38 = arith.index_cast %37 : i32 to index
        scf.for %arg11 = %31 to %35 step %c1_i32  : i32 {
          %39 = arith.divsi %arg11, %c2_i32 : i32
          %40 = arith.remsi %arg11, %c2_i32 : i32
          %41 = arith.addi %36, %39 : i32
          %42 = arith.muli %40, %c128_i32 : i32
          %43 = arith.muli %41, %c128_i32 : i32
          %44 = arith.index_cast %43 : i32 to index
          %45 = arith.index_cast %42 : i32 to index
          %46 = affine.apply #map1()[%38, %44, %45]
          %47 = memref_ext.alloc_workspace() from %arg2 offset = [%c65536] : from memref<?xi8, #hivm.address_space<gm>> to memref<128x256xf16, #hivm.address_space<gm>>
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 1
          %subview_14 = memref.subview %47[%7, 0] [64, 256] [1, 1] : memref<128x256xf16, #hivm.address_space<gm>> to memref<64x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          %alloc_15 = memref.alloc() {alignment = 64 : i64} : memref<64x256xf16, #hivm.address_space<ub>>
          %collapse_shape_16 = memref.collapse_shape %subview_14 [[0, 1]] : memref<64x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> into memref<16384xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape_17 = memref.collapse_shape %alloc_15 [[0, 1]] : memref<64x256xf16, #hivm.address_space<ub>> into memref<16384xf16, #hivm.address_space<ub>>
          hivm.hir.load ins(%collapse_shape_16 : memref<16384xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_17 : memref<16384xf16, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 2
          %alloc_18 = memref.alloc() {alignment = 64 : i64} : memref<64x256xf32, #hivm.address_space<ub>>
          %collapse_shape_19 = memref.collapse_shape %alloc_18 [[0, 1]] : memref<64x256xf32, #hivm.address_space<ub>> into memref<16384xf32, #hivm.address_space<ub>>
          hivm.hir.vcast ins(%collapse_shape_17 : memref<16384xf16, #hivm.address_space<ub>>) outs(%collapse_shape_19 : memref<16384xf32, #hivm.address_space<ub>>)
          %alloc_20 = memref.alloc() {alignment = 64 : i64} : memref<64x256xf32, #hivm.address_space<ub>>
          %collapse_shape_21 = memref.collapse_shape %alloc_20 [[0, 1]] : memref<64x256xf32, #hivm.address_space<ub>> into memref<16384xf32, #hivm.address_space<ub>>
          hivm.hir.vmul ins(%collapse_shape_19, %cst_3 : memref<16384xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_21 : memref<16384xf32, #hivm.address_space<ub>>)
          %alloc_22 = memref.alloc() {alignment = 64 : i64} : memref<64x256xf32, #hivm.address_space<ub>>
          %collapse_shape_23 = memref.collapse_shape %alloc_22 [[0, 1]] : memref<64x256xf32, #hivm.address_space<ub>> into memref<16384xf32, #hivm.address_space<ub>>
          hivm.hir.vadd ins(%collapse_shape_21, %cst : memref<16384xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_23 : memref<16384xf32, #hivm.address_space<ub>>)
          %alloc_24 = memref.alloc() {alignment = 64 : i64} : memref<64x256xf32, #hivm.address_space<ub>>
          %collapse_shape_25 = memref.collapse_shape %alloc_24 [[0, 1]] : memref<64x256xf32, #hivm.address_space<ub>> into memref<16384xf32, #hivm.address_space<ub>>
          hivm.hir.vmul ins(%collapse_shape_23, %cst_1 : memref<16384xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_25 : memref<16384xf32, #hivm.address_space<ub>>)
          %alloc_26 = memref.alloc() {alignment = 64 : i64} : memref<64x256xf32, #hivm.address_space<ub>>
          %collapse_shape_27 = memref.collapse_shape %alloc_26 [[0, 1]] : memref<64x256xf32, #hivm.address_space<ub>> into memref<16384xf32, #hivm.address_space<ub>>
          hivm.hir.vexp ins(%collapse_shape_25 : memref<16384xf32, #hivm.address_space<ub>>) outs(%collapse_shape_27 : memref<16384xf32, #hivm.address_space<ub>>)
          %alloc_28 = memref.alloc() {alignment = 64 : i64} : memref<64x256xf16, #hivm.address_space<ub>>
          %collapse_shape_29 = memref.collapse_shape %alloc_28 [[0, 1]] : memref<64x256xf16, #hivm.address_space<ub>> into memref<16384xf16, #hivm.address_space<ub>>
          hivm.hir.vcast ins(%collapse_shape_27 : memref<16384xf32, #hivm.address_space<ub>>) outs(%collapse_shape_29 : memref<16384xf16, #hivm.address_space<ub>>)
          %48 = memref_ext.alloc_workspace() from %arg2 offset = [%c131072] : from memref<?xi8, #hivm.address_space<gm>> to memref<128x256xf16, #hivm.address_space<gm>>
          %subview_30 = memref.subview %48[%7, 0] [64, 256] [1, 1] : memref<128x256xf16, #hivm.address_space<gm>> to memref<64x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 3
          %collapse_shape_31 = memref.collapse_shape %subview_30 [[0, 1]] : memref<64x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> into memref<16384xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.store ins(%collapse_shape_29 : memref<16384xf16, #hivm.address_space<ub>>) outs(%collapse_shape_31 : memref<16384xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>) {tiled_op}
          annotation.mark %subview_30 : memref<64x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 0
          %49 = memref_ext.alloc_workspace() from %arg2 offset = [%c196608] : from memref<?xi8, #hivm.address_space<gm>> to memref<128x128xf32, #hivm.address_space<gm>>
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 1
          %subview_32 = memref.subview %49[%7, 0] [64, 128] [1, 1] : memref<128x128xf32, #hivm.address_space<gm>> to memref<64x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
          %alloc_33 = memref.alloc() {alignment = 64 : i64} : memref<64x128xf32, #hivm.address_space<ub>>
          %collapse_shape_34 = memref.collapse_shape %subview_32 [[0, 1]] : memref<64x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> into memref<8192xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape_35 = memref.collapse_shape %alloc_33 [[0, 1]] : memref<64x128xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
          hivm.hir.load ins(%collapse_shape_34 : memref<8192xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_35 : memref<8192xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 4
          %alloc_36 = memref.alloc() {alignment = 64 : i64} : memref<64x1xf32, #hivm.address_space<ub>>
          %alloc_37 = memref.alloc() : memref<8192xf32, #hivm.address_space<ub>>
          hivm.hir.vreduce <sum> ins(%alloc_26 : memref<64x256xf32, #hivm.address_space<ub>>) outs(%alloc_36 : memref<64x1xf32, #hivm.address_space<ub>>) temp_buffer(%alloc_37 : memref<8192xf32, #hivm.address_space<ub>>) reduce_dims = [1]
          %collapse_shape_38 = memref.collapse_shape %alloc_36 [[0, 1]] : memref<64x1xf32, #hivm.address_space<ub>> into memref<64xf32, #hivm.address_space<ub>>
          %alloc_39 = memref.alloc() {alignment = 64 : i64} : memref<64xi1, #hivm.address_space<ub>>
          hivm.hir.vcmp ins(%collapse_shape_38, %alloc : memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>) outs(%alloc_39 : memref<64xi1, #hivm.address_space<ub>>) compare_mode = <gt>
          %alloc_40 = memref.alloc() {alignment = 64 : i64} : memref<64xf32, #hivm.address_space<ub>>
          %alloc_41 = memref.alloc() : memref<16xf32, #hivm.address_space<ub>>
          hivm.hir.vsel ins(%alloc_39, %collapse_shape_38, %cst_2 : memref<64xi1, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, f32) outs(%alloc_40 : memref<64xf32, #hivm.address_space<ub>>) temp_buffer(%alloc_41 : memref<16xf32, #hivm.address_space<ub>>)
          %expand_shape = memref.expand_shape %alloc_40 [[0, 1]] output_shape [64, 1] : memref<64xf32, #hivm.address_space<ub>> into memref<64x1xf32, #hivm.address_space<ub>>
          %alloc_42 = memref.alloc() {alignment = 64 : i64} : memref<64x128xf32, #hivm.address_space<ub>>
          %alloc_43 = memref.alloc() : memref<512xf32, #hivm.address_space<ub>>
          hivm.hir.vdiv ins(%alloc_33, %expand_shape : memref<64x128xf32, #hivm.address_space<ub>>, memref<64x1xf32, #hivm.address_space<ub>>) outs(%alloc_42 : memref<64x128xf32, #hivm.address_space<ub>>) temp_buffer(%alloc_43 : memref<512xf32, #hivm.address_space<ub>>) broadcast = [1]
          %alloc_44 = memref.alloc() {alignment = 64 : i64} : memref<64x1xi1, #hivm.address_space<ub>>
          %collapse_shape_45 = memref.collapse_shape %alloc_44 [[0, 1]] : memref<64x1xi1, #hivm.address_space<ub>> into memref<64xi1, #hivm.address_space<ub>>
          hivm.hir.vcmp ins(%collapse_shape_38, %collapse_shape : memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>) outs(%collapse_shape_45 : memref<64xi1, #hivm.address_space<ub>>) compare_mode = <gt>
          %alloc_46 = memref.alloc() {alignment = 64 : i64} : memref<64x1xf16, #hivm.address_space<ub>>
          %collapse_shape_47 = memref.collapse_shape %alloc_46 [[0, 1]] : memref<64x1xf16, #hivm.address_space<ub>> into memref<64xf16, #hivm.address_space<ub>>
          %alloc_48 = memref.alloc() : memref<48xf16, #hivm.address_space<ub>>
          hivm.hir.vcast ins(%collapse_shape_45 : memref<64xi1, #hivm.address_space<ub>>) outs(%collapse_shape_47 : memref<64xf16, #hivm.address_space<ub>>) temp_buffer(%alloc_48 : memref<48xf16, #hivm.address_space<ub>>) round_mode = <trunc>
          %alloc_49 = memref.alloc() {alignment = 64 : i64} : memref<64x128xf16, #hivm.address_space<ub>>
          %alloc_50 = memref.alloc() : memref<1024xf16, #hivm.address_space<ub>>
          hivm.hir.vbrc ins(%alloc_46 : memref<64x1xf16, #hivm.address_space<ub>>) outs(%alloc_49 : memref<64x128xf16, #hivm.address_space<ub>>) temp_buffer(%alloc_50 : memref<1024xf16, #hivm.address_space<ub>>) broadcast_dims = [1]
          %alloc_51 = memref.alloc() {alignment = 64 : i64} : memref<64x128xi1, #hivm.address_space<ub>>
          %collapse_shape_52 = memref.collapse_shape %alloc_49 [[0, 1]] : memref<64x128xf16, #hivm.address_space<ub>> into memref<8192xf16, #hivm.address_space<ub>>
          %collapse_shape_53 = memref.collapse_shape %alloc_51 [[0, 1]] : memref<64x128xi1, #hivm.address_space<ub>> into memref<8192xi1, #hivm.address_space<ub>>
          hivm.hir.vcmp ins(%collapse_shape_52, %cst_0 : memref<8192xf16, #hivm.address_space<ub>>, f16) outs(%collapse_shape_53 : memref<8192xi1, #hivm.address_space<ub>>)
          %alloc_54 = memref.alloc() {alignment = 64 : i64} : memref<64x128xi1, #hivm.address_space<ub>>
          %collapse_shape_55 = memref.collapse_shape %alloc_54 [[0, 1]] : memref<64x128xi1, #hivm.address_space<ub>> into memref<8192xi1, #hivm.address_space<ub>>
          hivm.hir.vnot ins(%collapse_shape_53 : memref<8192xi1, #hivm.address_space<ub>>) outs(%collapse_shape_55 : memref<8192xi1, #hivm.address_space<ub>>)
          %alloc_56 = memref.alloc() {alignment = 64 : i64} : memref<64x128xf32, #hivm.address_space<ub>>
          %collapse_shape_57 = memref.collapse_shape %alloc_42 [[0, 1]] : memref<64x128xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
          %collapse_shape_58 = memref.collapse_shape %alloc_56 [[0, 1]] : memref<64x128xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
          %alloc_59 = memref.alloc() : memref<16xf32, #hivm.address_space<ub>>
          hivm.hir.vsel ins(%collapse_shape_55, %collapse_shape_57, %cst_4 : memref<8192xi1, #hivm.address_space<ub>>, memref<8192xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_58 : memref<8192xf32, #hivm.address_space<ub>>) temp_buffer(%alloc_59 : memref<16xf32, #hivm.address_space<ub>>)
          %reinterpret_cast_60 = memref.reinterpret_cast %arg6 to offset: [%46], sizes: [128, 128], strides: [8192, 1] : memref<?xbf16, #hivm.address_space<gm>> to memref<128x128xbf16, strided<[8192, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_61 = memref.subview %reinterpret_cast_60[%7, 0] [64, 128] [1, 1] {to_be_bubbled_slice} : memref<128x128xbf16, strided<[8192, 1], offset: ?>, #hivm.address_space<gm>> to memref<64x128xbf16, strided<[8192, 1], offset: ?>, #hivm.address_space<gm>>
          %alloc_62 = memref.alloc() {alignment = 64 : i64} : memref<64x128xbf16, #hivm.address_space<ub>>
          %collapse_shape_63 = memref.collapse_shape %alloc_62 [[0, 1]] : memref<64x128xbf16, #hivm.address_space<ub>> into memref<8192xbf16, #hivm.address_space<ub>>
          hivm.hir.vcast ins(%collapse_shape_58 : memref<8192xf32, #hivm.address_space<ub>>) outs(%collapse_shape_63 : memref<8192xbf16, #hivm.address_space<ub>>)
          hivm.hir.store ins(%alloc_62 : memref<64x128xbf16, #hivm.address_space<ub>>) outs(%subview_61 : memref<64x128xbf16, strided<[8192, 1], offset: ?>, #hivm.address_space<gm>>) {tiled_op}
        }
      }
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 3
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 5
    }
    return
  }
}


