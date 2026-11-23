#map = affine_map<()[s0, s1] -> (s0 + s1)>
#map1 = affine_map<()[s0, s1, s2, s3, s4, s5, s6] -> ((s0 - s1) floordiv s3 + ((s4 - s5) floordiv s6) * ((-s1 + s2 + s3 - 1) floordiv s3))>
#map2 = affine_map<()[s0, s1, s2] -> (s0 + s1 + s2 * 8192)>
#map3 = affine_map<()[s0] -> (s0 * 64)>
#map4 = affine_map<()[s0] -> (s0 * 128)>
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
    %c2_i64 = arith.constant 2 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %c0_i64_0 = arith.constant 0 : i64
    %c163840_i64 = arith.constant 163840 : i64
    %c131072_i64 = arith.constant 131072 : i64
    %c65536_i64 = arith.constant 65536 : i64
    %c0_i64_1 = arith.constant 0 : i64
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
      hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
      hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID2>]
      hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
      scf.for %arg10 = %13 to %16 step %c1_i32  : i32 {
        %17 = arith.divsi %arg10, %c8_i32 : i32
        %18 = arith.remsi %arg10, %c8_i32 : i32
        %19 = arith.muli %17, %c262144_i32 : i32
        %20 = arith.index_cast %19 : i32 to index
        %21 = arith.muli %18, %c128_i32 : i32
        %22 = arith.index_cast %21 : i32 to index
        %23 = affine.apply #map()[%20, %22]
        %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%23], sizes: [256, 128], strides: [1024, 1] : memref<?xbf16, #hivm.address_space<gm>> to memref<256x128xbf16, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>>
        %24 = hivm.hir.pointer_cast(%c0_i64_1) : memref<8x16x16x16xbf16, #hivm.address_space<cbuf>>
        %cast = memref.cast %24 : memref<8x16x16x16xbf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>
        hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.pipe_barrier[<PIPE_MTE2>]
        hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast : memref<256x128xbf16, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast : memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>) init_out_buffer = false
        %25 = memref_ext.alloc_workspace() from %arg2 offset = [%c0] : from memref<?xi8, #hivm.address_space<gm>> to memref<256x128xf16, #hivm.address_space<gm>>
        hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 0
        %26 = hivm.hir.pointer_cast(%c65536_i64) : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>
        %cast_2 = memref.cast %26 : memref<8x16x16x16xf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xf16, #hivm.address_space<cbuf>>
        hivm.hir.nd2nz {dst_continuous} ins(%25 : memref<256x128xf16, #hivm.address_space<gm>>) outs(%cast_2 : memref<?x?x?x?xf16, #hivm.address_space<cbuf>>) init_out_buffer = false
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
          %37 = arith.index_cast %arg11 : i32 to index
          %38 = arith.index_cast %29 : i32 to index
          %39 = arith.index_cast %33 : i32 to index
          %40 = arith.index_cast %c1_i32 : i32 to index
          %41 = arith.index_cast %arg10 : i32 to index
          %42 = arith.index_cast %13 : i32 to index
          %43 = arith.index_cast %16 : i32 to index
          %44 = arith.index_cast %c1_i32 : i32 to index
          %45 = affine.apply #map1()[%37, %38, %39, %40, %41, %42, %44]
          %46 = arith.index_cast %45 : index to i64
          %47 = arith.divsi %arg11, %c2_i32 : i32
          %48 = arith.remsi %arg11, %c2_i32 : i32
          %49 = arith.addi %34, %47 : i32
          %50 = arith.muli %48, %c128_i32 : i32
          %51 = arith.muli %49, %c128_i32 : i32
          %52 = arith.index_cast %51 : i32 to index
          %53 = arith.index_cast %50 : i32 to index
          %54 = affine.apply #map2()[%36, %52, %53]
          %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [%54], sizes: [128, 128], strides: [8192, 1] : memref<?xbf16, #hivm.address_space<gm>> to memref<128x128xbf16, strided<[8192, 1], offset: ?>, #hivm.address_space<gm>>
          %55 = hivm.hir.pointer_cast(%c131072_i64) : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>
          %cast_4 = memref.cast %55 : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>
          hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
          hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast_3 : memref<128x128xbf16, strided<[8192, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_4 : memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>) init_out_buffer = false
          hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
          %56 = hivm.hir.pointer_cast(%c0_i64_1) : memref<16x8x16x16xf32, #hivm.address_space<cc>>
          %cast_5 = memref.cast %56 : memref<16x8x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
          hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
          %c-1_i64 = arith.constant -1 : i64
          hivm.hir.mmadL1 {b_transpose, fixpipe_already_inserted = true} ins(%cast_4, %cast, %true, %c128, %c128, %c256 : memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>, memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_5 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) sync_related_args(%c0_i64_0, %c-1_i64, %c1_i64, %c-1_i64, %46, %c-1_i64, %c-1_i64 : i64, i64, i64, i64, i64, i64, i64)
          hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
          %57 = memref_ext.alloc_workspace() from %arg2 offset = [%c65536] : from memref<?xi8, #hivm.address_space<gm>> to memref<128x256xf16, #hivm.address_space<gm>>
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 2
          hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, pre_quant = #hivm.fixpipe_pre_quant_mode<F322F16>} ins(%cast_5 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) outs(%57 : memref<128x256xf16, #hivm.address_space<gm>>)
          hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
          annotation.mark %57 : memref<128x256xf16, #hivm.address_space<gm>>
          hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 1
          %58 = memref_ext.alloc_workspace() from %arg2 offset = [%c131072] : from memref<?xi8, #hivm.address_space<gm>> to memref<128x256xf16, #hivm.address_space<gm>>
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 0
          %59 = hivm.hir.pointer_cast(%c163840_i64) : memref<16x8x16x16xf16, #hivm.address_space<cbuf>>
          %cast_6 = memref.cast %59 : memref<16x8x16x16xf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xf16, #hivm.address_space<cbuf>>
          hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID2>]
          hivm.hir.nd2nz {dst_continuous} ins(%58 : memref<128x256xf16, #hivm.address_space<gm>>) outs(%cast_6 : memref<?x?x?x?xf16, #hivm.address_space<cbuf>>) init_out_buffer = false
          hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
          hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 3
          %60 = hivm.hir.pointer_cast(%c0_i64_1) : memref<8x8x16x16xf32, #hivm.address_space<cc>>
          %cast_7 = memref.cast %60 : memref<8x8x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
          hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
          %c-1_i64_8 = arith.constant -1 : i64
          hivm.hir.mmadL1 {fixpipe_already_inserted = true} ins(%cast_6, %cast_2, %true, %c128, %c256, %c128 : memref<?x?x?x?xf16, #hivm.address_space<cbuf>>, memref<?x?x?x?xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_7 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) sync_related_args(%c0_i64, %c-1_i64_8, %c2_i64, %c-1_i64_8, %46, %c-1_i64_8, %c-1_i64_8 : i64, i64, i64, i64, i64, i64, i64)
          hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
          %61 = memref_ext.alloc_workspace() from %arg2 offset = [%c196608] : from memref<?xi8, #hivm.address_space<gm>> to memref<128x128xf32, #hivm.address_space<gm>>
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 4
          hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%cast_7 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) outs(%61 : memref<128x128xf32, #hivm.address_space<gm>>)
          hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
          annotation.mark %61 : memref<128x128xf32, #hivm.address_space<gm>>
          hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 1
        }
        hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
      }
      hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
      hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
      hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID2>]
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 2
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 4
    }
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
  func.func @_gqa_fwd_kernel_mix_aiv(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.storage_aligned, mix_mode = "mix", parallel_mode = "simd"} {
    %c181120_i64 = arith.constant 181120 : i64
    %c164736_i64 = arith.constant 164736 : i64
    %c164896_i64 = arith.constant 164896 : i64
    %c164640_i64 = arith.constant 164640 : i64
    %c164608_i64 = arith.constant 164608 : i64
    %c164352_i64 = arith.constant 164352 : i64
    %c131072_i64 = arith.constant 131072 : i64
    %c98304_i64 = arith.constant 98304 : i64
    %c32768_i64 = arith.constant 32768 : i64
    %c0_i64 = arith.constant 0 : i64
    %c164096_i64 = arith.constant 164096 : i64
    %c163840_i64 = arith.constant 163840 : i64
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
    %5 = hivm.hir.pointer_cast(%c163840_i64) : memref<64xf32, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%cst_4 : f32) outs(%5 : memref<64xf32, #hivm.address_space<ub>>)
    %6 = hivm.hir.pointer_cast(%c164096_i64) : memref<64x1xf32, #hivm.address_space<ub>>
    %collapse_shape = memref.collapse_shape %6 [[0, 1]] : memref<64x1xf32, #hivm.address_space<ub>> into memref<64xf32, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%cst_4 : f32) outs(%collapse_shape : memref<64xf32, #hivm.address_space<ub>>)
    %7 = hivm.hir.get_sub_block_idx -> i64
    %8 = arith.index_cast %7 : i64 to index
    %9 = affine.apply #map3()[%8]
    %10 = affine.apply #map4()[%8]
    hivm.hir.set_ffts_base_addr %arg0
    hivm.hir.set_mask_norm
    annotation.mark %1 {logical_block_num} : i32
    %11 = arith.divsi %3, %4 : i32
    %12 = arith.remsi %11, %arg7 : i32
    %13 = arith.muli %12, %c512_i32 : i32
    %14 = arith.divsi %13, %c20_i32 : i32
    %15 = arith.addi %12, %c1_i32 : i32
    %16 = arith.muli %15, %c512_i32 : i32
    %17 = arith.divsi %16, %c20_i32 : i32
    %18 = arith.cmpi sgt, %17, %14 : i32
    scf.if %18 {
      %19 = arith.divsi %14, %c16_i32 : i32
      %20 = arith.subi %17, %c1_i32 : i32
      %21 = arith.divsi %20, %c16_i32 : i32
      %22 = arith.addi %21, %c1_i32 : i32
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 2
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 4
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
      scf.for %arg10 = %19 to %22 step %c1_i32  : i32 {
        %23 = arith.divsi %arg10, %c8_i32 : i32
        %24 = arith.remsi %arg10, %c8_i32 : i32
        %25 = arith.muli %23, %c262144_i32 : i32
        %26 = arith.index_cast %25 : i32 to index
        %27 = arith.muli %24, %c128_i32 : i32
        %28 = arith.index_cast %27 : i32 to index
        %29 = affine.apply #map()[%26, %28]
        %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%29], sizes: [256, 128], strides: [1024, 1] : memref<?xbf16, #hivm.address_space<gm>> to memref<256x128xbf16, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>>
        %30 = hivm.hir.pointer_cast(%c0_i64) : memref<128x128xbf16, #hivm.address_space<ub>>
        %subview = memref.subview %reinterpret_cast[%10, 0] [128, 128] [1, 1] : memref<256x128xbf16, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>> to memref<128x128xbf16, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.load ins(%subview : memref<128x128xbf16, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>>) outs(%30 : memref<128x128xbf16, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %31 = hivm.hir.pointer_cast(%c32768_i64) : memref<128x128xf32, #hivm.address_space<ub>>
        %collapse_shape_5 = memref.collapse_shape %30 [[0, 1]] : memref<128x128xbf16, #hivm.address_space<ub>> into memref<16384xbf16, #hivm.address_space<ub>>
        %collapse_shape_6 = memref.collapse_shape %31 [[0, 1]] : memref<128x128xf32, #hivm.address_space<ub>> into memref<16384xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.vcast ins(%collapse_shape_5 : memref<16384xbf16, #hivm.address_space<ub>>) outs(%collapse_shape_6 : memref<16384xf32, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        %32 = hivm.hir.pointer_cast(%c32768_i64) : memref<128x128xf16, #hivm.address_space<ub>>
        %collapse_shape_7 = memref.collapse_shape %32 [[0, 1]] : memref<128x128xf16, #hivm.address_space<ub>> into memref<16384xf16, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vcast ins(%collapse_shape_6 : memref<16384xf32, #hivm.address_space<ub>>) outs(%collapse_shape_7 : memref<16384xf16, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
        %33 = memref_ext.alloc_workspace() from %arg2 offset = [%c0] : from memref<?xi8, #hivm.address_space<gm>> to memref<256x128xf16, #hivm.address_space<gm>>
        %subview_8 = memref.subview %33[%10, 0] [128, 128] [1, 1] : memref<256x128xf16, #hivm.address_space<gm>> to memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 5
        %collapse_shape_9 = memref.collapse_shape %subview_8 [[0, 1]] : memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> into memref<16384xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
        hivm.hir.store ins(%collapse_shape_7 : memref<16384xf16, #hivm.address_space<ub>>) outs(%collapse_shape_9 : memref<16384xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>) {tiled_op}
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
        annotation.mark %subview_8 : memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 0
        %34 = arith.muli %arg10, %c16_i32 : i32
        %35 = arith.maxsi %14, %34 : i32
        %36 = arith.subi %35, %34 : i32
        %37 = arith.addi %arg10, %c1_i32 : i32
        %38 = arith.muli %37, %c16_i32 : i32
        %39 = arith.minsi %17, %38 : i32
        %40 = arith.subi %39, %34 : i32
        %41 = arith.muli %24, %c8_i32 : i32
        %42 = arith.muli %23, %c2097152_i32 : i32
        %43 = arith.index_cast %42 : i32 to index
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
        scf.for %arg11 = %36 to %40 step %c1_i32  : i32 {
          %44 = arith.divsi %arg11, %c2_i32 : i32
          %45 = arith.remsi %arg11, %c2_i32 : i32
          %46 = arith.addi %41, %44 : i32
          %47 = arith.muli %45, %c128_i32 : i32
          %48 = arith.muli %46, %c128_i32 : i32
          %49 = arith.index_cast %48 : i32 to index
          %50 = arith.index_cast %47 : i32 to index
          %51 = affine.apply #map2()[%43, %49, %50]
          %52 = memref_ext.alloc_workspace() from %arg2 offset = [%c65536] : from memref<?xi8, #hivm.address_space<gm>> to memref<128x256xf16, #hivm.address_space<gm>>
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 1
          %subview_10 = memref.subview %52[%9, 0] [64, 256] [1, 1] : memref<128x256xf16, #hivm.address_space<gm>> to memref<64x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          %53 = hivm.hir.pointer_cast(%c98304_i64) : memref<64x256xf16, #hivm.address_space<ub>>
          %collapse_shape_11 = memref.collapse_shape %subview_10 [[0, 1]] : memref<64x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> into memref<16384xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape_12 = memref.collapse_shape %53 [[0, 1]] : memref<64x256xf16, #hivm.address_space<ub>> into memref<16384xf16, #hivm.address_space<ub>>
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
          hivm.hir.load ins(%collapse_shape_11 : memref<16384xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_12 : memref<16384xf16, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
          hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 2
          %54 = hivm.hir.pointer_cast(%c32768_i64) : memref<64x256xf32, #hivm.address_space<ub>>
          %collapse_shape_13 = memref.collapse_shape %54 [[0, 1]] : memref<64x256xf32, #hivm.address_space<ub>> into memref<16384xf32, #hivm.address_space<ub>>
          hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
          hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
          hivm.hir.vcast ins(%collapse_shape_12 : memref<16384xf16, #hivm.address_space<ub>>) outs(%collapse_shape_13 : memref<16384xf32, #hivm.address_space<ub>>)
          %55 = hivm.hir.pointer_cast(%c32768_i64) : memref<64x256xf32, #hivm.address_space<ub>>
          %collapse_shape_14 = memref.collapse_shape %55 [[0, 1]] : memref<64x256xf32, #hivm.address_space<ub>> into memref<16384xf32, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          hivm.hir.vmul ins(%collapse_shape_13, %cst_3 : memref<16384xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_14 : memref<16384xf32, #hivm.address_space<ub>>)
          %56 = hivm.hir.pointer_cast(%c32768_i64) : memref<64x256xf32, #hivm.address_space<ub>>
          %collapse_shape_15 = memref.collapse_shape %56 [[0, 1]] : memref<64x256xf32, #hivm.address_space<ub>> into memref<16384xf32, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          hivm.hir.vadd ins(%collapse_shape_14, %cst : memref<16384xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_15 : memref<16384xf32, #hivm.address_space<ub>>)
          %57 = hivm.hir.pointer_cast(%c32768_i64) : memref<64x256xf32, #hivm.address_space<ub>>
          %collapse_shape_16 = memref.collapse_shape %57 [[0, 1]] : memref<64x256xf32, #hivm.address_space<ub>> into memref<16384xf32, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          hivm.hir.vmul ins(%collapse_shape_15, %cst_1 : memref<16384xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_16 : memref<16384xf32, #hivm.address_space<ub>>)
          %58 = hivm.hir.pointer_cast(%c32768_i64) : memref<64x256xf32, #hivm.address_space<ub>>
          %collapse_shape_17 = memref.collapse_shape %58 [[0, 1]] : memref<64x256xf32, #hivm.address_space<ub>> into memref<16384xf32, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          hivm.hir.vexp ins(%collapse_shape_16 : memref<16384xf32, #hivm.address_space<ub>>) outs(%collapse_shape_17 : memref<16384xf32, #hivm.address_space<ub>>)
          %59 = hivm.hir.pointer_cast(%c131072_i64) : memref<64x256xf16, #hivm.address_space<ub>>
          %collapse_shape_18 = memref.collapse_shape %59 [[0, 1]] : memref<64x256xf16, #hivm.address_space<ub>> into memref<16384xf16, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          hivm.hir.vcast ins(%collapse_shape_17 : memref<16384xf32, #hivm.address_space<ub>>) outs(%collapse_shape_18 : memref<16384xf16, #hivm.address_space<ub>>)
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
          %60 = memref_ext.alloc_workspace() from %arg2 offset = [%c131072] : from memref<?xi8, #hivm.address_space<gm>> to memref<128x256xf16, #hivm.address_space<gm>>
          %subview_19 = memref.subview %60[%9, 0] [64, 256] [1, 1] : memref<128x256xf16, #hivm.address_space<gm>> to memref<64x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 3
          %collapse_shape_20 = memref.collapse_shape %subview_19 [[0, 1]] : memref<64x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> into memref<16384xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
          hivm.hir.store ins(%collapse_shape_18 : memref<16384xf16, #hivm.address_space<ub>>) outs(%collapse_shape_20 : memref<16384xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>) {tiled_op}
          annotation.mark %subview_19 : memref<64x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 0
          %61 = memref_ext.alloc_workspace() from %arg2 offset = [%c196608] : from memref<?xi8, #hivm.address_space<gm>> to memref<128x128xf32, #hivm.address_space<gm>>
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 1
          %subview_21 = memref.subview %61[%9, 0] [64, 128] [1, 1] : memref<128x128xf32, #hivm.address_space<gm>> to memref<64x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
          %62 = hivm.hir.pointer_cast(%c0_i64) : memref<64x128xf32, #hivm.address_space<ub>>
          %collapse_shape_22 = memref.collapse_shape %subview_21 [[0, 1]] : memref<64x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> into memref<8192xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape_23 = memref.collapse_shape %62 [[0, 1]] : memref<64x128xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
          hivm.hir.load ins(%collapse_shape_22 : memref<8192xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_23 : memref<8192xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
          hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 4
          %63 = hivm.hir.pointer_cast(%c164352_i64) : memref<64x1xf32, #hivm.address_space<ub>>
          %64 = hivm.hir.pointer_cast(%c98304_i64) : memref<8192xf32, #hivm.address_space<ub>>
          hivm.hir.vreduce <sum> ins(%58 : memref<64x256xf32, #hivm.address_space<ub>>) outs(%63 : memref<64x1xf32, #hivm.address_space<ub>>) temp_buffer(%64 : memref<8192xf32, #hivm.address_space<ub>>) reduce_dims = [1]
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
          %collapse_shape_24 = memref.collapse_shape %63 [[0, 1]] : memref<64x1xf32, #hivm.address_space<ub>> into memref<64xf32, #hivm.address_space<ub>>
          %65 = hivm.hir.pointer_cast(%c164608_i64) : memref<64xi1, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          hivm.hir.vcmp ins(%collapse_shape_24, %5 : memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>) outs(%65 : memref<64xi1, #hivm.address_space<ub>>) compare_mode = <gt>
          %66 = hivm.hir.pointer_cast(%c164640_i64) : memref<64xf32, #hivm.address_space<ub>>
          %67 = hivm.hir.pointer_cast(%c164896_i64) : memref<16xf32, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          hivm.hir.vsel ins(%65, %collapse_shape_24, %cst_2 : memref<64xi1, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, f32) outs(%66 : memref<64xf32, #hivm.address_space<ub>>) temp_buffer(%67 : memref<16xf32, #hivm.address_space<ub>>)
          %expand_shape = memref.expand_shape %66 [[0, 1]] output_shape [64, 1] : memref<64xf32, #hivm.address_space<ub>> into memref<64x1xf32, #hivm.address_space<ub>>
          %68 = hivm.hir.pointer_cast(%c0_i64) : memref<64x128xf32, #hivm.address_space<ub>>
          %69 = hivm.hir.pointer_cast(%c164896_i64) : memref<512xf32, #hivm.address_space<ub>>
          hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
          hivm.hir.pipe_barrier[<PIPE_V>]
          hivm.hir.vdiv ins(%62, %expand_shape : memref<64x128xf32, #hivm.address_space<ub>>, memref<64x1xf32, #hivm.address_space<ub>>) outs(%68 : memref<64x128xf32, #hivm.address_space<ub>>) temp_buffer(%69 : memref<512xf32, #hivm.address_space<ub>>) broadcast = [1]
          %70 = hivm.hir.pointer_cast(%c164352_i64) : memref<64x1xi1, #hivm.address_space<ub>>
          %collapse_shape_25 = memref.collapse_shape %70 [[0, 1]] : memref<64x1xi1, #hivm.address_space<ub>> into memref<64xi1, #hivm.address_space<ub>>
          hivm.hir.vcmp ins(%collapse_shape_24, %collapse_shape : memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>) outs(%collapse_shape_25 : memref<64xi1, #hivm.address_space<ub>>) compare_mode = <gt>
          %71 = hivm.hir.pointer_cast(%c164608_i64) : memref<64x1xf16, #hivm.address_space<ub>>
          %collapse_shape_26 = memref.collapse_shape %71 [[0, 1]] : memref<64x1xf16, #hivm.address_space<ub>> into memref<64xf16, #hivm.address_space<ub>>
          %72 = hivm.hir.pointer_cast(%c164736_i64) : memref<48xf16, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          hivm.hir.vcast ins(%collapse_shape_25 : memref<64xi1, #hivm.address_space<ub>>) outs(%collapse_shape_26 : memref<64xf16, #hivm.address_space<ub>>) temp_buffer(%72 : memref<48xf16, #hivm.address_space<ub>>) round_mode = <trunc>
          %73 = hivm.hir.pointer_cast(%c164736_i64) : memref<64x128xf16, #hivm.address_space<ub>>
          %74 = hivm.hir.pointer_cast(%c181120_i64) : memref<1024xf16, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          hivm.hir.vbrc ins(%71 : memref<64x1xf16, #hivm.address_space<ub>>) outs(%73 : memref<64x128xf16, #hivm.address_space<ub>>) temp_buffer(%74 : memref<1024xf16, #hivm.address_space<ub>>) broadcast_dims = [1]
          %75 = hivm.hir.pointer_cast(%c164736_i64) : memref<64x128xi1, #hivm.address_space<ub>>
          %collapse_shape_27 = memref.collapse_shape %73 [[0, 1]] : memref<64x128xf16, #hivm.address_space<ub>> into memref<8192xf16, #hivm.address_space<ub>>
          %collapse_shape_28 = memref.collapse_shape %75 [[0, 1]] : memref<64x128xi1, #hivm.address_space<ub>> into memref<8192xi1, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          hivm.hir.vcmp ins(%collapse_shape_27, %cst_0 : memref<8192xf16, #hivm.address_space<ub>>, f16) outs(%collapse_shape_28 : memref<8192xi1, #hivm.address_space<ub>>)
          %76 = hivm.hir.pointer_cast(%c164736_i64) : memref<64x128xi1, #hivm.address_space<ub>>
          %collapse_shape_29 = memref.collapse_shape %76 [[0, 1]] : memref<64x128xi1, #hivm.address_space<ub>> into memref<8192xi1, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          hivm.hir.vnot ins(%collapse_shape_28 : memref<8192xi1, #hivm.address_space<ub>>) outs(%collapse_shape_29 : memref<8192xi1, #hivm.address_space<ub>>)
          %77 = hivm.hir.pointer_cast(%c32768_i64) : memref<64x128xf32, #hivm.address_space<ub>>
          %collapse_shape_30 = memref.collapse_shape %68 [[0, 1]] : memref<64x128xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
          %collapse_shape_31 = memref.collapse_shape %77 [[0, 1]] : memref<64x128xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
          %78 = hivm.hir.pointer_cast(%c164352_i64) : memref<16xf32, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          hivm.hir.vsel ins(%collapse_shape_29, %collapse_shape_30, %cst_4 : memref<8192xi1, #hivm.address_space<ub>>, memref<8192xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_31 : memref<8192xf32, #hivm.address_space<ub>>) temp_buffer(%78 : memref<16xf32, #hivm.address_space<ub>>)
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
          %reinterpret_cast_32 = memref.reinterpret_cast %arg6 to offset: [%51], sizes: [128, 128], strides: [8192, 1] : memref<?xbf16, #hivm.address_space<gm>> to memref<128x128xbf16, strided<[8192, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_33 = memref.subview %reinterpret_cast_32[%9, 0] [64, 128] [1, 1] {to_be_bubbled_slice} : memref<128x128xbf16, strided<[8192, 1], offset: ?>, #hivm.address_space<gm>> to memref<64x128xbf16, strided<[8192, 1], offset: ?>, #hivm.address_space<gm>>
          %79 = hivm.hir.pointer_cast(%c32768_i64) : memref<64x128xbf16, #hivm.address_space<ub>>
          %collapse_shape_34 = memref.collapse_shape %79 [[0, 1]] : memref<64x128xbf16, #hivm.address_space<ub>> into memref<8192xbf16, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          hivm.hir.vcast ins(%collapse_shape_31 : memref<8192xf32, #hivm.address_space<ub>>) outs(%collapse_shape_34 : memref<8192xbf16, #hivm.address_space<ub>>)
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
          hivm.hir.store ins(%79 : memref<64x128xbf16, #hivm.address_space<ub>>) outs(%subview_33 : memref<64x128xbf16, strided<[8192, 1], offset: ?>, #hivm.address_space<gm>>) {tiled_op}
          hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
        }
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      }
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 3
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 5
    }
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}


