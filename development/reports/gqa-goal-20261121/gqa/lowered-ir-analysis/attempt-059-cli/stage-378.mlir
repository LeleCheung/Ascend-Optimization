// -----// IR Dump After InsertFreeLockVarBeforeReturn (hivm-insert-free-lock-var-before-return) ('builtin.module' operation) //----- //
#map = affine_map<()[s0, s1] -> ((s0 - s1) mod 2)>
#map1 = affine_map<()[s0, s1] -> (s0 + s1)>
#map2 = affine_map<()[s0, s1, s2, s3, s4] -> ((s0 - s1 + (s3 - s4) * (-s1 + s2)) mod 2)>
#map3 = affine_map<()[s0, s1, s2, s3, s4] -> (s0 - s1 + (s3 - s4) * (-s1 + s2))>
#map4 = affine_map<()[s0, s1, s2] -> (s0 + s1 + s2 * 8192)>
#map5 = affine_map<(d0) -> (d0 * 163840)>
#map6 = affine_map<(d0) -> (d0 * 163840 + 65536)>
#map7 = affine_map<(d0) -> (d0 * 163840 + 131072)>
#map8 = affine_map<()[s0] -> (s0 * 64)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 20 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 20 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 40 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>, hacc.hivmc_compatible_print = true, hacc.hivmc_version = #hacc.hivmc_version<"0.2.0">, hivm.module_core_type = #hivm.module_core_type<MIX>, memref.memref_as_ptr} {
  func.func @_gqa_fwd_kernel_infer_workspace_shape_function() -> index attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_workspace_shape_function>} {
    %c163840 = arith.constant 163840 : index
    return %c163840 : index
  }
  func.func @_gqa_fwd_kernel_infer_task_type_function() -> i8 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_task_type_function>} {
    %c32_i8 = arith.constant 32 : i8
    return %c32_i8 : i8
  }
  func.func @_gqa_fwd_kernel_mix_aic(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, hivm.storage_aligned, mix_mode = "mix", parallel_mode = "simd"} {
    %c256 = arith.constant 256 : index
    %c128 = arith.constant 128 : index
    %c1_i32 = arith.constant 1 : i32
    %true = arith.constant true
    %c512_i32 = arith.constant 512 : i32
    %c20_i32 = arith.constant 20 : i32
    %c16_i32 = arith.constant 16 : i32
    %c8_i32 = arith.constant 8 : i32
    %c262144_i32 = arith.constant 262144 : i32
    %c128_i32 = arith.constant 128 : i32
    %c2_i32 = arith.constant 2 : i32
    %c2097152_i32 = arith.constant 2097152 : i32
    %c1_i64 = arith.constant 1 : i64
    %c4_i64 = arith.constant 4 : i64
    %c5_i64 = arith.constant 5 : i64
    %c2_i64 = arith.constant 2 : i64
    %c3_i64 = arith.constant 3 : i64
    %c-1_i64 = arith.constant -1 : i64
    %c360448_i64 = arith.constant 360448 : i64
    %c131072_i64 = arith.constant 131072 : i64
    %c393216_i64 = arith.constant 393216 : i64
    %c163840_i64 = arith.constant 163840 : i64
    %c229376_i64 = arith.constant 229376 : i64
    %c0_i64 = arith.constant 0 : i64
    %c65536_i64 = arith.constant 65536 : i64
    %c294912_i64 = arith.constant 294912 : i64
    %0 = hivm.hir.pointer_cast(%c65536_i64) : memref<8x16x16x16xbf16, #hivm.address_space<cbuf>>
    %1 = hivm.hir.pointer_cast(%c294912_i64) : memref<8x16x16x16xbf16, #hivm.address_space<cbuf>>
    %2 = hivm.hir.pointer_cast(%c0_i64) : memref<8x16x16x16xbf16, #hivm.address_space<cbuf>>
    %3 = hivm.hir.pointer_cast(%c229376_i64) : memref<8x16x16x16xbf16, #hivm.address_space<cbuf>>
    %4 = hivm.hir.pointer_cast(%c163840_i64) : memref<16x8x16x16xbf16, #hivm.address_space<cbuf>>
    %5 = hivm.hir.pointer_cast(%c393216_i64) : memref<16x8x16x16xbf16, #hivm.address_space<cbuf>>
    %6 = hivm.hir.pointer_cast(%c131072_i64) : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>
    %7 = hivm.hir.pointer_cast(%c360448_i64) : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>
    hivm.hir.set_ffts_base_addr %arg0
    hivm.hir.set_mask_norm
    %8 = hivm.hir.get_block_idx -> i64
    %9 = arith.trunci %8 : i64 to i32
    %10 = arith.muli %arg9, %arg8 : i32
    %11 = arith.divsi %9, %10 : i32
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
      hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 3 ffts_base_addr = %arg0
      hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
      hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID2>]
      hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID3>]
      hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID4>]
      hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID5>]
      hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
      scf.for %arg10 = %19 to %22 step %c1_i32  : i32 {
        %23 = arith.index_cast %arg10 : i32 to index
        %24 = arith.index_cast %19 : i32 to index
        %25 = affine.apply #map()[%23, %24]
        %26 = arith.index_cast %25 : index to i64
        %27 = arith.cmpi eq, %26, %c1_i64 : i64
        %28 = arith.select %27, %1, %0 : memref<8x16x16x16xbf16, #hivm.address_space<cbuf>>
        %29 = arith.index_cast %arg10 : i32 to index
        %30 = arith.index_cast %19 : i32 to index
        %31 = affine.apply #map()[%29, %30]
        %32 = arith.index_cast %31 : index to i64
        %33 = arith.cmpi eq, %32, %c1_i64 : i64
        %34 = arith.select %33, %3, %2 : memref<8x16x16x16xbf16, #hivm.address_space<cbuf>>
        %35 = arith.index_cast %arg10 : i32 to index
        %36 = arith.index_cast %19 : i32 to index
        %37 = affine.apply #map()[%35, %36]
        %38 = arith.index_cast %37 : index to i1
        %39 = arith.xori %38, %true : i1
        %40 = arith.extui %39 : i1 to i64
        %41 = arith.divsi %arg10, %c8_i32 : i32
        %42 = arith.remsi %arg10, %c8_i32 : i32
        %43 = arith.muli %41, %c262144_i32 : i32
        %44 = arith.index_cast %43 : i32 to index
        %45 = arith.muli %42, %c128_i32 : i32
        %46 = arith.index_cast %45 : i32 to index
        %47 = affine.apply #map1()[%44, %46]
        %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%47], sizes: [256, 128], strides: [1024, 1] : memref<?xbf16, #hivm.address_space<gm>> to memref<256x128xbf16, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>>
        %cast = memref.cast %34 : memref<8x16x16x16xbf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>
        hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %40]
        hivm.hir.pipe_barrier[<PIPE_MTE2>]
        hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast : memref<256x128xbf16, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast : memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>) init_out_buffer = false
        %reinterpret_cast_0 = memref.reinterpret_cast %arg5 to offset: [%47], sizes: [256, 128], strides: [1024, 1] : memref<?xbf16, #hivm.address_space<gm>> to memref<256x128xbf16, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>>
        %cast_1 = memref.cast %28 : memref<8x16x16x16xbf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>
        hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast_0 : memref<256x128xbf16, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_1 : memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>) init_out_buffer = false
        %48 = arith.muli %arg10, %c16_i32 : i32
        %49 = arith.maxsi %14, %48 : i32
        %50 = arith.subi %49, %48 : i32
        %51 = arith.addi %arg10, %c1_i32 : i32
        %52 = arith.muli %51, %c16_i32 : i32
        %53 = arith.minsi %17, %52 : i32
        %54 = arith.subi %53, %48 : i32
        %55 = arith.muli %42, %c8_i32 : i32
        %56 = arith.muli %41, %c2097152_i32 : i32
        %57 = arith.index_cast %56 : i32 to index
        scf.for %arg11 = %50 to %54 step %c1_i32  : i32 {
          %58 = arith.index_cast %arg11 : i32 to index
          %59 = arith.index_cast %50 : i32 to index
          %60 = arith.index_cast %54 : i32 to index
          %61 = arith.index_cast %arg10 : i32 to index
          %62 = arith.index_cast %19 : i32 to index
          %63 = affine.apply #map2()[%58, %59, %60, %61, %62]
          %64 = arith.index_cast %63 : index to i64
          %65 = arith.cmpi eq, %64, %c1_i64 : i64
          %66 = arith.select %65, %5, %4 : memref<16x8x16x16xbf16, #hivm.address_space<cbuf>>
          %67 = arith.index_cast %arg11 : i32 to index
          %68 = arith.index_cast %50 : i32 to index
          %69 = arith.index_cast %54 : i32 to index
          %70 = arith.index_cast %arg10 : i32 to index
          %71 = arith.index_cast %19 : i32 to index
          %72 = affine.apply #map2()[%67, %68, %69, %70, %71]
          %73 = arith.index_cast %72 : index to i64
          %74 = arith.cmpi eq, %73, %c1_i64 : i64
          %75 = arith.select %74, %7, %6 : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>
          %76 = arith.index_cast %arg11 : i32 to index
          %77 = arith.index_cast %50 : i32 to index
          %78 = arith.index_cast %54 : i32 to index
          %79 = arith.index_cast %arg10 : i32 to index
          %80 = arith.index_cast %19 : i32 to index
          %81 = affine.apply #map3()[%76, %77, %78, %79, %80]
          %82 = arith.index_cast %81 : index to i64
          %83 = arith.index_cast %arg11 : i32 to index
          %84 = arith.index_cast %50 : i32 to index
          %85 = arith.index_cast %54 : i32 to index
          %86 = arith.index_cast %arg10 : i32 to index
          %87 = arith.index_cast %19 : i32 to index
          %88 = affine.apply #map2()[%83, %84, %85, %86, %87]
          %89 = arith.index_cast %88 : index to i1
          %90 = arith.select %89, %c4_i64, %c5_i64 : i64
          %91 = arith.index_cast %88 : index to i1
          %92 = arith.select %91, %c2_i64, %c3_i64 : i64
          %93 = arith.divsi %arg11, %c2_i32 : i32
          %94 = arith.remsi %arg11, %c2_i32 : i32
          %95 = arith.addi %55, %93 : i32
          %96 = arith.muli %94, %c128_i32 : i32
          %97 = arith.muli %95, %c128_i32 : i32
          %98 = arith.index_cast %97 : i32 to index
          %99 = arith.index_cast %96 : i32 to index
          %100 = affine.apply #map4()[%57, %98, %99]
          %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [%100], sizes: [128, 128], strides: [8192, 1] : memref<?xbf16, #hivm.address_space<gm>> to memref<128x128xbf16, strided<[8192, 1], offset: ?>, #hivm.address_space<gm>>
          %cast_3 = memref.cast %75 : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>
          hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %92]
          hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast_2 : memref<128x128xbf16, strided<[8192, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_3 : memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>) init_out_buffer = false
          hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
          %101 = hivm.hir.pointer_cast(%c0_i64) : memref<16x8x16x16xf32, #hivm.address_space<cc>>
          %cast_4 = memref.cast %101 : memref<16x8x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
          hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
          hivm.hir.mmadL1 {b_transpose, fixpipe_already_inserted = true} ins(%cast_3, %cast, %true, %c128, %c128, %c256 : memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>, memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_4 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) sync_related_args(%c0_i64, %c-1_i64, %92, %c-1_i64, %82, %c-1_i64, %c-1_i64 : i64, i64, i64, i64, i64, i64, i64)
          hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
          %102 = hivm.hir.get_block_idx -> i64
          %103 = arith.index_cast %102 : i64 to index
          %104 = affine.apply #map5(%103)
          %view = memref.view %arg2[%104][] : memref<?xi8, #hivm.address_space<gm>> to memref<128x256xf16, #hivm.address_space<gm>>
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 2
          hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, pre_quant = #hivm.fixpipe_pre_quant_mode<F322F16>} ins(%cast_4 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) outs(%view : memref<128x256xf16, #hivm.address_space<gm>>)
          hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
          hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 0 ffts_base_addr = %arg0
          %105 = hivm.hir.get_block_idx -> i64
          %106 = arith.index_cast %105 : i64 to index
          %107 = affine.apply #map6(%106)
          %view_5 = memref.view %arg2[%107][] : memref<?xi8, #hivm.address_space<gm>> to memref<128x256xbf16, #hivm.address_space<gm>>
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 1
          %cast_6 = memref.cast %66 : memref<16x8x16x16xbf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>
          hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %90]
          hivm.hir.nd2nz {dst_continuous} ins(%view_5 : memref<128x256xbf16, #hivm.address_space<gm>>) outs(%cast_6 : memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>) init_out_buffer = false
          hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
          hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 3 ffts_base_addr = %arg0
          %108 = hivm.hir.pointer_cast(%c0_i64) : memref<8x8x16x16xf32, #hivm.address_space<cc>>
          %cast_7 = memref.cast %108 : memref<8x8x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
          hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
          hivm.hir.mmadL1 {fixpipe_already_inserted = true} ins(%cast_6, %cast_1, %true, %c128, %c256, %c128 : memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>, memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_7 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) sync_related_args(%c0_i64, %c-1_i64, %90, %c-1_i64, %82, %c-1_i64, %c-1_i64 : i64, i64, i64, i64, i64, i64, i64)
          hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
          %109 = hivm.hir.get_block_idx -> i64
          %110 = arith.index_cast %109 : i64 to index
          %111 = affine.apply #map7(%110)
          %view_8 = memref.view %arg2[%111][] : memref<?xi8, #hivm.address_space<gm>> to memref<128x128xf16, #hivm.address_space<gm>>
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 4
          hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, pre_quant = #hivm.fixpipe_pre_quant_mode<F322F16>} ins(%cast_7 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) outs(%view_8 : memref<128x128xf16, #hivm.address_space<gm>>)
          hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
          hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 0 ffts_base_addr = %arg0
        }
        hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, %40]
      }
      hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
      hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
      hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID2>]
      hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID3>]
      hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID4>]
      hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID5>]
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 2
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 4
    }
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
  func.func @_gqa_fwd_kernel_mix_aiv(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.storage_aligned, mix_mode = "mix", parallel_mode = "simd"} {
    %c180224_i64 = arith.constant 180224 : i64
    %c81920_i64 = arith.constant 81920 : i64
    %c65536_i64 = arith.constant 65536 : i64
    %c32768_i64 = arith.constant 32768 : i64
    %c114688_i64 = arith.constant 114688 : i64
    %c0_i64 = arith.constant 0 : i64
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 0.0883883461 : f32
    %c512_i32 = arith.constant 512 : i32
    %c20_i32 = arith.constant 20 : i32
    %c16_i32 = arith.constant 16 : i32
    %c8_i32 = arith.constant 8 : i32
    %c128_i32 = arith.constant 128 : i32
    %c2_i32 = arith.constant 2 : i32
    %c2097152_i32 = arith.constant 2097152 : i32
    %0 = hivm.hir.get_block_idx -> i64
    %1 = arith.trunci %0 : i64 to i32
    %2 = arith.muli %arg9, %arg8 : i32
    %3 = hivm.hir.get_sub_block_idx -> i64
    %4 = arith.index_cast %3 : i64 to index
    %5 = affine.apply #map8()[%4]
    hivm.hir.set_ffts_base_addr %arg0
    hivm.hir.set_mask_norm
    %6 = arith.divsi %1, %2 : i32
    %7 = arith.remsi %6, %arg7 : i32
    %8 = arith.muli %7, %c512_i32 : i32
    %9 = arith.divsi %8, %c20_i32 : i32
    %10 = arith.addi %7, %c1_i32 : i32
    %11 = arith.muli %10, %c512_i32 : i32
    %12 = arith.divsi %11, %c20_i32 : i32
    %13 = arith.cmpi sgt, %12, %9 : i32
    scf.if %13 {
      %14 = arith.divsi %9, %c16_i32 : i32
      %15 = arith.subi %12, %c1_i32 : i32
      %16 = arith.divsi %15, %c16_i32 : i32
      %17 = arith.addi %16, %c1_i32 : i32
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 2 ffts_base_addr = %arg0
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 4 ffts_base_addr = %arg0
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      scf.for %arg10 = %14 to %17 step %c1_i32  : i32 {
        %18 = arith.divsi %arg10, %c8_i32 : i32
        %19 = arith.remsi %arg10, %c8_i32 : i32
        %20 = arith.muli %arg10, %c16_i32 : i32
        %21 = arith.maxsi %9, %20 : i32
        %22 = arith.subi %21, %20 : i32
        %23 = arith.addi %arg10, %c1_i32 : i32
        %24 = arith.muli %23, %c16_i32 : i32
        %25 = arith.minsi %12, %24 : i32
        %26 = arith.subi %25, %20 : i32
        %27 = arith.muli %19, %c8_i32 : i32
        %28 = arith.muli %18, %c2097152_i32 : i32
        %29 = arith.index_cast %28 : i32 to index
        scf.for %arg11 = %22 to %26 step %c1_i32  : i32 {
          %30 = arith.divsi %arg11, %c2_i32 : i32
          %31 = arith.remsi %arg11, %c2_i32 : i32
          %32 = arith.addi %27, %30 : i32
          %33 = arith.muli %31, %c128_i32 : i32
          %34 = arith.muli %32, %c128_i32 : i32
          %35 = arith.index_cast %34 : i32 to index
          %36 = arith.index_cast %33 : i32 to index
          %37 = affine.apply #map4()[%29, %35, %36]
          %38 = hivm.hir.get_block_idx -> i64
          %39 = arith.index_cast %38 : i64 to index
          %40 = affine.apply #map5(%39)
          %view = memref.view %arg2[%40][] : memref<?xi8, #hivm.address_space<gm>> to memref<128x256xf16, #hivm.address_space<gm>>
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 0
          %subview = memref.subview %view[%5, 0] [64, 256] [1, 1] : memref<128x256xf16, #hivm.address_space<gm>> to memref<64x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          %41 = hivm.hir.pointer_cast(%c0_i64) : memref<64x256xf16, #hivm.address_space<ub>>
          %collapse_shape = memref.collapse_shape %subview [[0, 1]] : memref<64x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> into memref<16384xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape_0 = memref.collapse_shape %41 [[0, 1]] : memref<64x256xf16, #hivm.address_space<ub>> into memref<16384xf16, #hivm.address_space<ub>>
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          hivm.hir.load ins(%collapse_shape : memref<16384xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_0 : memref<16384xf16, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
          hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 2 ffts_base_addr = %arg0
          %42 = hivm.hir.pointer_cast(%c114688_i64) : memref<64x256xf32, #hivm.address_space<ub>>
          %collapse_shape_1 = memref.collapse_shape %42 [[0, 1]] : memref<64x256xf32, #hivm.address_space<ub>> into memref<16384xf32, #hivm.address_space<ub>>
          hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
          hivm.hir.vcast ins(%collapse_shape_0 : memref<16384xf16, #hivm.address_space<ub>>) outs(%collapse_shape_1 : memref<16384xf32, #hivm.address_space<ub>>)
          %43 = hivm.hir.pointer_cast(%c114688_i64) : memref<64x256xf32, #hivm.address_space<ub>>
          %collapse_shape_2 = memref.collapse_shape %43 [[0, 1]] : memref<64x256xf32, #hivm.address_space<ub>> into memref<16384xf32, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          hivm.hir.vmul ins(%collapse_shape_1, %cst : memref<16384xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_2 : memref<16384xf32, #hivm.address_space<ub>>)
          %44 = hivm.hir.pointer_cast(%c114688_i64) : memref<64x256xf32, #hivm.address_space<ub>>
          %collapse_shape_3 = memref.collapse_shape %44 [[0, 1]] : memref<64x256xf32, #hivm.address_space<ub>> into memref<16384xf32, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          hivm.hir.vexp ins(%collapse_shape_2 : memref<16384xf32, #hivm.address_space<ub>>) outs(%collapse_shape_3 : memref<16384xf32, #hivm.address_space<ub>>)
          %45 = hivm.hir.pointer_cast(%c32768_i64) : memref<64x256xbf16, #hivm.address_space<ub>>
          %collapse_shape_4 = memref.collapse_shape %45 [[0, 1]] : memref<64x256xbf16, #hivm.address_space<ub>> into memref<16384xbf16, #hivm.address_space<ub>>
          hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
          hivm.hir.pipe_barrier[<PIPE_V>]
          hivm.hir.vcast ins(%collapse_shape_3 : memref<16384xf32, #hivm.address_space<ub>>) outs(%collapse_shape_4 : memref<16384xbf16, #hivm.address_space<ub>>)
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
          %46 = hivm.hir.get_block_idx -> i64
          %47 = arith.index_cast %46 : i64 to index
          %48 = affine.apply #map6(%47)
          %view_5 = memref.view %arg2[%48][] : memref<?xi8, #hivm.address_space<gm>> to memref<128x256xbf16, #hivm.address_space<gm>>
          %subview_6 = memref.subview %view_5[%5, 0] [64, 256] [1, 1] : memref<128x256xbf16, #hivm.address_space<gm>> to memref<64x256xbf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 3
          %collapse_shape_7 = memref.collapse_shape %subview_6 [[0, 1]] : memref<64x256xbf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> into memref<16384xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
          hivm.hir.store ins(%collapse_shape_4 : memref<16384xbf16, #hivm.address_space<ub>>) outs(%collapse_shape_7 : memref<16384xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>) {tiled_op}
          hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 1 ffts_base_addr = %arg0
          %49 = hivm.hir.get_block_idx -> i64
          %50 = arith.index_cast %49 : i64 to index
          %51 = affine.apply #map7(%50)
          %view_8 = memref.view %arg2[%51][] : memref<?xi8, #hivm.address_space<gm>> to memref<128x128xf16, #hivm.address_space<gm>>
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 0
          %subview_9 = memref.subview %view_8[%5, 0] [64, 128] [1, 1] : memref<128x128xf16, #hivm.address_space<gm>> to memref<64x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
          %52 = hivm.hir.pointer_cast(%c65536_i64) : memref<64x128xf16, #hivm.address_space<ub>>
          %collapse_shape_10 = memref.collapse_shape %subview_9 [[0, 1]] : memref<64x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> into memref<8192xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape_11 = memref.collapse_shape %52 [[0, 1]] : memref<64x128xf16, #hivm.address_space<ub>> into memref<8192xf16, #hivm.address_space<ub>>
          hivm.hir.load ins(%collapse_shape_10 : memref<8192xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_11 : memref<8192xf16, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
          hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 4 ffts_base_addr = %arg0
          %53 = hivm.hir.pointer_cast(%c81920_i64) : memref<64x128xf32, #hivm.address_space<ub>>
          %collapse_shape_12 = memref.collapse_shape %53 [[0, 1]] : memref<64x128xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
          hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
          hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
          hivm.hir.vcast ins(%collapse_shape_11 : memref<8192xf16, #hivm.address_space<ub>>) outs(%collapse_shape_12 : memref<8192xf32, #hivm.address_space<ub>>)
          %54 = hivm.hir.pointer_cast(%c180224_i64) : memref<64x1xf32, #hivm.address_space<ub>>
          %55 = hivm.hir.pointer_cast(%c0_i64) : memref<8192xf32, #hivm.address_space<ub>>
          hivm.hir.vreduce <sum> ins(%44 : memref<64x256xf32, #hivm.address_space<ub>>) outs(%54 : memref<64x1xf32, #hivm.address_space<ub>>) temp_buffer(%55 : memref<8192xf32, #hivm.address_space<ub>>) reduce_dims = [1]
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          %56 = hivm.hir.pointer_cast(%c81920_i64) : memref<64x128xf32, #hivm.address_space<ub>>
          %57 = hivm.hir.pointer_cast(%c114688_i64) : memref<512xf32, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          hivm.hir.vdiv ins(%53, %54 : memref<64x128xf32, #hivm.address_space<ub>>, memref<64x1xf32, #hivm.address_space<ub>>) outs(%56 : memref<64x128xf32, #hivm.address_space<ub>>) temp_buffer(%57 : memref<512xf32, #hivm.address_space<ub>>) broadcast = [1]
          %reinterpret_cast = memref.reinterpret_cast %arg6 to offset: [%37], sizes: [128, 128], strides: [8192, 1] : memref<?xbf16, #hivm.address_space<gm>> to memref<128x128xbf16, strided<[8192, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_13 = memref.subview %reinterpret_cast[%5, 0] [64, 128] [1, 1] {to_be_bubbled_slice} : memref<128x128xbf16, strided<[8192, 1], offset: ?>, #hivm.address_space<gm>> to memref<64x128xbf16, strided<[8192, 1], offset: ?>, #hivm.address_space<gm>>
          %58 = hivm.hir.pointer_cast(%c81920_i64) : memref<64x128xbf16, #hivm.address_space<ub>>
          %collapse_shape_14 = memref.collapse_shape %56 [[0, 1]] : memref<64x128xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
          %collapse_shape_15 = memref.collapse_shape %58 [[0, 1]] : memref<64x128xbf16, #hivm.address_space<ub>> into memref<8192xbf16, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          hivm.hir.vcast ins(%collapse_shape_14 : memref<8192xf32, #hivm.address_space<ub>>) outs(%collapse_shape_15 : memref<8192xbf16, #hivm.address_space<ub>>)
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
          hivm.hir.store ins(%58 : memref<64x128xbf16, #hivm.address_space<ub>>) outs(%subview_13 : memref<64x128xbf16, strided<[8192, 1], offset: ?>, #hivm.address_space<gm>>) {tiled_op}
          hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
        }
      }
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 3
    }
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}


