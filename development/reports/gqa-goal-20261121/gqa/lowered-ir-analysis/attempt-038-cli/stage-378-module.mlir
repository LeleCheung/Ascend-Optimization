#map = affine_map<()[s0, s1] -> (s0 + s1)>
#map1 = affine_map<(d0) -> (d0 * 262144)>
#map2 = affine_map<()[s0, s1, s2, s3, s4] -> (s0 - s1 + (s3 - s4) * (-s1 + s2))>
#map3 = affine_map<()[s0, s1, s2] -> (s0 + s1 + s2 * 8192)>
#map4 = affine_map<(d0) -> (d0 * 262144 + 65536)>
#map5 = affine_map<(d0) -> (d0 * 262144 + 131072)>
#map6 = affine_map<(d0) -> (d0 * 262144 + 196608)>
#map7 = affine_map<()[s0] -> (s0 * 64)>
#map8 = affine_map<()[s0] -> (s0 * 128)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 20 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 20 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 40 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>, hacc.hivmc_compatible_print = true, hacc.hivmc_version = #hacc.hivmc_version<"0.2.0">, hivm.module_core_type = #hivm.module_core_type<MIX>, memref.memref_as_ptr} {
  func.func @_gqa_fwd_kernel_infer_workspace_shape_function() -> index attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_workspace_shape_function>} {
    %c262144 = arith.constant 262144 : index
    return %c262144 : index
  }
  func.func @_gqa_fwd_kernel_infer_task_type_function() -> i8 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_task_type_function>} {
    %c32_i8 = arith.constant 32 : i8
    return %c32_i8 : i8
  }
  func.func @_gqa_fwd_kernel_mix_aic(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, hivm.storage_aligned, mix_mode = "mix", parallel_mode = "simd"} {
    %false = arith.constant false
    %c0_i8 = arith.constant 0 : i8
    %c-1_i64 = arith.constant -1 : i64
    %c2_i64 = arith.constant 2 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %c163840_i64 = arith.constant 163840 : i64
    %c131072_i64 = arith.constant 131072 : i64
    %c65536_i64 = arith.constant 65536 : i64
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
    hivm.hir.set_ffts_base_addr %arg0
    hivm.hir.set_mask_norm
    %0 = hivm.hir.get_block_idx -> i64
    %1 = arith.trunci %0 : i64 to i32
    %2 = arith.muli %arg9, %arg8 : i32
    %3 = arith.divsi %1, %2 : i32
    %4 = arith.remsi %3, %arg7 : i32
    %5 = arith.muli %4, %c512_i32 : i32
    %6 = arith.divsi %5, %c20_i32 : i32
    %7 = arith.addi %4, %c1_i32 : i32
    %8 = arith.muli %7, %c512_i32 : i32
    %9 = arith.divsi %8, %c20_i32 : i32
    %10 = arith.cmpi sgt, %9, %6 : i32
    scf.if %10 {
      %11 = arith.divsi %6, %c16_i32 : i32
      %12 = arith.subi %9, %c1_i32 : i32
      %13 = arith.divsi %12, %c16_i32 : i32
      %14 = arith.addi %13, %c1_i32 : i32
      hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 3 ffts_base_addr = %arg0
      hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 5 ffts_base_addr = %arg0
      hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
      hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID2>]
      hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
      scf.for %arg10 = %11 to %14 step %c1_i32  : i32 {
        %15 = arith.divsi %arg10, %c8_i32 : i32
        %16 = arith.remsi %arg10, %c8_i32 : i32
        %17 = arith.muli %15, %c262144_i32 : i32
        %18 = arith.index_cast %17 : i32 to index
        %19 = arith.muli %16, %c128_i32 : i32
        %20 = arith.index_cast %19 : i32 to index
        %21 = affine.apply #map()[%18, %20]
        %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%21], sizes: [256, 128], strides: [1024, 1] : memref<?xbf16, #hivm.address_space<gm>> to memref<256x128xbf16, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>>
        %22 = hivm.hir.pointer_cast(%c0_i64) : memref<8x16x16x16xbf16, #hivm.address_space<cbuf>>
        hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.pipe_barrier[<PIPE_MTE2>]
        %cast = memref.cast %reinterpret_cast : memref<256x128xbf16, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        %cast_0 = memref.cast %22 : memref<8x16x16x16xbf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
        func.call @nd2nz_bfloat16_t(%cast, %cast_0) : (memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
        %23 = hivm.hir.get_block_idx -> i64
        %24 = arith.index_cast %23 : i64 to index
        %25 = affine.apply #map1(%24)
        %view = memref.view %arg2[%25][] : memref<?xi8, #hivm.address_space<gm>> to memref<256x128xf16, #hivm.address_space<gm>>
        hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 0
        %26 = hivm.hir.pointer_cast(%c65536_i64) : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>
        %cast_1 = memref.cast %view : memref<256x128xf16, #hivm.address_space<gm>> to memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        %cast_2 = memref.cast %26 : memref<8x16x16x16xf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
        func.call @nd2nz_half(%cast_1, %cast_2) : (memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
        hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 5 ffts_base_addr = %arg0
        %27 = arith.muli %arg10, %c16_i32 : i32
        %28 = arith.maxsi %6, %27 : i32
        %29 = arith.subi %28, %27 : i32
        %30 = arith.addi %arg10, %c1_i32 : i32
        %31 = arith.muli %30, %c16_i32 : i32
        %32 = arith.minsi %9, %31 : i32
        %33 = arith.subi %32, %27 : i32
        %34 = arith.muli %16, %c8_i32 : i32
        %35 = arith.muli %15, %c2097152_i32 : i32
        %36 = arith.index_cast %35 : i32 to index
        scf.for %arg11 = %29 to %33 step %c1_i32  : i32 {
          %37 = arith.index_cast %arg11 : i32 to index
          %38 = arith.index_cast %29 : i32 to index
          %39 = arith.index_cast %33 : i32 to index
          %40 = arith.index_cast %arg10 : i32 to index
          %41 = arith.index_cast %11 : i32 to index
          %42 = affine.apply #map2()[%37, %38, %39, %40, %41]
          %43 = arith.index_cast %42 : index to i64
          %44 = arith.divsi %arg11, %c2_i32 : i32
          %45 = arith.remsi %arg11, %c2_i32 : i32
          %46 = arith.addi %34, %44 : i32
          %47 = arith.muli %45, %c128_i32 : i32
          %48 = arith.muli %46, %c128_i32 : i32
          %49 = arith.index_cast %48 : i32 to index
          %50 = arith.index_cast %47 : i32 to index
          %51 = affine.apply #map3()[%36, %49, %50]
          %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [%51], sizes: [128, 128], strides: [8192, 1] : memref<?xbf16, #hivm.address_space<gm>> to memref<128x128xbf16, strided<[8192, 1], offset: ?>, #hivm.address_space<gm>>
          %52 = hivm.hir.pointer_cast(%c131072_i64) : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>
          hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
          %cast_4 = memref.cast %reinterpret_cast_3 : memref<128x128xbf16, strided<[8192, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
          %cast_5 = memref.cast %52 : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
          func.call @nd2nz_bfloat16_t(%cast_4, %cast_5) : (memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
          hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
          %53 = hivm.hir.pointer_cast(%c0_i64) : memref<16x8x16x16xf32, #hivm.address_space<cc>>
          hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
          %cast_6 = memref.cast %52 : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
          %cast_7 = memref.cast %22 : memref<8x16x16x16xbf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
          %cast_8 = memref.cast %53 : memref<16x8x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
          func.call @mma_tile_bfloat16_t_to_float_tb(%cast_6, %cast_7, %true, %c128, %c128, %c256, %cast_8, %c0_i64, %c-1_i64, %c1_i64, %c-1_i64, %43, %c-1_i64, %c-1_i64, %c0_i8) : (memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, i1, index, index, index, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, i64, i64, i64, i64, i64, i64, i64, i8) -> ()
          hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
          %54 = hivm.hir.get_block_idx -> i64
          %55 = arith.index_cast %54 : i64 to index
          %56 = affine.apply #map4(%55)
          %view_9 = memref.view %arg2[%56][] : memref<?xi8, #hivm.address_space<gm>> to memref<128x256xf16, #hivm.address_space<gm>>
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 2
          hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
          %cast_10 = memref.cast %53 : memref<16x8x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
          %cast_11 = memref.cast %view_9 : memref<128x256xf16, #hivm.address_space<gm>> to memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
          func.call @fixpipe_nz2nd_float_to_half_4d_to_2d(%cast_10, %cast_11, %c1_i64, %c0_i64, %false, %c0_i8) : (memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i64, i64, i1, i8) -> ()
          hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
          hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 1 ffts_base_addr = %arg0
          %57 = hivm.hir.get_block_idx -> i64
          %58 = arith.index_cast %57 : i64 to index
          %59 = affine.apply #map5(%58)
          %view_12 = memref.view %arg2[%59][] : memref<?xi8, #hivm.address_space<gm>> to memref<128x256xf16, #hivm.address_space<gm>>
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 0
          %60 = hivm.hir.pointer_cast(%c163840_i64) : memref<16x8x16x16xf16, #hivm.address_space<cbuf>>
          hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID2>]
          %cast_13 = memref.cast %view_12 : memref<128x256xf16, #hivm.address_space<gm>> to memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
          %cast_14 = memref.cast %60 : memref<16x8x16x16xf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
          func.call @nd2nz_half(%cast_13, %cast_14) : (memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
          hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
          hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 3 ffts_base_addr = %arg0
          %61 = hivm.hir.pointer_cast(%c0_i64) : memref<8x8x16x16xf32, #hivm.address_space<cc>>
          hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
          %cast_15 = memref.cast %60 : memref<16x8x16x16xf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
          %cast_16 = memref.cast %26 : memref<8x16x16x16xf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
          %cast_17 = memref.cast %61 : memref<8x8x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
          func.call @mma_tile_half_to_float(%cast_15, %cast_16, %true, %c128, %c256, %c128, %cast_17, %c0_i64, %c-1_i64, %c2_i64, %c-1_i64, %43, %c-1_i64, %c-1_i64, %c0_i8) : (memref<?x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, memref<?x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, i1, index, index, index, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, i64, i64, i64, i64, i64, i64, i64, i8) -> ()
          hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
          %62 = hivm.hir.get_block_idx -> i64
          %63 = arith.index_cast %62 : i64 to index
          %64 = affine.apply #map6(%63)
          %view_18 = memref.view %arg2[%64][] : memref<?xi8, #hivm.address_space<gm>> to memref<128x128xf32, #hivm.address_space<gm>>
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 4
          hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
          %cast_19 = memref.cast %61 : memref<8x8x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
          %cast_20 = memref.cast %view_18 : memref<128x128xf32, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
          func.call @fixpipe_nz2nd_float_to_float_4d_to_2d(%cast_19, %cast_20, %c0_i64, %c0_i64, %false, %c0_i8) : (memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i64, i64, i1, i8) -> ()
          hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
          hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 1 ffts_base_addr = %arg0
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
  func.func private @nd2nz_bfloat16_t(memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface}
  func.func private @nd2nz_half(memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface}
  func.func private @mma_tile_bfloat16_t_to_float_tb(memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, i1, index, index, index, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, i64, i64, i64, i64, i64, i64, i64, i8) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface}
  func.func private @fixpipe_nz2nd_float_to_half_4d_to_2d(memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i64, i64, i1, i8) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface}
  func.func private @mma_tile_half_to_float(memref<?x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, memref<?x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, i1, index, index, index, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, i64, i64, i64, i64, i64, i64, i64, i8) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface}
  func.func private @fixpipe_nz2nd_float_to_float_4d_to_2d(memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i64, i64, i1, i8) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface}
  func.func private @broadcast_scalar_float_to_1d(f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @load_gm_to_ubuf_2d_bfloat16_t(memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, bf16, index) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vcast_bfloat16_t_to_float_1d_with_mode(memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vcast_float_to_half_1d_with_mode(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_1d_half(memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @load_gm_to_ubuf_1d_half(memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f16, index) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vcast_half_to_float_1d_with_mode(memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vmuls_vs_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vadds_vs_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vexp_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @load_gm_to_ubuf_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @enablevc_reduce_sum_ar_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vcmp_gt_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vsel_vs_1d_float(memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vdiv_2d_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vcast_bool_to_half_1d_with_temp(memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @broadcast_last_axis_align_2d_half(memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vcmps_eq_1d_half(memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, f16, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vnot_1d_bool(memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vcast_float_to_bfloat16_t_1d_with_mode(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_2d_bfloat16_t(memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func @_gqa_fwd_kernel_mix_aiv(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.storage_aligned, mix_mode = "mix", parallel_mode = "simd"} {
    %c4_i32 = arith.constant 4 : i32
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %c0_i32 = arith.constant 0 : i32
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
    %cst_0 = arith.constant -1.32223117 : f32
    %c1_i32 = arith.constant 1 : i32
    %cst_1 = arith.constant 0.000000e+00 : f16
    %cst_2 = arith.constant 0.693147182 : f32
    %cst_3 = arith.constant 1.000000e+00 : f32
    %cst_4 = arith.constant 0.127517432 : f32
    %c512_i32 = arith.constant 512 : i32
    %c20_i32 = arith.constant 20 : i32
    %c16_i32 = arith.constant 16 : i32
    %c8_i32 = arith.constant 8 : i32
    %c262144_i32 = arith.constant 262144 : i32
    %c128_i32 = arith.constant 128 : i32
    %c2_i32 = arith.constant 2 : i32
    %c2097152_i32 = arith.constant 2097152 : i32
    %cst_5 = arith.constant 0.000000e+00 : f32
    %0 = hivm.hir.get_block_idx -> i64
    %1 = arith.trunci %0 : i64 to i32
    %2 = arith.muli %arg9, %arg8 : i32
    %3 = hivm.hir.pointer_cast(%c163840_i64) : memref<64xf32, #hivm.address_space<ub>>
    %cast = memref.cast %3 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @broadcast_scalar_float_to_1d(%cst_5, %cast) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    %4 = hivm.hir.pointer_cast(%c164096_i64) : memref<64x1xf32, #hivm.address_space<ub>>
    %collapse_shape = memref.collapse_shape %4 [[0, 1]] : memref<64x1xf32, #hivm.address_space<ub>> into memref<64xf32, #hivm.address_space<ub>>
    %cast_6 = memref.cast %collapse_shape : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @broadcast_scalar_float_to_1d(%cst_5, %cast_6) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    %5 = hivm.hir.get_sub_block_idx -> i64
    %6 = arith.index_cast %5 : i64 to index
    %7 = affine.apply #map7()[%6]
    %8 = affine.apply #map8()[%6]
    hivm.hir.set_ffts_base_addr %arg0
    hivm.hir.set_mask_norm
    %9 = arith.divsi %1, %2 : i32
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
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 2 ffts_base_addr = %arg0
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 4 ffts_base_addr = %arg0
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
      scf.for %arg10 = %17 to %20 step %c1_i32  : i32 {
        %21 = arith.divsi %arg10, %c8_i32 : i32
        %22 = arith.remsi %arg10, %c8_i32 : i32
        %23 = arith.muli %21, %c262144_i32 : i32
        %24 = arith.index_cast %23 : i32 to index
        %25 = arith.muli %22, %c128_i32 : i32
        %26 = arith.index_cast %25 : i32 to index
        %27 = affine.apply #map()[%24, %26]
        %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%27], sizes: [256, 128], strides: [1024, 1] : memref<?xbf16, #hivm.address_space<gm>> to memref<256x128xbf16, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>>
        %28 = hivm.hir.pointer_cast(%c0_i64) : memref<128x128xbf16, #hivm.address_space<ub>>
        %subview = memref.subview %reinterpret_cast[%8, 0] [128, 128] [1, 1] : memref<256x128xbf16, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>> to memref<128x128xbf16, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        %cast_7 = memref.cast %subview : memref<128x128xbf16, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        %cast_8 = memref.cast %28 : memref<128x128xbf16, #hivm.address_space<ub>> to memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        func.call @load_gm_to_ubuf_2d_bfloat16_t(%cast_7, %cast_8, %c0_i32, %cst, %c0) : (memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, bf16, index) -> ()
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %29 = hivm.hir.pointer_cast(%c32768_i64) : memref<128x128xf32, #hivm.address_space<ub>>
        %collapse_shape_9 = memref.collapse_shape %28 [[0, 1]] : memref<128x128xbf16, #hivm.address_space<ub>> into memref<16384xbf16, #hivm.address_space<ub>>
        %collapse_shape_10 = memref.collapse_shape %29 [[0, 1]] : memref<128x128xf32, #hivm.address_space<ub>> into memref<16384xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %cast_11 = memref.cast %collapse_shape_9 : memref<16384xbf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_12 = memref.cast %collapse_shape_10 : memref<16384xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vcast_bfloat16_t_to_float_1d_with_mode(%cast_11, %cast_12, %c0_i32) : (memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        %30 = hivm.hir.pointer_cast(%c32768_i64) : memref<128x128xf16, #hivm.address_space<ub>>
        %collapse_shape_13 = memref.collapse_shape %30 [[0, 1]] : memref<128x128xf16, #hivm.address_space<ub>> into memref<16384xf16, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_14 = memref.cast %collapse_shape_10 : memref<16384xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_15 = memref.cast %collapse_shape_13 : memref<16384xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vcast_float_to_half_1d_with_mode(%cast_14, %cast_15, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
        %31 = hivm.hir.get_block_idx -> i64
        %32 = arith.index_cast %31 : i64 to index
        %33 = affine.apply #map1(%32)
        %view = memref.view %arg2[%33][] : memref<?xi8, #hivm.address_space<gm>> to memref<256x128xf16, #hivm.address_space<gm>>
        %subview_16 = memref.subview %view[%8, 0] [128, 128] [1, 1] : memref<256x128xf16, #hivm.address_space<gm>> to memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 5
        %collapse_shape_17 = memref.collapse_shape %subview_16 [[0, 1]] : memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> into memref<16384xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
        %cast_18 = memref.cast %collapse_shape_13 : memref<16384xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_19 = memref.cast %collapse_shape_17 : memref<16384xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
        func.call @store_ubuf_to_gm_1d_half(%cast_18, %cast_19, %c0_i32) : (memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 0 ffts_base_addr = %arg0
        %34 = arith.muli %arg10, %c16_i32 : i32
        %35 = arith.maxsi %12, %34 : i32
        %36 = arith.subi %35, %34 : i32
        %37 = arith.addi %arg10, %c1_i32 : i32
        %38 = arith.muli %37, %c16_i32 : i32
        %39 = arith.minsi %15, %38 : i32
        %40 = arith.subi %39, %34 : i32
        %41 = arith.muli %22, %c8_i32 : i32
        %42 = arith.muli %21, %c2097152_i32 : i32
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
          %51 = affine.apply #map3()[%43, %49, %50]
          %52 = hivm.hir.get_block_idx -> i64
          %53 = arith.index_cast %52 : i64 to index
          %54 = affine.apply #map4(%53)
          %view_20 = memref.view %arg2[%54][] : memref<?xi8, #hivm.address_space<gm>> to memref<128x256xf16, #hivm.address_space<gm>>
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 1
          %subview_21 = memref.subview %view_20[%7, 0] [64, 256] [1, 1] : memref<128x256xf16, #hivm.address_space<gm>> to memref<64x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          %55 = hivm.hir.pointer_cast(%c98304_i64) : memref<64x256xf16, #hivm.address_space<ub>>
          %collapse_shape_22 = memref.collapse_shape %subview_21 [[0, 1]] : memref<64x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> into memref<16384xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape_23 = memref.collapse_shape %55 [[0, 1]] : memref<64x256xf16, #hivm.address_space<ub>> into memref<16384xf16, #hivm.address_space<ub>>
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
          %cast_24 = memref.cast %collapse_shape_22 : memref<16384xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
          %cast_25 = memref.cast %collapse_shape_23 : memref<16384xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @load_gm_to_ubuf_1d_half(%cast_24, %cast_25, %c0_i32, %cst_1, %c0) : (memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f16, index) -> ()
          hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 2 ffts_base_addr = %arg0
          %56 = hivm.hir.pointer_cast(%c32768_i64) : memref<64x256xf32, #hivm.address_space<ub>>
          %collapse_shape_26 = memref.collapse_shape %56 [[0, 1]] : memref<64x256xf32, #hivm.address_space<ub>> into memref<16384xf32, #hivm.address_space<ub>>
          hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
          hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
          %cast_27 = memref.cast %collapse_shape_23 : memref<16384xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_28 = memref.cast %collapse_shape_26 : memref<16384xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vcast_half_to_float_1d_with_mode(%cast_27, %cast_28, %c0_i32) : (memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
          %57 = hivm.hir.pointer_cast(%c32768_i64) : memref<64x256xf32, #hivm.address_space<ub>>
          %collapse_shape_29 = memref.collapse_shape %57 [[0, 1]] : memref<64x256xf32, #hivm.address_space<ub>> into memref<16384xf32, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          %58 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
          %cast_30 = memref.cast %collapse_shape_26 : memref<16384xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_31 = memref.cast %collapse_shape_29 : memref<16384xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_32 = memref.cast %58 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vmuls_vs_1d_float(%cast_30, %cst_4, %cast_31, %cast_32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          %59 = hivm.hir.pointer_cast(%c32768_i64) : memref<64x256xf32, #hivm.address_space<ub>>
          %collapse_shape_33 = memref.collapse_shape %59 [[0, 1]] : memref<64x256xf32, #hivm.address_space<ub>> into memref<16384xf32, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          %60 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
          %cast_34 = memref.cast %collapse_shape_29 : memref<16384xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_35 = memref.cast %collapse_shape_33 : memref<16384xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_36 = memref.cast %60 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vadds_vs_1d_float(%cast_34, %cst_0, %cast_35, %cast_36) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          %61 = hivm.hir.pointer_cast(%c32768_i64) : memref<64x256xf32, #hivm.address_space<ub>>
          %collapse_shape_37 = memref.collapse_shape %61 [[0, 1]] : memref<64x256xf32, #hivm.address_space<ub>> into memref<16384xf32, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          %62 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
          %cast_38 = memref.cast %collapse_shape_33 : memref<16384xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_39 = memref.cast %collapse_shape_37 : memref<16384xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_40 = memref.cast %62 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vmuls_vs_1d_float(%cast_38, %cst_2, %cast_39, %cast_40) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          %63 = hivm.hir.pointer_cast(%c32768_i64) : memref<64x256xf32, #hivm.address_space<ub>>
          %collapse_shape_41 = memref.collapse_shape %63 [[0, 1]] : memref<64x256xf32, #hivm.address_space<ub>> into memref<16384xf32, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          %64 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
          %cast_42 = memref.cast %collapse_shape_37 : memref<16384xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_43 = memref.cast %collapse_shape_41 : memref<16384xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_44 = memref.cast %64 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vexp_1d_float(%cast_42, %cast_43, %cast_44) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          %65 = hivm.hir.pointer_cast(%c131072_i64) : memref<64x256xf16, #hivm.address_space<ub>>
          %collapse_shape_45 = memref.collapse_shape %65 [[0, 1]] : memref<64x256xf16, #hivm.address_space<ub>> into memref<16384xf16, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          %cast_46 = memref.cast %collapse_shape_41 : memref<16384xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_47 = memref.cast %collapse_shape_45 : memref<16384xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vcast_float_to_half_1d_with_mode(%cast_46, %cast_47, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
          %66 = hivm.hir.get_block_idx -> i64
          %67 = arith.index_cast %66 : i64 to index
          %68 = affine.apply #map5(%67)
          %view_48 = memref.view %arg2[%68][] : memref<?xi8, #hivm.address_space<gm>> to memref<128x256xf16, #hivm.address_space<gm>>
          %subview_49 = memref.subview %view_48[%7, 0] [64, 256] [1, 1] : memref<128x256xf16, #hivm.address_space<gm>> to memref<64x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 3
          %collapse_shape_50 = memref.collapse_shape %subview_49 [[0, 1]] : memref<64x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> into memref<16384xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
          %cast_51 = memref.cast %collapse_shape_45 : memref<16384xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_52 = memref.cast %collapse_shape_50 : memref<16384xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
          func.call @store_ubuf_to_gm_1d_half(%cast_51, %cast_52, %c0_i32) : (memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 0 ffts_base_addr = %arg0
          %69 = hivm.hir.get_block_idx -> i64
          %70 = arith.index_cast %69 : i64 to index
          %71 = affine.apply #map6(%70)
          %view_53 = memref.view %arg2[%71][] : memref<?xi8, #hivm.address_space<gm>> to memref<128x128xf32, #hivm.address_space<gm>>
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 1
          %subview_54 = memref.subview %view_53[%7, 0] [64, 128] [1, 1] : memref<128x128xf32, #hivm.address_space<gm>> to memref<64x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
          %72 = hivm.hir.pointer_cast(%c0_i64) : memref<64x128xf32, #hivm.address_space<ub>>
          %collapse_shape_55 = memref.collapse_shape %subview_54 [[0, 1]] : memref<64x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> into memref<8192xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape_56 = memref.collapse_shape %72 [[0, 1]] : memref<64x128xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
          %cast_57 = memref.cast %collapse_shape_55 : memref<8192xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
          %cast_58 = memref.cast %collapse_shape_56 : memref<8192xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @load_gm_to_ubuf_1d_float(%cast_57, %cast_58, %c0_i32, %cst_5, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
          hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 4 ffts_base_addr = %arg0
          %73 = hivm.hir.pointer_cast(%c164352_i64) : memref<64x1xf32, #hivm.address_space<ub>>
          %74 = hivm.hir.pointer_cast(%c98304_i64) : memref<8192xf32, #hivm.address_space<ub>>
          %cast_59 = memref.cast %63 : memref<64x256xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
          %cast_60 = memref.cast %73 : memref<64x1xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
          %cast_61 = memref.cast %74 : memref<8192xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @enablevc_reduce_sum_ar_float(%cast_59, %cast_60, %cast_61, %cst_5) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) -> ()
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
          %collapse_shape_62 = memref.collapse_shape %73 [[0, 1]] : memref<64x1xf32, #hivm.address_space<ub>> into memref<64xf32, #hivm.address_space<ub>>
          %75 = hivm.hir.pointer_cast(%c164608_i64) : memref<64xi1, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          %cast_63 = memref.cast %collapse_shape_62 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_64 = memref.cast %3 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_65 = memref.cast %75 : memref<64xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vcmp_gt_1d_float(%cast_63, %cast_64, %cast_65) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          %76 = hivm.hir.pointer_cast(%c164640_i64) : memref<64xf32, #hivm.address_space<ub>>
          %77 = hivm.hir.pointer_cast(%c164896_i64) : memref<16xf32, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          %cast_66 = memref.cast %75 : memref<64xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_67 = memref.cast %collapse_shape_62 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_68 = memref.cast %76 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_69 = memref.cast %77 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vsel_vs_1d_float(%cast_66, %cast_67, %cst_3, %cast_68, %cast_69) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          %expand_shape = memref.expand_shape %76 [[0, 1]] output_shape [64, 1] : memref<64xf32, #hivm.address_space<ub>> into memref<64x1xf32, #hivm.address_space<ub>>
          %78 = hivm.hir.pointer_cast(%c0_i64) : memref<64x128xf32, #hivm.address_space<ub>>
          %79 = hivm.hir.pointer_cast(%c164896_i64) : memref<512xf32, #hivm.address_space<ub>>
          hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
          hivm.hir.pipe_barrier[<PIPE_V>]
          %cast_70 = memref.cast %72 : memref<64x128xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
          %cast_71 = memref.cast %expand_shape : memref<64x1xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
          %cast_72 = memref.cast %78 : memref<64x128xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
          %cast_73 = memref.cast %79 : memref<512xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vdiv_2d_float(%cast_70, %cast_71, %cast_72, %cast_73) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          %80 = hivm.hir.pointer_cast(%c164352_i64) : memref<64x1xi1, #hivm.address_space<ub>>
          %collapse_shape_74 = memref.collapse_shape %80 [[0, 1]] : memref<64x1xi1, #hivm.address_space<ub>> into memref<64xi1, #hivm.address_space<ub>>
          %cast_75 = memref.cast %collapse_shape_62 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_76 = memref.cast %collapse_shape : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_77 = memref.cast %collapse_shape_74 : memref<64xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vcmp_gt_1d_float(%cast_75, %cast_76, %cast_77) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          %81 = hivm.hir.pointer_cast(%c164608_i64) : memref<64x1xf16, #hivm.address_space<ub>>
          %collapse_shape_78 = memref.collapse_shape %81 [[0, 1]] : memref<64x1xf16, #hivm.address_space<ub>> into memref<64xf16, #hivm.address_space<ub>>
          %82 = hivm.hir.pointer_cast(%c164736_i64) : memref<48xf16, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          %cast_79 = memref.cast %collapse_shape_74 : memref<64xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_80 = memref.cast %collapse_shape_78 : memref<64xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_81 = memref.cast %82 : memref<48xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vcast_bool_to_half_1d_with_temp(%cast_79, %cast_80, %cast_81, %c4_i32) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
          %83 = hivm.hir.pointer_cast(%c164736_i64) : memref<64x128xf16, #hivm.address_space<ub>>
          %84 = hivm.hir.pointer_cast(%c181120_i64) : memref<1024xf16, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          %cast_82 = memref.cast %81 : memref<64x1xf16, #hivm.address_space<ub>> to memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
          %cast_83 = memref.cast %83 : memref<64x128xf16, #hivm.address_space<ub>> to memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
          %cast_84 = memref.cast %84 : memref<1024xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @broadcast_last_axis_align_2d_half(%cast_82, %cast_83, %cast_84) : (memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          %85 = hivm.hir.pointer_cast(%c164736_i64) : memref<64x128xi1, #hivm.address_space<ub>>
          %collapse_shape_85 = memref.collapse_shape %83 [[0, 1]] : memref<64x128xf16, #hivm.address_space<ub>> into memref<8192xf16, #hivm.address_space<ub>>
          %collapse_shape_86 = memref.collapse_shape %85 [[0, 1]] : memref<64x128xi1, #hivm.address_space<ub>> into memref<8192xi1, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          %cast_87 = memref.cast %collapse_shape_85 : memref<8192xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_88 = memref.cast %collapse_shape_86 : memref<8192xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vcmps_eq_1d_half(%cast_87, %cst_1, %cast_88) : (memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, f16, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          %86 = hivm.hir.pointer_cast(%c164736_i64) : memref<64x128xi1, #hivm.address_space<ub>>
          %collapse_shape_89 = memref.collapse_shape %86 [[0, 1]] : memref<64x128xi1, #hivm.address_space<ub>> into memref<8192xi1, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          %87 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi1, #hivm.address_space<ub>>
          %cast_90 = memref.cast %collapse_shape_86 : memref<8192xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_91 = memref.cast %collapse_shape_89 : memref<8192xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_92 = memref.cast %87 : memref<0xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vnot_1d_bool(%cast_90, %cast_91, %cast_92) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          %88 = hivm.hir.pointer_cast(%c32768_i64) : memref<64x128xf32, #hivm.address_space<ub>>
          %collapse_shape_93 = memref.collapse_shape %78 [[0, 1]] : memref<64x128xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
          %collapse_shape_94 = memref.collapse_shape %88 [[0, 1]] : memref<64x128xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
          %89 = hivm.hir.pointer_cast(%c164352_i64) : memref<16xf32, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          %cast_95 = memref.cast %collapse_shape_89 : memref<8192xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_96 = memref.cast %collapse_shape_93 : memref<8192xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_97 = memref.cast %collapse_shape_94 : memref<8192xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_98 = memref.cast %89 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vsel_vs_1d_float(%cast_95, %cast_96, %cst_5, %cast_97, %cast_98) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
          %reinterpret_cast_99 = memref.reinterpret_cast %arg6 to offset: [%51], sizes: [128, 128], strides: [8192, 1] : memref<?xbf16, #hivm.address_space<gm>> to memref<128x128xbf16, strided<[8192, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_100 = memref.subview %reinterpret_cast_99[%7, 0] [64, 128] [1, 1] {to_be_bubbled_slice} : memref<128x128xbf16, strided<[8192, 1], offset: ?>, #hivm.address_space<gm>> to memref<64x128xbf16, strided<[8192, 1], offset: ?>, #hivm.address_space<gm>>
          %90 = hivm.hir.pointer_cast(%c32768_i64) : memref<64x128xbf16, #hivm.address_space<ub>>
          %collapse_shape_101 = memref.collapse_shape %90 [[0, 1]] : memref<64x128xbf16, #hivm.address_space<ub>> into memref<8192xbf16, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          %cast_102 = memref.cast %collapse_shape_94 : memref<8192xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_103 = memref.cast %collapse_shape_101 : memref<8192xbf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vcast_float_to_bfloat16_t_1d_with_mode(%cast_102, %cast_103, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
          %cast_104 = memref.cast %90 : memref<64x128xbf16, #hivm.address_space<ub>> to memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
          %cast_105 = memref.cast %subview_100 : memref<64x128xbf16, strided<[8192, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
          func.call @store_ubuf_to_gm_2d_bfloat16_t(%cast_104, %cast_105, %c0_i32) : (memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
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
