// -----// IR Dump After CVPipelining (cv-pipelining) ('func.func' operation: @_gqa_fwd_kernel) //----- //
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 20 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 20 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 40 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>, hacc.hivmc_compatible_print = false, hacc.hivmc_version = #hacc.hivmc_version<"0.2.0">, hivm.module_core_type = #hivm.module_core_type<MIX>, memref.memref_as_ptr} {
  func.func @_gqa_fwd_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix", parallel_mode = "simd"} {
    %c256 = arith.constant 256 : index
    %c128 = arith.constant 128 : index
    %c1_i32 = arith.constant 1 : i32
    %true = arith.constant true
    %cst = arith.constant -1.000000e+00 : f32
    %cst_0 = arith.constant 2.000000e+00 : f32
    %c8192 = arith.constant 8192 : index
    %cst_1 = arith.constant 0.0883883461 : f32
    %c512_i32 = arith.constant 512 : i32
    %c20_i32 = arith.constant 20 : i32
    %c16_i32 = arith.constant 16 : i32
    %c8_i32 = arith.constant 8 : i32
    %c262144_i32 = arith.constant 262144 : i32
    %c128_i32 = arith.constant 128 : i32
    %c2_i32 = arith.constant 2 : i32
    %c2097152_i32 = arith.constant 2097152 : i32
    hivm.hir.set_mask_norm
    %0 = arith.muli %arg7, %arg8 : i32
    %1 = arith.muli %0, %arg9 : i32
    annotation.mark %1 {logical_block_num} : i32
    %2 = hivm.hir.get_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = arith.muli %arg9, %arg8 : i32
    %5 = arith.divsi %3, %4 : i32
    %6 = arith.remsi %5, %arg7 : i32
    %7 = tensor.empty() : tensor<128x256xf32>
    %8 = tensor.empty() : tensor<128x128xf32>
    %9 = arith.muli %6, %c512_i32 : i32
    %10 = arith.divsi %9, %c20_i32 : i32
    %11 = arith.addi %6, %c1_i32 : i32
    %12 = arith.muli %11, %c512_i32 : i32
    %13 = arith.divsi %12, %c20_i32 : i32
    %14 = arith.cmpi sgt, %13, %10 : i32
    scf.if %14 {
      %15 = arith.divsi %10, %c16_i32 : i32
      %16 = arith.subi %13, %c1_i32 : i32
      %17 = arith.divsi %16, %c16_i32 : i32
      %18 = arith.addi %17, %c1_i32 : i32
      scf.for %arg10 = %15 to %18 step %c1_i32  : i32 {
        %19 = arith.divsi %arg10, %c8_i32 : i32
        %20 = arith.remsi %arg10, %c8_i32 : i32
        %21 = arith.muli %19, %c262144_i32 : i32
        %22 = arith.index_cast %21 : i32 to index
        %23 = arith.muli %20, %c128_i32 : i32
        %24 = arith.index_cast %23 : i32 to index
        %25 = arith.addi %22, %24 : index
        %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%25], sizes: [256, 128], strides: [1024, 1] : memref<?xbf16> to memref<256x128xbf16, strided<[1024, 1], offset: ?>>
        %alloc = memref.alloc() : memref<256x128xbf16>
        hivm.hir.load ins(%reinterpret_cast : memref<256x128xbf16, strided<[1024, 1], offset: ?>>) outs(%alloc : memref<256x128xbf16>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
        %26 = bufferization.to_tensor %alloc restrict writable : memref<256x128xbf16>
        %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [%25], sizes: [256, 128], strides: [1024, 1] : memref<?xbf16> to memref<256x128xbf16, strided<[1024, 1], offset: ?>>
        %alloc_3 = memref.alloc() : memref<256x128xbf16>
        hivm.hir.load ins(%reinterpret_cast_2 : memref<256x128xbf16, strided<[1024, 1], offset: ?>>) outs(%alloc_3 : memref<256x128xbf16>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
        %27 = bufferization.to_tensor %alloc_3 restrict writable : memref<256x128xbf16>
        %28 = arith.muli %arg10, %c16_i32 : i32
        %29 = arith.maxsi %10, %28 : i32
        %30 = arith.subi %29, %28 : i32
        %31 = arith.addi %arg10, %c1_i32 : i32
        %32 = arith.muli %31, %c16_i32 : i32
        %33 = arith.minsi %13, %32 : i32
        %34 = arith.subi %33, %28 : i32
        scf.for %arg11 = %30 to %34 step %c1_i32  : i32 {
          %35 = arith.divsi %arg11, %c2_i32 : i32
          %36 = arith.remsi %arg11, %c2_i32 : i32
          %37 = arith.muli %20, %c8_i32 : i32
          %38 = arith.addi %37, %35 : i32
          %39 = arith.muli %36, %c128_i32 : i32
          %40 = arith.muli %19, %c2097152_i32 : i32
          %41 = arith.index_cast %40 : i32 to index
          %42 = arith.muli %38, %c128_i32 : i32
          %43 = arith.index_cast %42 : i32 to index
          %44 = arith.addi %41, %43 : index
          %45 = arith.index_cast %39 : i32 to index
          %46 = arith.muli %45, %c8192 : index
          %47 = arith.addi %44, %46 : index
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%47], sizes: [128, 128], strides: [8192, 1] : memref<?xbf16> to memref<128x128xbf16, strided<[8192, 1], offset: ?>>
          %alloc_5 = memref.alloc() : memref<128x128xbf16>
          hivm.hir.load ins(%reinterpret_cast_4 : memref<128x128xbf16, strided<[8192, 1], offset: ?>>) outs(%alloc_5 : memref<128x128xbf16>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
          %48 = bufferization.to_tensor %alloc_5 restrict writable : memref<128x128xbf16>
          %49 = tensor.empty() : tensor<128x256xbf16>
          %50 = tensor.empty() : tensor<128x256xf32>
          %51 = hivm.hir.mmadL1 {b_transpose, fixpipe_already_inserted = true} ins(%48, %26, %true, %c128, %c128, %c256 : tensor<128x128xbf16>, tensor<256x128xbf16>, i1, index, index, index) outs(%50 : tensor<128x256xf32>) -> tensor<128x256xf32>
          %52 = memref_ext.alloc_workspace() from %arg2 : from memref<?xi8> to memref<128x256xf16>
          %53 = bufferization.to_tensor %52 restrict writable : memref<128x256xf16>
          %54 = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, pre_quant = #hivm.fixpipe_pre_quant_mode<F322F16>} ins(%51 : tensor<128x256xf32>) outs(%53 : tensor<128x256xf16>) -> tensor<128x256xf16>
          %55 = tensor.empty() : tensor<128x256xf16>
          %56 = hivm.hir.load ins(%54 : tensor<128x256xf16>) outs(%55 : tensor<128x256xf16>) init_out_buffer = false may_implicit_transpose_with_last_axis = false -> tensor<128x256xf16>
          %57 = hivm.hir.vcast ins(%56 : tensor<128x256xf16>) outs(%7 : tensor<128x256xf32>) -> tensor<128x256xf32>
          %58 = hivm.hir.vmul ins(%57, %cst_1 : tensor<128x256xf32>, f32) outs(%7 : tensor<128x256xf32>) -> tensor<128x256xf32>
          %59 = hivm.hir.vexp ins(%58 : tensor<128x256xf32>) outs(%7 : tensor<128x256xf32>) -> tensor<128x256xf32>
          %60 = hivm.hir.vcast ins(%59 : tensor<128x256xf32>) outs(%49 : tensor<128x256xbf16>) -> tensor<128x256xbf16>
          %61 = memref_ext.alloc_workspace() from %arg2 : from memref<?xi8> to memref<128x256xbf16>
          %62 = bufferization.to_tensor %61 restrict writable : memref<128x256xbf16>
          %63 = hivm.hir.store ins(%60 : tensor<128x256xbf16>) outs(%62 : tensor<128x256xbf16>) -> tensor<128x256xbf16>
          %64 = tensor.empty() : tensor<128x256xbf16>
          %65 = hivm.hir.load ins(%63 : tensor<128x256xbf16>) outs(%64 : tensor<128x256xbf16>) init_out_buffer = false may_implicit_transpose_with_last_axis = false -> tensor<128x256xbf16>
          %66 = tensor.empty() : tensor<128x128xf32>
          %67 = hivm.hir.mmadL1 {fixpipe_already_inserted = true} ins(%65, %27, %true, %c128, %c256, %c128 : tensor<128x256xbf16>, tensor<256x128xbf16>, i1, index, index, index) outs(%66 : tensor<128x128xf32>) -> tensor<128x128xf32>
          %68 = memref_ext.alloc_workspace() from %arg2 : from memref<?xi8> to memref<128x128xf16>
          %69 = bufferization.to_tensor %68 restrict writable : memref<128x128xf16>
          %70 = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, pre_quant = #hivm.fixpipe_pre_quant_mode<F322F16>} ins(%67 : tensor<128x128xf32>) outs(%69 : tensor<128x128xf16>) -> tensor<128x128xf16>
          %71 = tensor.empty() : tensor<128x128xf16>
          %72 = hivm.hir.load ins(%70 : tensor<128x128xf16>) outs(%71 : tensor<128x128xf16>) init_out_buffer = false may_implicit_transpose_with_last_axis = false -> tensor<128x128xf16>
          %73 = hivm.hir.vcast ins(%72 : tensor<128x128xf16>) outs(%8 : tensor<128x128xf32>) -> tensor<128x128xf32>
          %74 = tensor.empty() : tensor<128xf32>
          %75 = tensor.empty() : tensor<128x1xf32>
          %76 = hivm.hir.vreduce <sum> ins(%59 : tensor<128x256xf32>) outs(%75 : tensor<128x1xf32>) reduce_dims = [1] -> tensor<128x1xf32>
          %collapsed = tensor.collapse_shape %76 [[0, 1]] : tensor<128x1xf32> into tensor<128xf32>
          %77 = hivm.hir.vrec ins(%collapsed : tensor<128xf32>) outs(%74 : tensor<128xf32>) -> tensor<128xf32>
          %78 = hivm.hir.vmul ins(%collapsed, %77 : tensor<128xf32>, tensor<128xf32>) outs(%74 : tensor<128xf32>) -> tensor<128xf32>
          %79 = hivm.hir.vmul ins(%78, %cst : tensor<128xf32>, f32) outs(%74 : tensor<128xf32>) -> tensor<128xf32>
          %80 = hivm.hir.vadd ins(%79, %cst_0 : tensor<128xf32>, f32) outs(%74 : tensor<128xf32>) -> tensor<128xf32>
          %81 = hivm.hir.vmul ins(%77, %80 : tensor<128xf32>, tensor<128xf32>) outs(%74 : tensor<128xf32>) -> tensor<128xf32>
          %expanded = tensor.expand_shape %81 [[0, 1]] output_shape [128, 1] : tensor<128xf32> into tensor<128x1xf32>
          %82 = hivm.hir.vmul ins(%73, %expanded : tensor<128x128xf32>, tensor<128x1xf32>) outs(%8 : tensor<128x128xf32>) broadcast = [1] -> tensor<128x128xf32>
          %reinterpret_cast_6 = memref.reinterpret_cast %arg6 to offset: [%47], sizes: [128, 128], strides: [8192, 1] : memref<?xbf16> to memref<128x128xbf16, strided<[8192, 1], offset: ?>>
          %83 = tensor.empty() : tensor<128x128xbf16>
          %84 = hivm.hir.vcast ins(%82 : tensor<128x128xf32>) outs(%83 : tensor<128x128xbf16>) -> tensor<128x128xbf16>
          hivm.hir.store ins(%84 : tensor<128x128xbf16>) outs(%reinterpret_cast_6 : memref<128x128xbf16, strided<[8192, 1], offset: ?>>)
        }
      }
    }
    return
  }
}


