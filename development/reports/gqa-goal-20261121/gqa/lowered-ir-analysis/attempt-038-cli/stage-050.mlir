module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 20 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 20 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 40 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>, hacc.hivmc_compatible_print = false, hacc.hivmc_version = #hacc.hivmc_version<"0.2.0">, memref.memref_as_ptr} {
  func.func @_gqa_fwd_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "mix", parallel_mode = "simd"} {
    %c256 = arith.constant 256 : index
    %c128 = arith.constant 128 : index
    %c1_i32 = arith.constant 1 : i32
    %true = arith.constant true
    %cst = arith.constant 0.000000e+00 : f16
    %cst_0 = arith.constant 0.693147182 : f32
    %c8192 = arith.constant 8192 : index
    %cst_1 = arith.constant 1.000000e+00 : f32
    %cst_2 = arith.constant 1.32223117 : f32
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
    %9 = tensor.empty() : tensor<128xf32>
    %10 = hivm.hir.vbrc ins(%cst_4 : f32) outs(%9 : tensor<128xf32>) -> tensor<128xf32>
    %11 = tensor.empty() : tensor<128x1xf32>
    %12 = hivm.hir.vbrc ins(%cst_4 : f32) outs(%11 : tensor<128x1xf32>) -> tensor<128x1xf32>
    %13 = arith.muli %6, %c512_i32 : i32
    %14 = arith.divsi %13, %c20_i32 : i32
    %15 = arith.addi %6, %c1_i32 : i32
    %16 = arith.muli %15, %c512_i32 : i32
    %17 = arith.divsi %16, %c20_i32 : i32
    %18 = arith.cmpi sgt, %17, %14 : i32
    scf.if %18 {
      %19 = arith.divsi %14, %c16_i32 : i32
      %20 = arith.subi %17, %c1_i32 : i32
      %21 = arith.divsi %20, %c16_i32 : i32
      %22 = arith.addi %21, %c1_i32 : i32
      scf.for %arg10 = %19 to %22 step %c1_i32  : i32 {
        %23 = arith.divsi %arg10, %c8_i32 : i32
        %24 = arith.remsi %arg10, %c8_i32 : i32
        %25 = arith.muli %23, %c262144_i32 : i32
        %26 = arith.index_cast %25 : i32 to index
        %27 = arith.muli %24, %c128_i32 : i32
        %28 = arith.index_cast %27 : i32 to index
        %29 = arith.addi %26, %28 : index
        %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%29], sizes: [256, 128], strides: [1024, 1] : memref<?xbf16> to memref<256x128xbf16, strided<[1024, 1], offset: ?>>
        %alloc = memref.alloc() : memref<256x128xbf16>
        hivm.hir.load ins(%reinterpret_cast : memref<256x128xbf16, strided<[1024, 1], offset: ?>>) outs(%alloc : memref<256x128xbf16>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
        %30 = bufferization.to_tensor %alloc restrict writable : memref<256x128xbf16>
        %reinterpret_cast_5 = memref.reinterpret_cast %arg5 to offset: [%29], sizes: [256, 128], strides: [1024, 1] : memref<?xbf16> to memref<256x128xbf16, strided<[1024, 1], offset: ?>>
        %alloc_6 = memref.alloc() : memref<256x128xbf16>
        hivm.hir.load ins(%reinterpret_cast_5 : memref<256x128xbf16, strided<[1024, 1], offset: ?>>) outs(%alloc_6 : memref<256x128xbf16>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
        %31 = bufferization.to_tensor %alloc_6 restrict writable : memref<256x128xbf16>
        %32 = tensor.empty() : tensor<256x128xf32>
        %33 = hivm.hir.vcast ins(%31 : tensor<256x128xbf16>) outs(%32 : tensor<256x128xf32>) -> tensor<256x128xf32>
        %34 = tensor.empty() : tensor<256x128xf16>
        %35 = hivm.hir.vcast ins(%33 : tensor<256x128xf32>) outs(%34 : tensor<256x128xf16>) -> tensor<256x128xf16>
        %36 = tensor.empty() : tensor<256x128xf16>
        %37 = hivm.hir.store ins(%35 : tensor<256x128xf16>) outs(%36 : tensor<256x128xf16>) -> tensor<256x128xf16>
        %38 = tensor.empty() : tensor<256x128xf16>
        %39 = hivm.hir.load ins(%37 : tensor<256x128xf16>) outs(%38 : tensor<256x128xf16>) init_out_buffer = false may_implicit_transpose_with_last_axis = false -> tensor<256x128xf16>
        %40 = arith.muli %arg10, %c16_i32 : i32
        %41 = arith.maxsi %14, %40 : i32
        %42 = arith.subi %41, %40 : i32
        %43 = arith.addi %arg10, %c1_i32 : i32
        %44 = arith.muli %43, %c16_i32 : i32
        %45 = arith.minsi %17, %44 : i32
        %46 = arith.subi %45, %40 : i32
        scf.for %arg11 = %42 to %46 step %c1_i32  : i32 {
          %47 = arith.divsi %arg11, %c2_i32 : i32
          %48 = arith.remsi %arg11, %c2_i32 : i32
          %49 = arith.muli %24, %c8_i32 : i32
          %50 = arith.addi %49, %47 : i32
          %51 = arith.muli %48, %c128_i32 : i32
          %52 = arith.muli %23, %c2097152_i32 : i32
          %53 = arith.index_cast %52 : i32 to index
          %54 = arith.muli %50, %c128_i32 : i32
          %55 = arith.index_cast %54 : i32 to index
          %56 = arith.addi %53, %55 : index
          %57 = arith.index_cast %51 : i32 to index
          %58 = arith.muli %57, %c8192 : index
          %59 = arith.addi %56, %58 : index
          %reinterpret_cast_7 = memref.reinterpret_cast %arg3 to offset: [%59], sizes: [128, 128], strides: [8192, 1] : memref<?xbf16> to memref<128x128xbf16, strided<[8192, 1], offset: ?>>
          %alloc_8 = memref.alloc() : memref<128x128xbf16>
          hivm.hir.load ins(%reinterpret_cast_7 : memref<128x128xbf16, strided<[8192, 1], offset: ?>>) outs(%alloc_8 : memref<128x128xbf16>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
          %60 = bufferization.to_tensor %alloc_8 restrict writable : memref<128x128xbf16>
          %61 = tensor.empty() : tensor<128x256xf32>
          %62 = hivm.hir.mmadL1 {b_transpose, fixpipe_already_inserted = true} ins(%60, %30, %true, %c128, %c128, %c256 : tensor<128x128xbf16>, tensor<256x128xbf16>, i1, index, index, index) outs(%61 : tensor<128x256xf32>) -> tensor<128x256xf32>
          %63 = tensor.empty() : tensor<128x256xf16>
          %64 = tensor.empty() : tensor<128x256xf16>
          %65 = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, pre_quant = #hivm.fixpipe_pre_quant_mode<F322F16>} ins(%62 : tensor<128x256xf32>) outs(%64 : tensor<128x256xf16>) -> tensor<128x256xf16>
          %66 = tensor.empty() : tensor<128x256xf16>
          %67 = hivm.hir.load ins(%65 : tensor<128x256xf16>) outs(%66 : tensor<128x256xf16>) init_out_buffer = false may_implicit_transpose_with_last_axis = false -> tensor<128x256xf16>
          %68 = hivm.hir.vcast ins(%67 : tensor<128x256xf16>) outs(%7 : tensor<128x256xf32>) -> tensor<128x256xf32>
          %69 = hivm.hir.vmul ins(%68, %cst_3 : tensor<128x256xf32>, f32) outs(%7 : tensor<128x256xf32>) -> tensor<128x256xf32>
          %70 = hivm.hir.vsub ins(%69, %cst_2 : tensor<128x256xf32>, f32) outs(%7 : tensor<128x256xf32>) -> tensor<128x256xf32>
          %71 = hivm.hir.vmul ins(%70, %cst_0 : tensor<128x256xf32>, f32) outs(%7 : tensor<128x256xf32>) -> tensor<128x256xf32>
          %72 = hivm.hir.vexp ins(%71 : tensor<128x256xf32>) outs(%7 : tensor<128x256xf32>) -> tensor<128x256xf32>
          %73 = hivm.hir.vcast ins(%72 : tensor<128x256xf32>) outs(%63 : tensor<128x256xf16>) -> tensor<128x256xf16>
          %74 = tensor.empty() : tensor<128x256xf16>
          %75 = hivm.hir.store ins(%73 : tensor<128x256xf16>) outs(%74 : tensor<128x256xf16>) -> tensor<128x256xf16>
          %76 = tensor.empty() : tensor<128x256xf16>
          %77 = hivm.hir.load ins(%75 : tensor<128x256xf16>) outs(%76 : tensor<128x256xf16>) init_out_buffer = false may_implicit_transpose_with_last_axis = false -> tensor<128x256xf16>
          %78 = tensor.empty() : tensor<128x128xf32>
          %79 = hivm.hir.mmadL1 {fixpipe_already_inserted = true} ins(%77, %39, %true, %c128, %c256, %c128 : tensor<128x256xf16>, tensor<256x128xf16>, i1, index, index, index) outs(%78 : tensor<128x128xf32>) -> tensor<128x128xf32>
          %80 = tensor.empty() : tensor<128x128xf32>
          %81 = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%79 : tensor<128x128xf32>) outs(%80 : tensor<128x128xf32>) -> tensor<128x128xf32>
          %82 = tensor.empty() : tensor<128x128xf32>
          %83 = hivm.hir.load ins(%81 : tensor<128x128xf32>) outs(%82 : tensor<128x128xf32>) init_out_buffer = false may_implicit_transpose_with_last_axis = false -> tensor<128x128xf32>
          %expanded = tensor.expand_shape %10 [[0, 1]] output_shape [128, 1] : tensor<128xf32> into tensor<128x1xf32>
          %84 = hivm.hir.vreduce <sum> ins(%72 : tensor<128x256xf32>) outs(%expanded : tensor<128x1xf32>) reduce_dims = [1] -> tensor<128x1xf32>
          %collapsed = tensor.collapse_shape %84 [[0, 1]] : tensor<128x1xf32> into tensor<128xf32>
          %85 = tensor.empty() : tensor<128xi1>
          %86 = hivm.hir.vcmp ins(%collapsed, %10 : tensor<128xf32>, tensor<128xf32>) outs(%85 : tensor<128xi1>) compare_mode = <gt> -> tensor<128xi1>
          %87 = hivm.hir.vsel ins(%86, %collapsed, %cst_1 : tensor<128xi1>, tensor<128xf32>, f32) outs(%9 : tensor<128xf32>) -> tensor<128xf32>
          %expanded_9 = tensor.expand_shape %87 [[0, 1]] output_shape [128, 1] : tensor<128xf32> into tensor<128x1xf32>
          %88 = hivm.hir.vbrc ins(%expanded_9 : tensor<128x1xf32>) outs(%8 : tensor<128x128xf32>) broadcast_dims = [1] -> tensor<128x128xf32>
          %89 = hivm.hir.vdiv ins(%83, %88 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%8 : tensor<128x128xf32>) -> tensor<128x128xf32>
          %90 = tensor.empty() : tensor<128x1xi1>
          %91 = hivm.hir.vcmp ins(%84, %12 : tensor<128x1xf32>, tensor<128x1xf32>) outs(%90 : tensor<128x1xi1>) compare_mode = <gt> -> tensor<128x1xi1>
          %92 = tensor.empty() : tensor<128x1xf16>
          %93 = hivm.hir.vcast ins(%91 : tensor<128x1xi1>) outs(%92 : tensor<128x1xf16>) round_mode = <trunc> -> tensor<128x1xf16>
          %94 = tensor.empty() : tensor<128x128xf16>
          %95 = hivm.hir.vbrc ins(%93 : tensor<128x1xf16>) outs(%94 : tensor<128x128xf16>) broadcast_dims = [1] -> tensor<128x128xf16>
          %96 = tensor.empty() : tensor<128x128xi1>
          %97 = hivm.hir.vcmp ins(%95, %cst : tensor<128x128xf16>, f16) outs(%96 : tensor<128x128xi1>) -> tensor<128x128xi1>
          %98 = hivm.hir.vnot ins(%97 : tensor<128x128xi1>) outs(%96 : tensor<128x128xi1>) -> tensor<128x128xi1>
          %99 = hivm.hir.vsel ins(%98, %89, %cst_4 : tensor<128x128xi1>, tensor<128x128xf32>, f32) outs(%8 : tensor<128x128xf32>) -> tensor<128x128xf32>
          %reinterpret_cast_10 = memref.reinterpret_cast %arg6 to offset: [%59], sizes: [128, 128], strides: [8192, 1] : memref<?xbf16> to memref<128x128xbf16, strided<[8192, 1], offset: ?>>
          %100 = tensor.empty() : tensor<128x128xbf16>
          %101 = hivm.hir.vcast ins(%99 : tensor<128x128xf32>) outs(%100 : tensor<128x128xbf16>) -> tensor<128x128xbf16>
          hivm.hir.store ins(%101 : tensor<128x128xbf16>) outs(%reinterpret_cast_10 : memref<128x128xbf16, strided<[8192, 1], offset: ?>>)
        }
      }
    }
    return
  }
}


