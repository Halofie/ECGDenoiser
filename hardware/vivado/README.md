# ZCU104 Vivado integration

`create_design.tcl` is the starting point for the programmable-logic design.
It targets the ZCU104 XCZU7EV device and connects:

```text
Zynq UltraScale+ MPSoC
  -> AXI DMA
  -> ECG denoiser HLS IP
  -> AXI DMA
```

The generated HLS IP must expose the interface names documented at the top of
the Tcl file. If hls4ml uses different names, adapt only those interface
references after inspecting the packaged IP.

The intended Vivado sequence is:

1. Generate and synthesize the HLS project.
2. Package the HLS output as Vivado IP.
3. Create a Vivado project using the ZCU104 part.
4. Add the packaged IP repository.
5. Set `HLS_IP_VLNV`, `PROJECT_NAME`, and `PROJECT_DIR`.
6. Source `create_design.tcl`.
7. Validate the block design and resolve address assignment.
8. Run synthesis, implementation, and bitstream generation.
9. Copy the `.bit` and `.hwh` files to the PYNQ image.

Example Tcl variables:

```tcl
set HLS_IP_VLNV "user.org:hls:ecg_denoiser:1.0"
set PROJECT_NAME "ecg_denoiser_zcu104"
set PROJECT_DIR "C:/fpga/ecg_denoiser_zcu104"
source hardware/vivado/create_design.tcl
```

The exact HLS IP VLNV and stream port names cannot be finalized until HLS
conversion succeeds.
