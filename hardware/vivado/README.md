# ZCU104 Vivado integration

Target toolchain: Vivado/Vitis 2023.2.

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

1. Generate the HLS project with `scripts/export_hls.py` (hls4ml
   `backend="Vitis"`) and synthesize it with Vitis HLS 2023.2. Vivado HLS was
   retired after the 2019.x line, so 2023.2 only ships Vitis HLS.
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

Standard Xilinx IP versions are pinned to values confirmed present in a real
Vivado 2023.2 IP catalog:

| IP | VLNV |
| --- | --- |
| `zynq_ultra_ps_e` | `xilinx.com:ip:zynq_ultra_ps_e:3.5` |
| `axi_dma` | `xilinx.com:ip:axi_dma:7.1` |
| `smartconnect` | `xilinx.com:ip:smartconnect:1.0` |
| `proc_sys_reset` | `xilinx.com:ip:proc_sys_reset:5.0` |

If a pinned version is missing from the installed catalog (e.g. on a later
Vivado release), the script falls back to resolving the highest available
version for that IP name and prints a warning, rather than failing outright.
