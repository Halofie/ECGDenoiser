# Create the ZCU104 ECG denoiser block design.
#
# Target toolchain: Vivado 2023.2 (unified Vitis/Vivado installation).
#
# Prerequisites:
#   1. Run scripts/export_hls.py and synthesize the generated HLS project
#      with Vitis HLS 2023.2 (Vivado HLS was retired after the 2019.x line;
#      2023.2 only ships Vitis HLS).
#   2. Package the synthesized HLS output as Vivado IP.
#   3. Add the packaged IP repository to the Vivado project.
#   4. Set HLS_IP_VLNV to the packaged accelerator VLNV.
#
# The HLS IP must expose:
#   - AXI4-Stream slave named s_axis
#   - AXI4-Stream master named m_axis
#   - AXI4-Lite control interface named s_axi_control
#   - ap_clk and ap_rst_n

set required_vars {HLS_IP_VLNV PROJECT_NAME PROJECT_DIR}
foreach variable_name $required_vars {
    if {![info exists $variable_name] || [string equal [set $variable_name] ""]} {
        error "Set $variable_name before sourcing create_design.tcl"
    }
}

set PART "xczu7ev-ffvc1156-2-e"
set BD_NAME "ecg_denoiser"

# Standard Xilinx IP versions confirmed present in the Vivado 2023.2 IP
# catalog (via `get_ipdefs -filter {VLNV =~ "xilinx.com:ip:<name>:*"}`):
#   zynq_ultra_ps_e:3.5
#   axi_dma:7.1
#   smartconnect:1.0
#   proc_sys_reset:5.0
set ZYNQ_ULTRA_PS_E_VLNV   "xilinx.com:ip:zynq_ultra_ps_e:3.5"
set AXI_DMA_VLNV           "xilinx.com:ip:axi_dma:7.1"
set SMARTCONNECT_VLNV      "xilinx.com:ip:smartconnect:1.0"
set PROC_SYS_RESET_VLNV    "xilinx.com:ip:proc_sys_reset:5.0"

# Fallback for a Vivado install where the pinned versions above are not
# present in the IP catalog (e.g. a later release that removes an older
# minor version): resolve to the highest available version instead of
# failing outright.
proc resolve_ip_vlnv {vendor_lib_name pinned_vlnv} {
    if {[llength [get_ipdefs -filter "VLNV == \"${pinned_vlnv}\""]] > 0} {
        return $pinned_vlnv
    }
    puts "WARNING: pinned VLNV ${pinned_vlnv} not found in this IP catalog; \
resolving the highest available xilinx.com:ip:${vendor_lib_name}:* version instead."
    set matches [get_ipdefs -filter "VLNV =~ \"xilinx.com:ip:${vendor_lib_name}:*\""]
    if {[llength $matches] == 0} {
        error "No IP definitions found in the catalog for xilinx.com:ip:${vendor_lib_name}:*. \
Verify the IP is available in this Vivado installation."
    }
    set best ""
    set best_version {0 0}
    foreach ip_def $matches {
        set vlnv [get_property VLNV $ip_def]
        set version_str [lindex [split $vlnv ":"] 3]
        set version_parts [split $version_str "."]
        if {[llength $version_parts] < 2} {
            continue
        }
        set candidate [list [lindex $version_parts 0] [lindex $version_parts 1]]
        if {([lindex $candidate 0] > [lindex $best_version 0]) || \
            ([lindex $candidate 0] == [lindex $best_version 0] && \
             [lindex $candidate 1] > [lindex $best_version 1])} {
            set best_version $candidate
            set best $vlnv
        }
    }
    if {[string equal $best ""]} {
        error "Could not determine a version for xilinx.com:ip:${vendor_lib_name}"
    }
    puts "Resolved ${vendor_lib_name} -> $best"
    return $best
}

create_project $PROJECT_NAME $PROJECT_DIR -part $PART -force
set_property target_language Verilog [current_project]

create_bd_design $BD_NAME

create_bd_cell -type ip -vlnv [resolve_ip_vlnv zynq_ultra_ps_e $ZYNQ_ULTRA_PS_E_VLNV] zynq_ultra_ps_e_0
set_property -dict [list \
    CONFIG.PSU__USE__M_AXI_GP0 {1} \
    CONFIG.PSU__USE__S_AXI_GP2 {0} \
    CONFIG.PSU__FPGA_PL0_ENABLE {1} \
] [get_bd_cells zynq_ultra_ps_e_0]

create_bd_cell -type ip -vlnv [resolve_ip_vlnv axi_dma $AXI_DMA_VLNV] axi_dma_0
set_property -dict [list \
    CONFIG.c_include_sg {0} \
    CONFIG.c_sg_length_width {26} \
    CONFIG.c_addr_width {64} \
    CONFIG.c_m_axi_mm2s_data_width {128} \
    CONFIG.c_m_axi_s2mm_data_width {128} \
    CONFIG.c_m_axis_mm2s_tdata_width {16} \
    CONFIG.c_s_axis_s2mm_tdata_width {16} \
] [get_bd_cells axi_dma_0]

create_bd_cell -type ip -vlnv [resolve_ip_vlnv smartconnect $SMARTCONNECT_VLNV] axi_ctrl_interconnect
set_property CONFIG.NUM_SI {1} [get_bd_cells axi_ctrl_interconnect]
set_property CONFIG.NUM_MI {2} [get_bd_cells axi_ctrl_interconnect]

create_bd_cell -type ip -vlnv [resolve_ip_vlnv proc_sys_reset $PROC_SYS_RESET_VLNV] proc_sys_reset_0
create_bd_cell -type ip -vlnv $HLS_IP_VLNV ecg_denoiser_0

connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] \
    [get_bd_pins proc_sys_reset_0/slowest_sync_clk] \
    [get_bd_pins axi_dma_0/s_axi_lite_aclk] \
    [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] \
    [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] \
    [get_bd_pins axi_ctrl_interconnect/aclk] \
    [get_bd_pins ecg_denoiser_0/ap_clk]
connect_bd_net [get_bd_pins proc_sys_reset_0/peripheral_aresetn] \
    [get_bd_pins axi_dma_0/axi_resetn] \
    [get_bd_pins axi_ctrl_interconnect/aresetn] \
    [get_bd_pins ecg_denoiser_0/ap_rst_n]

connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S] \
    [get_bd_intf_pins ecg_denoiser_0/s_axis]
connect_bd_intf_net [get_bd_intf_pins ecg_denoiser_0/m_axis] \
    [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]

connect_bd_intf_net [get_bd_intf_pins zynq_ultra_ps_e_0/M_AXI_HPM0_FPD] \
    [get_bd_intf_pins axi_ctrl_interconnect/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_ctrl_interconnect/M00_AXI] \
    [get_bd_intf_pins axi_dma_0/S_AXI_LITE]
connect_bd_intf_net [get_bd_intf_pins axi_ctrl_interconnect/M01_AXI] \
    [get_bd_intf_pins ecg_denoiser_0/s_axi_control]
connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] \
    [get_bd_intf_pins zynq_ultra_ps_e_0/S_AXI_HP0_FPD]
connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] \
    [get_bd_intf_pins zynq_ultra_ps_e_0/S_AXI_HP1_FPD]

assign_bd_address
save_bd_design
validate_bd_design
make_wrapper -files [get_files $PROJECT_DIR/$PROJECT_NAME.srcs/sources_1/bd/$BD_NAME/$BD_NAME.bd] -top
update_compile_order -fileset sources_1
