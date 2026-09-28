# Create the ZCU104 ECG denoiser block design.
#
# Prerequisites:
#   1. Run scripts/export_hls.py and package the generated HLS IP.
#   2. Add the packaged IP repository to the Vivado project.
#   3. Set HLS_IP_VLNV to the packaged accelerator VLNV.
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

create_project $PROJECT_NAME $PROJECT_DIR -part $PART -force
set_property target_language Verilog [current_project]

create_bd_design $BD_NAME

create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:3.5 zynq_ultra_ps_e_0
set_property -dict [list \
    CONFIG.PSU__USE__M_AXI_GP0 {1} \
    CONFIG.PSU__USE__S_AXI_GP2 {0} \
    CONFIG.PSU__FPGA_PL0_ENABLE {1} \
] [get_bd_cells zynq_ultra_ps_e_0]

create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0
set_property -dict [list \
    CONFIG.c_include_sg {0} \
    CONFIG.c_sg_length_width {26} \
    CONFIG.c_addr_width {64} \
    CONFIG.c_m_axi_mm2s_data_width {128} \
    CONFIG.c_m_axi_s2mm_data_width {128} \
    CONFIG.c_m_axis_mm2s_tdata_width {16} \
    CONFIG.c_s_axis_s2mm_tdata_width {16} \
] [get_bd_cells axi_dma_0]

create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 axi_ctrl_interconnect
set_property CONFIG.NUM_SI {1} [get_bd_cells axi_ctrl_interconnect]
set_property CONFIG.NUM_MI {2} [get_bd_cells axi_ctrl_interconnect]

create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.1 proc_sys_reset_0
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
