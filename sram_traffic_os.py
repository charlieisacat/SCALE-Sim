import math 
from tqdm import tqdm

neg_inf = -1 * math.pow(2,32)
# output stationary
def sram_traffic(
        dimension_rows=4,#行数 array h
        dimension_cols=4,#列数 array w
        ifmap_h=7, ifmap_w=7,
        filt_h=3, filt_w=3,
        num_channels=3,
        strides=1, num_filt=8,
        ofmap_base=2000000, filt_base=1000000, ifmap_base=0,
        sram_read_trace_file="sram_read.csv",
        sram_write_trace_file="sram_write.csv"
):


    # Dimensions of output feature map channel
    E_h = (ifmap_h - filt_h + strides) / strides
    E_w = (ifmap_w - filt_w + strides) / strides
    
    # Number of pixels in one convolution window
    px_per_conv_window = filt_h * filt_w * num_channels
    r2c = px_per_conv_window

    # Total number of ofmap px across all channels
    # num_filt 输出通道
    # num_ofmap_px 输出的元素个数
    num_ofmap_px = E_h * E_w * num_filt
    e2  = E_h * E_w
    e2m = num_ofmap_px
    
    # Variables to calculate folds in runtime
    # 单个输出通道的像素数 / 行数 
    # e2个元素, 每列每次处理dimension_rows个, 一共执行num_h_fold次
    num_h_fold = math.ceil(e2/dimension_rows)
    # num_filt个通道, 每行处理dimension_cols个, 一共执行的次数
    num_v_fold = math.ceil(num_filt/dimension_cols)

    cycles = 0

    read_cycles, util = gen_read_trace(
                            cycle = cycles,
                            dim_rows = dimension_rows,
                            dim_cols = dimension_cols,
                            num_v_fold = int(num_v_fold),
                            num_h_fold = int(num_h_fold),
                            ifmap_h = ifmap_h, ifmap_w= ifmap_w,
                            filt_h= filt_h, filt_w= filt_w,
                            num_channels= num_channels, stride=strides,
                            ofmap_h= int(E_h), ofmap_w= int(E_w), num_filters = num_filt,
                            filt_base= filt_base, ifmap_base= ifmap_base,
                            sram_read_trace_file= sram_read_trace_file
                            )

    write_cycles = gen_write_trace(
                        cycle = cycles,
                        dim_rows = dimension_rows,
                        dim_cols = dimension_cols,
                        #num_v_fold = int(num_v_fold),
                        #num_h_fold = int(num_h_fold),
                        ofmap_h = int(E_h), ofmap_w = int(E_w),
                        num_filters = num_filt,
                        ofmap_base = ofmap_base,
                        conv_window_size = r2c,
                        sram_write_trace_file = sram_write_trace_file
                        )

    cycles = max(read_cycles, write_cycles)
    str_cycles = str(cycles)
    return(str_cycles, util)
# End of sram_traffic()

        
def gen_read_trace(
        cycle = 0,
        dim_rows = 4, 
        dim_cols = 4,
        num_v_fold = 1,
        num_h_fold = 1,
        ifmap_h = 7, ifmap_w = 7,
        filt_h = 3, filt_w =3,
        num_channels = 3, stride = 1,
        ofmap_h =5, ofmap_w = 5, num_filters = 8, 
        filt_base = 1000000, ifmap_base = 0,
        sram_read_trace_file = "sram_read.csv",
        #sram_write_trace_file = "sram_write.csv"
):
    # Layer specific variables
    r2c = filt_h * filt_w * num_channels
    rc = filt_w * num_channels
    hc = ifmap_w * num_channels
    e2 = ofmap_h * ofmap_w
    #num_ofmap_px = e2 * num_filters
    
    # Tracking variables
    local_cycle     = 0
    #remaining_px    = e2           # Need tracking for individual v folds
    #remaining_px     = []
    remaining_filt  = num_filters
    ifmap_done      = False
    filt_done       = False
    row_base_addr   = []
    row_clk_offset  = []
    row_ofmap_idx   = []
    v_fold_row      = []
    col_base_addr   = []
    col_clk_offset  = []
    v_fold_col      = []
    h_fold_col      = []
    lane_done       = []
    v_fold_barrier  = []

    # Variables for utilization calculation
    rows_used = 0
    cols_used = 0
    util      = 0

    # This initialization assumes num_rows << num_ofmap_px
    # The assignment logic needs to be modified if that is not the case
    
    # dim_rows 阵列行数
    for r in range(dim_rows):
        print('==== r', r, 'ofmap_w', ofmap_w, math.floor(r / ofmap_w))
        #找到每个ofmap像素对应的ifmap首个px的位置
        base_row_id = math.floor(r / ofmap_w) * stride
        base_col_id = r % ofmap_w * stride
        # hc = ifmap_w * num_channels
        base_addr  = base_row_id * hc + base_col_id * num_channels 

        if r < e2:
            clk_offset = r * -1             # Clock offset takes care of the skew due to store and forward
        else:
            clk_offset = neg_inf            # In case num_ofamp_px < dim_rows

        row_base_addr.append(base_addr)
        row_clk_offset.append(clk_offset)
        row_ofmap_idx.append(r)
        v_fold_row.append(0)
        v_fold_barrier.append(False)

    for c in range(dim_cols):
        # r2c = filt_h * filt_w * num_channels
        # 因为每一列计算不同的output channel
        # 所以每个filter需要在一个col内用完
        base_addr = c * r2c

        # Anand: TODO
        if c < remaining_filt: #输出通道 =num_filters
            clk_offset = c * -1
            lane_done.append(False)
        else:
            clk_offset = neg_inf
            lane_done.append(True)

        col_base_addr.append(base_addr)
        col_clk_offset.append(clk_offset)
        v_fold_col.append(0)
        h_fold_col.append(0)


    # Open tracefile for writing
    outfile     = open(sram_read_trace_file, 'w')
    #ofmap_out   = open(sram_write_trace_file, 'w')

    # Adding progress bar
    # 因为是os, 在使用array(W*H)计算时，W*H个px同时算出来，
    # 总共需要计算num_v_fold次, 每次的时间是e2(e2个元素在一列中计算)
    tot  = e2 * num_v_fold
    #print("Total = " + str(tot))
    pbar = tqdm(total=tot)

    print('row_clk_offset ====', row_clk_offset)

    # Generate traces here
    # The condition checks
    #       1)  if the all the input traces for last v fold is generated
    # and   2)  if all the filter traces have been generated
    #while(remaining_px[num_v_fold-1] > 0) or (filt_done == False):
    while(ifmap_done == False) or (filt_done == False):
        ifmap_read = ""
        filt_read  = ""
        rows_used = 0
        cols_used = 0
        
        # Generate address for ifmap
        for r in range(dim_rows):

            if(row_clk_offset[r] >= 0):     # Take care of the skew

                inc = row_clk_offset[r]

                # rc = filt_w * num_channels
                # 在kernel范围内取数据, base_addr定位到kernel的起始位置
                addr_row_offset = math.floor(inc / rc) * ifmap_w * num_channels
                addr_col_offset = inc % rc #这个偏移只能看成是kernel范围内的偏移
                ifmap_addr = row_base_addr[r] + addr_row_offset + addr_col_offset 
                print('=======', 'r, ifmap_addr, row_base_addr[r], addr_row_offset, addr_col_offset, inc, rc, ifmap_w, num_channels, filter_w')
                print('=======', r, ifmap_addr, row_base_addr[r], addr_row_offset, addr_col_offset, inc, rc, ifmap_w, num_channels, filt_w)
                ifmap_read += str(int(ifmap_addr)) + ", "
                rows_used += 1
            else:
                ifmap_read += ", "

            # 初始时row_clk_offset除了[0]全是负的
            row_clk_offset[r] += 1

            if (row_clk_offset[r] > 0) and (row_clk_offset[r] % r2c == 0):   #Completed MAC for one OFMAP px
                
                # 算完一个px之后开始下一个，内存地址在dim_rows之后
                # A1 B1 C1; A2 B2 C2
                # 第一个地址为A1,下一个地址为A2，所以要+3
                row_ofmap_idx[r] += dim_rows
                print('after === row_ofmap_idx[%d]='%r,row_ofmap_idx[r], dim_rows)
                ofmap_idx = row_ofmap_idx[r]

                # Update progress bar
                pbar.update(1)

                if ofmap_idx < e2: #还在计算同一个output channel内的px
                    row_clk_offset[r] = 0
                    base_row_id = math.floor(ofmap_idx / ofmap_w) * stride
                    base_col_id = ofmap_idx % ofmap_w * stride
                    print('base_row_id=%d'%base_row_id, 'base_col_id=%d'%base_col_id, 'stride=%d'%stride, 'ofmap_w=%d'%ofmap_w)
                    base_addr  = base_row_id * hc + base_col_id * num_channels
                    row_base_addr[r] = base_addr
                else: #ofmap_idx > e2说明当前channel内的所有像素已经计算完毕，开始计算另一个channel
                    v_fold_row[r] += 1
                    #pbar.update(e2)
                    
                    if(v_fold_row[r] < num_v_fold):
                        row_ofmap_idx[r]  = r

                        base_row_id = math.floor(r / ofmap_w) * stride
                        base_col_id = r % ofmap_w * stride
                        base_addr  = base_row_id * hc + base_col_id * num_channels
                        row_base_addr[r]  = base_addr

                        # Stall this col from proceeding until all the rows reach the v_fold boundary
                        if (r != 0) and ((v_fold_row[r] > v_fold_row[r-1]) or (v_fold_barrier[r-1] == True)):
                            row_clk_offset[r] = neg_inf
                            v_fold_barrier[r] = True
                        else:
                            # r=0时v_fold_barrier仍时false
                            # 但是v_fold_row[1]<v_fold_row[0]，所以r=1时会被barrier
                            row_clk_offset[r] = 0

                    else:
                        row_clk_offset[r] = neg_inf

        # Get out of the barrier one by one
        # IMPORTANT: The barrier insertion and recovery is in separate loops to ensure that
        #            in a given clock cycle insertion for all rows strictly happen before the release.
        #            The flag ensures only one col is released per cycle
        # Since indx 0 never enters the barrier, this should work fine
        flag = False
        for r in range(dim_rows):
            if v_fold_barrier[r] and flag==False:
                if (v_fold_row[r] == v_fold_row[r-1]) and (v_fold_barrier[r-1] == False):
                    v_fold_barrier[r] = False
                    flag = True
                    row_clk_offset[r] = row_clk_offset[r-1] -1

        # Check if all input traces are done
        ifmap_done = True
        for r in range(dim_rows):
            if row_clk_offset[r] > 0: #当全部被设置成neg_inf时结束
                ifmap_done = False

        # Generate address for filters
        for c in range(dim_cols):
            if(col_clk_offset[c] >= 0):     # Take care of the skew
                inc = col_clk_offset[c]
                
                filt_addr = col_base_addr[c] + inc + filt_base 
                filt_read += str(filt_addr) + ", "
                cols_used += 1
            else:
                filt_read += ", "

            col_clk_offset[c] += 1

            if(col_clk_offset[c] > 0) and (col_clk_offset[c] % r2c == 0):

                # Get the v fold this col is working on and check the status of input trace generation
                #rem_px = remaining_px[v_fold_col[c]]

                #Tracking on the basis of h_folds
                h_fold_col[c] += 1

                # Anand: Check if all the input traces are generated for the given v fold
                if (h_fold_col[c] < num_h_fold): #还在计算同一个通道
                    col_clk_offset[c] = 0
                else:
                    v_fold_col[c] += 1
                    filt_id = v_fold_col[c] * dim_cols + c

                    # All filters might not be active in the last fold
                    # Adding the filter ID check to ensure only valid cols are active
                    if(v_fold_col[c] < num_v_fold) and (filt_id < num_filters):
                        col_clk_offset[c] = 0
                        h_fold_col[c] = 0

                        base = filt_id * r2c
                        col_base_addr[c] = base

                    else:
                        col_clk_offset[c] = neg_inf
                        lane_done[c] = True

        # Check if all filter traces are generated
        filt_done = True
        for c in range(dim_cols):
            if lane_done[c] == False:
                filt_done = False

                                                
        # Write to trace file
        global_cycle = cycle + local_cycle
        entry = str(global_cycle) + ", " + ifmap_read + filt_read + "\n"
        outfile.write(entry)

        #当前cycle矩阵的利用率
        this_util = (rows_used * cols_used) / (dim_rows * dim_cols)
        util += this_util
        print('util=%d'%util, 'this_util=%d'%this_util, 'rows_used=%d'%rows_used, 'cols_used=%d'%cols_used)

        # Update tracking variables
        local_cycle += 1

    pbar.close()
    outfile.close()
    #ofmap_out.close()

    #平均利用率
    util_perc = (util / local_cycle) * 100

    return (local_cycle + cycle), util_perc
# End of gen_read_trace()


def gen_write_trace(
        cycle = 0,
        dim_rows = 4,
        dim_cols = 4,
        #num_v_fold = 1,
        #num_h_fold = 1,
        ofmap_h = 5, ofmap_w = 5,
        num_filters = 4,
        ofmap_base = 2000000,
        conv_window_size = 9,                      # The number of pixels in a convolution window
        sram_write_trace_file = "sram_write.csv"
):

    # Layer specific variables
    r2c = conv_window_size
    e2  = ofmap_h * ofmap_w

    # Tracking variables
    id_row = []             # List of OFMAP ID for each row
    id_col = []             # List of filter ID for each col
    base_addr_col =[]       # Starting address of each output channel
    remaining_px  = e2
    remaining_filt= num_filters
    active_row = min(dim_rows, e2)
    active_col = min(dim_cols, num_filters)
    local_cycle = 0
    sticky_flag = False     # This flag is in place to fix the OFMAP cycle shaving bug

    for r in range(active_row):
        id_row.append(r)

    for c in range(active_col):
        id_col.append(c)

        base_col = c
        base_addr_col.append(base_col)

    #Open the file for writing
    outfile = open(sram_write_trace_file,"w")

    #This is the cycle when all the OFMAP elements in the first col become available
    # r2c: 单个px被计算出来的时间 =T
    local_cycle = r2c + active_col - 1

    while (remaining_px > 0) or (remaining_filt > 0):

        active_row = min(dim_rows, remaining_px)

        for r in range(active_row):
            local_px = id_row[r]
            remaining_px -= 1
            id_row[r] += active_row     # Taking care of horizontal fold

            ofmap_trace = ""
            for c in range(active_col):
                addr = ofmap_base + base_addr_col[c] + local_px * num_filters # r=0,c=0与r=1,c=0之间差了1*num_filters个地址
                ofmap_trace += str(addr) + ", "

            # Write the generated traces to the file
            entry = str(local_cycle + r) + ", " + ofmap_trace + "\n"
            outfile.write(entry)

        # Take care of the vertical fold
        if remaining_px == 0:
            remaining_filt -= active_col

            # In case of vertical fold we have to track when the output of (0,0) is generated
            # Shifting back local cycles to capture the last OFMAP generation in (0,0) for this fold
            last_fold_cycle   = local_cycle + active_row #当前active_row的计算结束时间 
            local_cycle -= (active_row + active_col - 1)
            sticky_flag = True

            # There are more OFMAP channels to go
            if remaining_filt > 0:
                remaining_px = e2
                last_active_col = active_col
                active_col = min(remaining_filt, dim_cols)

                # Reassign col base addresses
                for c in range(active_col):
                    base_addr_col[c] += last_active_col

                active_row = min(dim_rows, remaining_px)
                # Reassign row base addresses
                for r in range(active_row):
                    id_row[r] = r

                local_cycle += r2c + active_col
                if local_cycle < last_fold_cycle:
                    local_cycle = last_fold_cycle


            else:   # Restore the local cycle to return to the main function
                local_cycle = last_fold_cycle
                #local_cycle += (active_row + active_col)
                #sticky_flag = False

        else:   # If this is not a vertical fold then it is business as usual
            local_cycle += max(r2c, active_row)

    outfile.close()

    #if sticky_flag:
    #    local_cycle += (active_row + active_col)
    #    sticky_flag = False

    return(local_cycle + cycle)
# End of gen_write_trace()


if __name__ == "__main__":
   sram_traffic(
       dimension_rows = 8,
       dimension_cols = 4,
       ifmap_h = 7, ifmap_w = 7,
       filt_h = 2, filt_w = 2,
       num_channels = 1, strides = 1,
       num_filt = 7
   )
