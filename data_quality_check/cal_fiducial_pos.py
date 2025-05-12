
def cal_pos(index, n_ch, pos_range):
    return pos_range[0] + (index) * (pos_range[1] - pos_range[0]) / (n_ch)

def main():
    scifi_n_ch = 1536
    scifi_hor_ch = [300, 1336]
    scifi_ver_ch = [200, 1200]
    scfit_hor_limit_pos = [14.21, 53.86]
    scfit_ver_limit_pos = [-46.09, -6.99]

    DS_n_bar = 60
    DS_hor_bar = [10, 50]
    DS_ver_bar = [15, 50]#DS_ver_bar = [70-60, 105-60]
    DS_hor_limit_pos = [7.61, 67.58]
    DS_ver_limit_pos = [-61.98, 1.72]

    scifi_hor_pos = []
    scifi_ver_pos = []
    DS_hor_pos = []
    DS_ver_pos = []

    scifi_hor_pos = [cal_pos(ch, scifi_n_ch, scfit_hor_limit_pos) for ch in scifi_hor_ch]
    scifi_ver_pos = [cal_pos(ch, scifi_n_ch, scfit_ver_limit_pos) for ch in scifi_ver_ch]

    DS_hor_pos = [cal_pos(bar, DS_n_bar, DS_hor_limit_pos) for bar in DS_hor_bar]
    DS_ver_pos = [cal_pos(bar, DS_n_bar, DS_ver_limit_pos) for bar in DS_ver_bar]

    print(f"Scifi Horizontal Positions: {scifi_hor_pos}")
    print(f"Scifi Vertical Positions: {scifi_ver_pos}")
    print(f"DS Horizontal Positions: {DS_hor_pos}")
    print(f"DS Vertical Positions: {DS_ver_pos}")


if __name__ == "__main__":
    main()