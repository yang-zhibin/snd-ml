import pandas as pd
import matplotlib.pyplot as plt

def plot_mu(df):
    # Create the plot with n_mu as the x-axis
    fig, ax = plt.subplots()

    # Plot both y1 and y2 data on the same y-axis
    ax.plot(df['n_mu'], df['has_veto_no_scifi1_ratio'], color='b', marker='o', label='Has Veto No SciFi 1 Ratio')
    ax.plot(df['n_mu'], df['has_veto_no_scifi2_ratio'], color='r', marker='x', label='Has Veto No SciFi 2 Ratio')
    ax.set_xlabel('n_mu')
    ax.set_ylabel('Has Veto No SciFi Ratios')
    ax.legend()

    # Adding a title
    plt.title('n_mu vs Has Veto No SciFi Ratios')

    # Save the plot as a PNG file
    plt.savefig('plots/n_mu_vs_ratios_plot.png')
    print(f"Plot saved successfully to 'plots/n_mu_vs_ratios_plot.png'")

def plot_scifi(df):
    # Create the plot with n_scifi as the x-axis
    fig, ax = plt.subplots()

    # Plot both y1 and y2 data on the same y-axis
    ax.plot(df['n_scifi'], df['has_veto_no_scifi1_ratio'], color='b', marker='o', label='Has Veto No SciFi 1 Ratio')
    ax.plot(df['n_scifi'], df['has_veto_no_scifi2_ratio'], color='r', marker='x', label='Has Veto No SciFi 2 Ratio')
    ax.set_xlabel('n_scifi')
    ax.set_ylabel('Has Veto No SciFi Ratios')
    ax.legend()

    # Adding a title
    plt.title('n_scifi vs Has Veto No SciFi Ratios')

    # Save the plot as a PNG file
    plt.savefig('plots/n_scifi_vs_ratios_plot.png')
    print(f"Plot saved successfully to 'plots/n_scifi_vs_ratios_plot.png'")
def main():
    mu_df = pd.read_csv('csv/scifi_ineff_loop_n_mu.csv')
    scifi_df = pd.read_csv('csv/scifi_ineff_loop_n_scifi.csv')

    plot_mu(mu_df)
    plot_scifi(scifi_df)

if __name__ == "__main__":
    main()

