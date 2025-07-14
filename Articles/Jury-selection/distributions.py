# Compare methods of removing a percentage of people:
# - Process 1. Select a number of indivdiual to remove, using a Normal distribution
# - Process 2. Select each individual given a probability for each selection

# www.solvermax.com

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd

def simulate_normal_distribution_selection(mean, stdev, num_individuals, num_simulations):
    selected_counts = []
    for _ in range(num_simulations):
        count = np.random.normal(loc=mean*num_individuals, scale=stdev*num_individuals)
        count = max(0, min(num_individuals, int(round(count))))
        selected_counts.append(count)
    return selected_counts

def simulate_probabilistic_selection(probability, num_individuals, num_simulations):
    selected_counts = []
    for _ in range(num_simulations):
        count = np.sum(np.random.rand(num_individuals) < probability)
        selected_counts.append(count)
    return selected_counts

def calculate_theoretical_distributions(mean, stdev, probability, num_individuals, num_simulations):
    normal_mean = mean * num_individuals
    normal_stdev = stdev * num_individuals
    binom_n = num_individuals
    binom_p = probability
    x = np.arange(0, num_individuals + 1)
    normal_pdf = stats.norm.pdf(x, loc=normal_mean, scale=normal_stdev) * num_simulations
    binom_pmf = stats.binom.pmf(x, n=binom_n, p=binom_p) * num_simulations
    return x, normal_pdf, binom_pmf

def plot_distributions(results_normal, results_probabilistic, x, normal_pdf, binom_pmf, num_simulations, num_individuals, plot_x_max, plot_y_max):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    hist_normal, bins_normal, _ = ax1.hist(results_normal, bins=np.arange(-0.5, num_individuals + 1.5, 1), alpha=0.5, color='#ff0000', label='Simulated', density=True)
    ax1.plot(x, normal_pdf / num_simulations, 'o', color='#ff0000', label='Theoretical')
    ax1.set_xlabel('Number of individuals selected')
    ax1.set_ylabel('Frequency')
    ax1.set_xlim(-0.5, plot_x_max)  # Set x-axis range to show whole bars
    ax1.set_ylim(0, plot_y_max)
    ax1.set_title('Normal distribution for number of individuals selected')
    ax1.legend(loc='upper right')

    hist_prob, bins_prob, _ = ax2.hist(results_probabilistic, bins=np.arange(-0.5, num_individuals + 1.5, 1), alpha=0.5, color='#0000ff', label='Simulated', density=True)
    ax2.plot(x, binom_pmf / num_simulations, 'o', color='#0000ff', label='Theoretical')
    ax2.set_xlabel('Number of individuals selected')
    ax2.set_ylabel('Frequency')
    ax2.set_xlim(-0.5, plot_x_max)  # Set x-axis range to show whole bars
    ax2.set_ylim(0, plot_y_max)
    ax2.set_title('Select each individual, given a probability')
    ax2.legend(loc='upper right')

    fig.subplots_adjust(hspace=0.3)
    plt.show()

    return hist_normal, bins_normal, normal_pdf, hist_prob, bins_prob, binom_pmf

def print_data(hist_normal, bins_normal, normal_pdf, hist_prob, bins_prob, binom_pmf, num_simulations):
    bins_normal_adj = np.arange(len(hist_normal))
    bins_prob_adj = np.arange(len(hist_prob))
    data_normal = pd.DataFrame({'Bins': bins_normal_adj, 'Simulated': hist_normal, 'Theoretical': normal_pdf / num_simulations}).round(4)
    data_probabilistic = pd.DataFrame({'Bins': bins_prob_adj, 'Simulated': hist_prob, 'Theoretical': binom_pmf / num_simulations}).round(4)
    print("Normal distribution data:")
    print(data_normal.to_string(index=False))
    print("\nProbabilistic distribution data:")
    print(data_probabilistic.to_string(index=False))
    print()
    
def main(scenario_name, num_individuals, num_simulations, mean, stdev, plot_x_max, plot_y_max):
    print(scenario_name, '\n')
    probability = mean
    results_normal = simulate_normal_distribution_selection(mean, stdev, num_individuals, num_simulations)
    results_probabilistic = simulate_probabilistic_selection(probability, num_individuals, num_simulations)
    x, normal_pdf, binom_pmf = calculate_theoretical_distributions(mean, stdev, probability, num_individuals, num_simulations)
    hist_normal, bins_normal, normal_pdf, hist_prob, bins_prob, binom_pmf = plot_distributions(results_normal, results_probabilistic, x, normal_pdf, binom_pmf, num_simulations, num_individuals, plot_x_max, plot_y_max)
    print_data(hist_normal, bins_normal, normal_pdf, hist_prob, bins_prob, binom_pmf, num_simulations)

if __name__ == "__main__":
    num_individuals = 20
    num_simulations = 1000000
    mean = 0.2
    stdev = 0.05
    plot_x_max = 12.5
    plot_y_max = 0.42
    scenario_name = f'Scenario: {num_individuals} individuals, {mean} mean, {stdev} stedev'
    main(scenario_name, num_individuals, num_simulations, mean, stdev, plot_x_max, plot_y_max)
    
    num_individuals = 54
    num_simulations = 1000000
    mean = 0.2
    stdev = 0.05
    plot_x_max = 22.5
    plot_y_max = 0.16
    scenario_name = f'Scenario: {num_individuals} individuals, {mean} mean, {stdev} stedev'
    main(scenario_name, num_individuals, num_simulations, mean, stdev, plot_x_max, plot_y_max)