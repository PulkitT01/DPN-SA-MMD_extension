from DPN_SA.mmd_Experiments import Experiments
from Graphs.Graphs import Graphs

from DCN.Model25_10_25 import Model_25_1_25

if __name__ == '__main__':
    print("Using original data")
    Experiments().run_all_experiments(iterations=1, running_mode="ihdp")
    # Model_25_1_25().run_all_expeiments()
    # Graphs().draw_scatter_plots()
