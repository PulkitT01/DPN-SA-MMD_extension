from DPN_SA.Experiments import Experiments
from Graphs.Graphs import Graphs

# from DCN.Model25_10_25 import Model_25_1_25
# from DCN_PD_test import DCN_PD_Deep

if __name__ == '__main__':
    print("Using original data")
    Experiments().run_all_experiments(iterations=1, running_mode="ihdp")
    # Model_25_1_25().run_all_expeiments()
    # Graphs().draw_scatter_plots()
