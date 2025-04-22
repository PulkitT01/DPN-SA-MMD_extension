from collections import OrderedDict

import numpy as np

from DPN_SA.mmd_DPN_SA_Deep import DPN_SA_Deep
from Utils.Utils import Utils
from Utils.dataloader import DataLoader


class Experiments:
    def run_all_experiments(self, iterations, running_mode):

        if running_mode == "ihdp":
            train_path = "Dataset/ihdp_npci_1-100.train.npz"
            test_path = "Dataset/ihdp_npci_1-100.test.npz"
        else:
            train_path = "Dataset/jobs_DW_bin.new.10.train.npz"
            test_path = "Dataset/jobs_DW_bin.new.10.test.npz"
        split_size = 0.8
        device = Utils.get_device()
        print(device)
        results_list = []

        train_parameters_SAE = {
            "epochs": 400,
            "lr": 0.001,
            "batch_size": 32,
            "shuffle": True,
            "sparsity_probability": 0.8,
            "weight_decay": 0.0003,
            "BETA": 0.1,
        }
        run_parameters = self.__get_run_parameters(running_mode)

        print(str(train_parameters_SAE))
        file1 = open(run_parameters["summary_file_name"], "a")
        file1.write(str(train_parameters_SAE))
        file1.write("\n")
        file1.write("\n")
        for iter_id in range(iterations):
            print("########### 400 epochs ###########")
            print("--" * 20)
            print("iter_id: {0}".format(iter_id))
            print("--" * 20)
            # load data for propensity network
            dL = DataLoader()

            np_covariates_X_train, np_covariates_X_test, np_covariates_Y_train, np_covariates_Y_test \
                = self.load_data(running_mode, dL, train_path, test_path, iter_id)

            dp_sa = DPN_SA_Deep()
            trained_models = dp_sa.train_eval_DCN(iter_id,
                                                  np_covariates_X_train,
                                                  np_covariates_Y_train,
                                                  dL, device, run_parameters,
                                                  is_synthetic=run_parameters["is_synthetic"])

            sparse_classifier = trained_models["sparse_classifier"]
            sae_classifier_stacked_all_layer_active = trained_models["sae_classifier_stacked_all_layer_active"]
            sae_classifier_stacked_cur_layer_active = trained_models["sae_classifier_stacked_cur_layer_active"]

            # test DCN network
            reply = dp_sa.test_DCN(iter_id,
                                   np_covariates_X_test,
                                   np_covariates_Y_test,
                                   dL,
                                   sparse_classifier,
                                   sae_classifier_stacked_all_layer_active,
                                   sae_classifier_stacked_cur_layer_active,
                                   device, run_parameters)

            # MSE_SAE_e2e = reply["MSE_SAE_e2e"]
            # MSE_SAE_stacked_all_layer_active = reply["MSE_SAE_stacked_all_layer_active"]
            # MSE_SAE_stacked_cur_layer_active = reply["MSE_SAE_stacked_cur_layer_active"]
            # MSE_NN = reply["MSE_NN"]
            # MSE_LR = reply["MSE_LR"]
            # MSE_LR_lasso = reply["MSE_LR_Lasso"]
            #
            # true_ATE_NN = reply["true_ATE_NN"]
            # true_ATE_SAE_e2e = reply["true_ATE_SAE_e2e"]
            # true_ATE_SAE_stacked_all_layer_active = reply["true_ATE_SAE_stacked_all_layer_active"]
            # true_ATE_SAE_stacked_cur_layer_active = reply["true_ATE_SAE_stacked_cur_layer_active"]
            # true_ATE_LR = reply["true_ATE_LR"]
            # true_ATE_LR_Lasso = reply["true_ATE_LR_Lasso"]
            #
            # predicted_ATE_NN = reply["predicted_ATE_NN"]
            # predicted_ATE_SAE_e2e = reply["predicted_ATE_SAE_e2e"]
            # predicted_ATE_SAE_stacked_all_layer_active = reply["predicted_ATE_SAE_stacked_all_layer_active"]
            # predicted_ATE_SAE_stacked_cur_layer_active = reply["predicted_ATE_SAE_stacked_cur_layer_active"]
            # predicted_ATE_LR = reply["predicted_ATE_LR"]
            # predicted_ATE_LR_Lasso = reply["predicted_ATE_LR_Lasso"]

            SAE_e2e_ate_pred = reply["SAE_e2e_ate_pred"]
            SAE_e2e_att_pred = reply["SAE_e2e_att_pred"]
            SAE_e2e_bias_att = reply["SAE_e2e_bias_att"]
            SAE_e2e_atc_pred = reply["SAE_e2e_atc_pred"]
            SAE_e2e_policy_value = reply["SAE_e2e_policy_value"]
            SAE_e2e_policy_risk = reply["SAE_e2e_policy_risk"]
            SAE_e2e_err_fact = reply["SAE_e2e_err_fact"]

            SAE_stacked_all_layer_active_ate_pred = reply["SAE_stacked_all_layer_active_ate_pred"]
            SAE_stacked_all_layer_active_att_pred = reply["SAE_stacked_all_layer_active_att_pred"]
            SAE_stacked_all_layer_active_bias_att = reply["SAE_stacked_all_layer_active_bias_att"]
            SAE_stacked_all_layer_active_atc_pred = reply["SAE_stacked_all_layer_active_atc_pred"]
            SAE_stacked_all_layer_active_policy_value = reply["SAE_stacked_all_layer_active_policy_value"]
            SAE_stacked_all_layer_active_policy_risk = reply["SAE_stacked_all_layer_active_policy_risk"]
            SAE_stacked_all_layer_active_err_fact = reply["SAE_stacked_all_layer_active_err_fact"]

            SAE_stacked_cur_layer_active_ate_pred = reply["SAE_stacked_cur_layer_active_ate_pred"]
            SAE_stacked_cur_layer_active_att_pred = reply["SAE_stacked_cur_layer_active_att_pred"]
            SAE_stacked_cur_layer_active_bias_att = reply["SAE_stacked_cur_layer_active_bias_att"]
            SAE_stacked_cur_layer_active_atc_pred = reply["SAE_stacked_cur_layer_active_atc_pred"]
            SAE_stacked_cur_layer_active_policy_value = reply["SAE_stacked_cur_layer_active_policy_value"]
            SAE_stacked_cur_layer_active_policy_risk = reply["SAE_stacked_cur_layer_active_policy_risk"]
            SAE_stacked_cur_layer_active_err_fact = reply["SAE_stacked_cur_layer_active_err_fact"]


            file1.write("Iter: {0}, "
                        "SAE_e2e_bias_att: {1},  "
                        "SAE_stacked_all_layer_active_bias_att: {2},  "
                        "SAE_stacked_cur_layer_active_bias_att: {3}, "
                        "SAE_e2e_policy_risk: {4},  "
                        "SAE_stacked_all_layer_active_policy_risk: {5}, "
                        "SAE_stacked_cur_layer_active_policy_risk: {6} "
                        

                        .format(iter_id, SAE_e2e_bias_att,
                                SAE_stacked_all_layer_active_bias_att,
                                SAE_stacked_cur_layer_active_bias_att,
                                SAE_e2e_policy_risk,
                                SAE_stacked_all_layer_active_policy_risk,
                                SAE_stacked_cur_layer_active_policy_risk))
            result_dict = OrderedDict()

            result_dict["SAE_e2e_ate_pred"] = SAE_e2e_ate_pred
            result_dict["SAE_e2e_att_pred"] = SAE_e2e_att_pred
            result_dict["SAE_e2e_bias_att"] = SAE_e2e_bias_att
            result_dict["SAE_e2e_atc_pred"] = SAE_e2e_atc_pred
            result_dict["SAE_e2e_policy_value"] = SAE_e2e_policy_value
            result_dict["SAE_e2e_policy_risk"] = SAE_e2e_policy_risk
            result_dict["SAE_e2e_err_fact"] = SAE_e2e_err_fact

            result_dict["SAE_stacked_all_layer_active_ate_pred"] = SAE_stacked_all_layer_active_ate_pred
            result_dict["SAE_stacked_all_layer_active_att_pred"] = SAE_stacked_all_layer_active_att_pred
            result_dict["SAE_stacked_all_layer_active_bias_att"] = SAE_stacked_all_layer_active_bias_att
            result_dict["SAE_stacked_all_layer_active_atc_pred"] = SAE_stacked_all_layer_active_atc_pred
            result_dict["SAE_stacked_all_layer_active_policy_value"] = SAE_stacked_all_layer_active_policy_value
            result_dict["SAE_stacked_all_layer_active_policy_risk"] = SAE_stacked_all_layer_active_policy_risk
            result_dict["SAE_stacked_all_layer_active_err_fact"] = SAE_stacked_all_layer_active_err_fact

            result_dict["SAE_stacked_cur_layer_active_ate_pred"] = SAE_stacked_cur_layer_active_ate_pred
            result_dict["SAE_stacked_cur_layer_active_att_pred"] = SAE_stacked_cur_layer_active_att_pred
            result_dict["SAE_stacked_cur_layer_active_bias_att"] = SAE_stacked_cur_layer_active_bias_att
            result_dict["SAE_stacked_cur_layer_active_atc_pred"] = SAE_stacked_cur_layer_active_atc_pred
            result_dict["SAE_stacked_cur_layer_active_policy_value"] = SAE_stacked_cur_layer_active_policy_value
            result_dict["SAE_stacked_cur_layer_active_policy_risk"] = SAE_stacked_cur_layer_active_policy_risk
            result_dict["SAE_stacked_cur_layer_active_err_fact"] = SAE_stacked_cur_layer_active_err_fact
            results_list.append(result_dict)

        bias_att_set_SAE_E2E = []
        policy_risk_set_SAE_E2E = []

        bias_att_set_SAE_stacked_all_layer_active = []
        policy_risk_set_SAE_stacked_all_layer_active = []

        bias_att_set_SAE_stacked_cur_layer = []
        policy_risk_set_SAE_stacked_cur_layer = []

        for result in results_list:
            bias_att_set_SAE_E2E.append(result["SAE_e2e_bias_att"])
            policy_risk_set_SAE_E2E.append(result["SAE_e2e_policy_risk"])

            bias_att_set_SAE_stacked_all_layer_active.append(result["SAE_stacked_all_layer_active_bias_att"])
            policy_risk_set_SAE_stacked_all_layer_active.append(result["SAE_stacked_all_layer_active_policy_risk"])

            bias_att_set_SAE_stacked_cur_layer.append(result["SAE_stacked_cur_layer_active_bias_att"])
            policy_risk_set_SAE_stacked_cur_layer.append(result["SAE_stacked_cur_layer_active_policy_risk"])

        
        bias_att_set_SAE_E2E_mean = np.mean(np.array(bias_att_set_SAE_E2E))
        bias_att_set_SAE_E2E_std = np.std(bias_att_set_SAE_E2E)
        policy_risk_set_SAE_E2E_mean = np.mean(np.array(policy_risk_set_SAE_E2E))
        policy_risk_set_SAE_E2E_std = np.std(policy_risk_set_SAE_E2E)

        print("Using SAE E2E, bias_att: {0}, SD: {1}".format(bias_att_set_SAE_E2E_mean, bias_att_set_SAE_E2E_std))
        print("Using SAE E2E, policy_risk: {0}, SD: {1}".format(policy_risk_set_SAE_E2E_mean,
                                                                policy_risk_set_SAE_E2E_std))
        print("\n-------------------------------------------------\n")

        bias_att_set_SAE_stacked_all_layer_active_mean = np.mean(np.array(bias_att_set_SAE_stacked_all_layer_active))
        bias_att_set_SAE_stacked_all_layer_active_std = np.std(bias_att_set_SAE_stacked_all_layer_active)
        policy_risk_set_SAE_stacked_all_layer_active_mean = np.mean(
            np.array(policy_risk_set_SAE_stacked_all_layer_active))
        policy_risk_set_SAE_stacked_all_layer_active_std = np.std(policy_risk_set_SAE_stacked_all_layer_active)

        print(
            "Using SAE stacked all layer active, bias_att: {0}, SD: {1}".format(
                bias_att_set_SAE_stacked_all_layer_active_mean,
                bias_att_set_SAE_stacked_all_layer_active_std))
        print("Using SAE stacked all layer active, policy_risk: {0}, SD: {1}".format(
            policy_risk_set_SAE_stacked_all_layer_active_mean,
            policy_risk_set_SAE_stacked_all_layer_active_std))
        print("\n-------------------------------------------------\n")

        bias_att_set_SAE_stacked_cur_layer_mean = np.mean(np.array(bias_att_set_SAE_stacked_cur_layer))
        bias_att_set_SAE_stacked_cur_layer_std = np.std(bias_att_set_SAE_stacked_cur_layer)
        policy_risk_set_SAE_stacked_cur_layer_mean = np.mean(np.array(policy_risk_set_SAE_stacked_cur_layer))
        policy_risk_set_SAE_stacked_cur_layer_std = np.std(policy_risk_set_SAE_stacked_cur_layer)

        print(
            "Using SAE stacked cur layer active, bias_att: {0}, SD: {1}".format(bias_att_set_SAE_stacked_cur_layer_mean,
                                                                                bias_att_set_SAE_stacked_cur_layer_std))
        print("Using SAE stacked cur layer active, policy_risk: {0}, SD: {1}".format(
            policy_risk_set_SAE_stacked_cur_layer_mean,
            policy_risk_set_SAE_stacked_cur_layer_std))

        print("\n-------------------------------------------------\n")
        print("--" * 20)

        file1.write("Using SAE E2E, bias att: {0}, SD: {1}".format(bias_att_set_SAE_E2E_mean, bias_att_set_SAE_E2E_std))
        file1.write("\nUsing SAE E2E, policy risk: {0}, SD: {1}".format(policy_risk_set_SAE_E2E_mean,
                                                                        policy_risk_set_SAE_E2E_std))
        file1.write("\n-------------------------------------------------\n")

        file1.write(
            "Using SAE stacked all layer active, bias att: {0}, SD: {1}"
                .format(bias_att_set_SAE_stacked_all_layer_active_mean,
                        bias_att_set_SAE_stacked_all_layer_active_std))
        file1.write("\nUsing SAE stacked all layer active,  policy risk: {0}, SD: {1}"
                    .format(policy_risk_set_SAE_stacked_all_layer_active_mean,
                            policy_risk_set_SAE_stacked_all_layer_active_std))
        file1.write("\n-------------------------------------------------\n")

        file1.write("Using SAE stacked cur layer active, bias att: {0}, SD: {1}"
                    .format(bias_att_set_SAE_stacked_cur_layer_mean,
                            bias_att_set_SAE_stacked_cur_layer_std))
        file1.write("\nUsing SAE stacked cur layer active, policy risk: {0}, SD: {1}"
                    .format(policy_risk_set_SAE_stacked_cur_layer_mean,
                            policy_risk_set_SAE_stacked_cur_layer_mean))
        file1.write("\n-------------------------------------------------\n")
        file1.write("\n##################################################")

        Utils.write_to_csv(run_parameters["consolidated_file_path"], results_list)

    @staticmethod
    def __get_run_parameters(running_mode):
        run_parameters = {}
        if running_mode == "jobs":
            run_parameters["input_nodes"] = 17
            run_parameters["running_mode"] = running_mode
            run_parameters["consolidated_file_path"] = "Results/Output/Results_consolidated_mmd.csv"

            # SAE
            run_parameters["sae_e2e_prop_file"] = "Results/Output/SAE_E2E_Prop_score_{0}_mmd.csv"
            run_parameters["sae_stacked_all_prop_file"] = "Results/Output/SAE_stacked_all_Prop_score_{0}_mmd.csv"
            run_parameters["sae_stacked_cur_prop_file"] = "Results/Output/SAE_stacked_cur_Prop_score_{0}_mmd.csv"

            run_parameters["sae_e2e_iter_file"] = "Results/Output/ITE/ITE_SAE_E2E_iter_{0}_mmd.csv"
            run_parameters["sae_stacked_all_iter_file"] = "Results/Output/ITE/ITE_SAE_stacked_all_iter_{0}_mmd.csv"
            run_parameters["sae_stacked_cur_iter_file"] = "Results/Output/ITE/ITE_SAE_stacked_cur_Prop_iter_{0}_mmd.csv"

            run_parameters["summary_file_name"] = "Results/Logs/summary_results_mmd.txt"
            run_parameters["is_synthetic"] = False

        elif running_mode == "ihdp":
            run_parameters["input_nodes"] = 25
            run_parameters["running_mode"] = running_mode
            run_parameters["consolidated_file_path"] = "Results/Output_IHDP/Results_consolidated.csv"
        
            run_parameters["nn_prop_file"] = "Results/Output_IHDP/NN_Prop_score_{0}.csv"
            run_parameters["nn_iter_file"] = "Results/Output_IHDP/ITE/ITE_NN_iter_{0}.csv"
        
            run_parameters["sae_e2e_prop_file"] = "Results/Output_IHDP/SAE_E2E_Prop_score_{0}.csv"
            run_parameters["sae_stacked_all_prop_file"] = "Results/Output_IHDP/SAE_stacked_all_Prop_score_{0}.csv"
            run_parameters["sae_stacked_cur_prop_file"] = "Results/Output_IHDP/SAE_stacked_cur_Prop_score_{0}.csv"
        
            run_parameters["sae_e2e_iter_file"] = "Results/Output_IHDP/ITE/ITE_SAE_E2E_iter_{0}.csv"
            run_parameters["sae_stacked_all_iter_file"] = "Results/Output_IHDP/ITE/ITE_SAE_stacked_all_iter_{0}.csv"
            run_parameters["sae_stacked_cur_iter_file"] = "Results/Output_IHDP/ITE/ITE_SAE_stacked_cur_Prop_iter_{0}.csv"
        
            run_parameters["lr_prop_file"] = "Results/Output_IHDP/LR_Prop_score_{0}.csv"
            run_parameters["lr_iter_file"] = "Results/Output_IHDP/ITE/ITE_LR_iter_{0}.csv"
        
            run_parameters["lr_lasso_prop_file"] = "Results/Output_IHDP/LR_lasso_Prop_score_{0}.csv"
            run_parameters["lr_lasso_iter_file"] = "Results/Output_IHDP/ITE/ITE_LR_Lasso_iter_{0}.csv"
        
            run_parameters["summary_file_name"] = "Results/Logs/summary_results_ihdp.txt"
            run_parameters["is_synthetic"] = False
        
        elif running_mode == "synthetic_data":
            run_parameters["input_nodes"] = 225
            run_parameters["running_mode"] = running_mode
            run_parameters["consolidated_file_path"] = "Results/Output_Augmented/Results_consolidated_mmd.csv"

            
            # SAE
            run_parameters["sae_e2e_prop_file"] = "Results/Output_Augmented/SAE_E2E_Prop_score_{0}_mmd.csv"
            run_parameters["sae_stacked_all_prop_file"] = "Results/Output_Augmented/SAE_stacked_all_Prop_score_{0}_mmd.csv"
            run_parameters["sae_stacked_cur_prop_file"] = "Results/Output_Augmented/SAE_stacked_cur_Prop_score_{0}_mmd.csv"

            run_parameters["sae_e2e_iter_file"] = "Results/Output_Augmented/ITE/ITE_SAE_E2E_iter_{0}_mmd.csv"
            run_parameters["sae_stacked_all_iter_file"] = "Results/Output_Augmented/ITE/ITE_SAE_stacked_all_iter_{0}_mmd.csv"
            run_parameters["sae_stacked_cur_iter_file"] = "Results/Output_Augmented/ITE/ITE_SAE_stacked_cur_Prop_iter_{0}_mmd.csv"

            run_parameters["summary_file_name"] = "Results/Logs/summary_results_augmented_mmd.txt"
            run_parameters["is_synthetic"] = True

        return run_parameters

    @staticmethod
    def load_data(running_mode, dL, train_path, test_path, iter_id):
        
        if running_mode == "synthetic_data":
            return dL.preprocess_data_from_csv_augmented(train_path, test_path, iter_id)
    
        elif running_mode in ["ihdp", "jobs"]:
            return dL.preprocess_data_from_csv(train_path, test_path, iter_id)
