from datetime import datetime

import numpy as np
import os
from DCN.DCN_network import DCN_network
from PropensityModels.Propensity_socre_network import Propensity_socre_network
from PropensityModels.mmd_Sparse_Propensity_score import Sparse_Propensity_score
from Utils.Utils import Utils


class DPN_SA_Deep:
    def train_eval_DCN(self, iter_id, np_covariates_X_train,
                       np_covariates_Y_train,
                       dL, device,
                       run_parameters,
                       is_synthetic=False):
        print("----------- Training and evaluation phase ------------")
        ps_train_set = dL.convert_to_tensor(np_covariates_X_train, np_covariates_Y_train)
        # diff = start - end
        # diff_minutes = divmod(diff.seconds, 60)
        # print('Time to train: ', diff_minutes[0], 'minutes',
        #       diff_minutes[1], 'seconds')

        # using SAE
        start = datetime.now()
        sparse_classifier, \
        sae_classifier_stacked_all_layer_active, sae_classifier_stacked_cur_layer_active = \
            self.__train_propensity_net_SAE(ps_train_set,
                                            np_covariates_X_train,
                                            np_covariates_Y_train,
                                            dL,
                                            iter_id, device,
                                            run_parameters["input_nodes"],
                                            is_synthetic)

        return {
            "sparse_classifier": sparse_classifier,
            "sae_classifier_stacked_all_layer_active": sae_classifier_stacked_all_layer_active,
            "sae_classifier_stacked_cur_layer_active": sae_classifier_stacked_cur_layer_active
        }

    def test_DCN(self, iter_id, np_covariates_X_test, np_covariates_Y_test, dL,
                 sparse_classifier,
                 sae_classifier_stacked_all_layer_active,
                 sae_classifier_stacked_cur_layer_active, device,
                 run_parameters):
        print("----------- Testing phase ------------")
        ps_test_set = dL.convert_to_tensor(np_covariates_X_test,
                                           np_covariates_Y_test)

        # using SAE
        epochs = 400  # Set dynamically if needed
        lr = 0.0001   # Set dynamically if needed
        os.makedirs("Results/Models", exist_ok=True)
        
        model_path_e2e = f"Results/Models/SAE_E2E_DCN_model_iter_id_{iter_id}_epoch_{epochs}_lr_{lr}_mmd.pth"
        model_path_stacked_all = f"Results/Models/SAE_stacked_all_DCN_model_iter_id_{iter_id}_epoch_{epochs}_lr_{lr}_mmd.pth"
        model_path_stacked_cur = f"Results/Models/SAE_stacked_cur_DCN_model_iter_id_{iter_id}_epoch_{epochs}_lr_{lr}_mmd.pth"


        propensity_score_save_path_e2e = run_parameters["sae_e2e_prop_file"]
        propensity_score_save_path_stacked_all = run_parameters["sae_stacked_all_prop_file"]
        propensity_score_save_path_stacked_cur = run_parameters["sae_stacked_cur_prop_file"]

        ITE_save_path_e2e = run_parameters["sae_e2e_iter_file"]
        ITE_save_path_stacked_all = run_parameters["sae_stacked_all_iter_file"]
        ITE_save_path_stacked_cur = run_parameters["sae_stacked_cur_iter_file"]

        # MSE_SAE_e2e = 0
        # true_ATE_SAE_e2e = 0
        # predicted_ATE_SAE_e2e = 0
        #
        # MSE_SAE_stacked_all_layer_active = 0
        # true_ATE_SAE_stacked_all_layer_active = 0
        # predicted_ATE_SAE_stacked_all_layer_active = 0
        #
        # MSE_SAE_stacked_cur_layer_active = 0
        # true_ATE_SAE_stacked_cur_layer_active = 0
        # predicted_ATE_SAE_stacked_cur_layer_active = 0
        #
        # MSE_LR = 0
        # true_ATE_LR = 0
        # predicted_ATE_LR = 0
        # MSE_LR_Lasso = 0
        # true_ATE_LR_Lasso = 0
        # predicted_ATE_LR_Lasso = 0

        print("############### DCN Testing using SAE E2E ###############")

        SAE_e2e_ate_pred, SAE_e2e_att_pred, SAE_e2e_bias_att, SAE_e2e_atc_pred, SAE_e2e_policy_value, \
        SAE_e2e_policy_risk, SAE_e2e_err_fact = \
            self.__test_DCN_SAE(iter_id, np_covariates_X_test,
                                np_covariates_Y_test, dL, device,
                                ps_test_set, sparse_classifier, model_path_e2e,
                                propensity_score_save_path_e2e, ITE_save_path_e2e,
                                run_parameters["is_synthetic"],
                                run_parameters["input_nodes"])

        print("############### DCN Testing using SAE Stacked all layer active ###############")
        SAE_stacked_all_layer_active_ate_pred, SAE_stacked_all_layer_active_att_pred, \
        SAE_stacked_all_layer_active_bias_att, SAE_stacked_all_layer_active_atc_pred, \
        SAE_stacked_all_layer_active_policy_value, \
        SAE_stacked_all_layer_active_policy_risk, SAE_stacked_all_layer_active_err_fact = \
            self.__test_DCN_SAE(iter_id, np_covariates_X_test,
                                np_covariates_Y_test, dL, device,
                                ps_test_set,
                                sae_classifier_stacked_all_layer_active, model_path_stacked_all,
                                propensity_score_save_path_stacked_all,
                                ITE_save_path_stacked_all,
                                run_parameters["is_synthetic"],
                                run_parameters["input_nodes"])

        print("############### DCN Testing using SAE cur layer active ###############")
        SAE_stacked_cur_layer_active_ate_pred, SAE_stacked_cur_layer_active_att_pred, \
        SAE_stacked_cur_layer_active_bias_att, SAE_stacked_cur_layer_active_atc_pred, \
        SAE_stacked_cur_layer_active_policy_value, \
        SAE_stacked_cur_layer_active_policy_risk, SAE_stacked_cur_layer_active_err_fact = \
            self.__test_DCN_SAE(iter_id, np_covariates_X_test,
                                np_covariates_Y_test, dL, device,
                                ps_test_set,
                                sae_classifier_stacked_cur_layer_active, model_path_stacked_cur,
                                propensity_score_save_path_stacked_cur,
                                ITE_save_path_stacked_cur,
                                run_parameters["is_synthetic"],
                                run_parameters["input_nodes"])

        
        return {
            "SAE_e2e_ate_pred": SAE_e2e_ate_pred,
            "SAE_e2e_att_pred": SAE_e2e_att_pred,
            "SAE_e2e_bias_att": SAE_e2e_bias_att,
            "SAE_e2e_atc_pred": SAE_e2e_atc_pred,
            "SAE_e2e_policy_value": SAE_e2e_policy_value,
            "SAE_e2e_policy_risk": SAE_e2e_policy_risk,
            "SAE_e2e_err_fact": SAE_e2e_policy_risk,

            "SAE_stacked_all_layer_active_ate_pred": SAE_stacked_all_layer_active_ate_pred,
            "SAE_stacked_all_layer_active_att_pred": SAE_stacked_all_layer_active_att_pred,
            "SAE_stacked_all_layer_active_bias_att": SAE_stacked_all_layer_active_bias_att,
            "SAE_stacked_all_layer_active_atc_pred": SAE_stacked_all_layer_active_bias_att,
            "SAE_stacked_all_layer_active_policy_value": SAE_stacked_all_layer_active_policy_value,
            "SAE_stacked_all_layer_active_policy_risk": SAE_stacked_all_layer_active_policy_risk,
            "SAE_stacked_all_layer_active_err_fact": SAE_stacked_all_layer_active_err_fact,

            "SAE_stacked_cur_layer_active_ate_pred": SAE_stacked_cur_layer_active_ate_pred,
            "SAE_stacked_cur_layer_active_att_pred": SAE_stacked_cur_layer_active_att_pred,
            "SAE_stacked_cur_layer_active_bias_att": SAE_stacked_cur_layer_active_bias_att,
            "SAE_stacked_cur_layer_active_atc_pred": SAE_stacked_cur_layer_active_atc_pred,
            "SAE_stacked_cur_layer_active_policy_value": SAE_stacked_cur_layer_active_policy_value,
            "SAE_stacked_cur_layer_active_policy_risk": SAE_stacked_cur_layer_active_policy_risk,
            "SAE_stacked_cur_layer_active_err_fact": SAE_stacked_cur_layer_active_err_fact

        }

    def __train_propensity_net_SAE(self,
                                   ps_train_set, np_covariates_X_train, np_covariates_Y_train,
                                   dL,
                                   iter_id, device, input_nodes, is_synthetic):
        # !!! best parameter list
        train_parameters_SAE = {
            "epochs": 400,
            "lr": 0.001,
            "batch_size": 32,
            "shuffle": True,
            "train_set": ps_train_set,
            "sparsity_probability": 0.8,
            "weight_decay": 0.0003,
            "BETA": 0.1,
            "input_nodes": input_nodes
        }

        # train_parameters_SAE = {
        #     "epochs": 2000,
        #     "lr": 0.001,
        #     "batch_size": 32,
        #     "shuffle": True,
        #     "train_set": ps_train_set,
        #     "sparsity_probability": 0.8,
        #     "weight_decay": 0.0003,
        #     "BETA": 0.1,
        #     "input_nodes": input_nodes
        # }

        print(str(train_parameters_SAE))
        ps_net_SAE = Sparse_Propensity_score()
        print("############### Propensity Score SAE net Training ###############")
        sparse_classifier, sae_classifier_stacked_all_layer_active, \
        sae_classifier_stacked_cur_layer_active = ps_net_SAE.train(train_parameters_SAE, device, phase="train")

        # eval propensity network using SAE
        epochs = 400  # Set dynamically if needed
        lr = 0.0001   # Set dynamically if needed
        os.makedirs("Results/Models", exist_ok=True)
        model_path_e2e = f"Results/Models/SAE_E2E_DCN_model_iter_id_{iter_id}_epoch_{epochs}_lr_{lr}_mmd.pth"
        model_path_stacked_all = f"Results/Models/SAE_stacked_all_DCN_model_iter_id_{iter_id}_epoch_{epochs}_lr_{lr}_mmd.pth"
        model_path_stacked_cur = f"Results/Models/SAE_stacked_cur_DCN_model_iter_id_{iter_id}_epoch_{epochs}_lr_{lr}_mmd.pth"
        print("---" * 25)
        print("End to End SAE training")
        print("---" * 25)

        start = datetime.now()
        self.__train_DCN_SAE(ps_net_SAE, ps_train_set, device, np_covariates_X_train,
                             np_covariates_Y_train,
                             iter_id, dL, sparse_classifier,
                             model_path_e2e, input_nodes, is_synthetic)
        end = datetime.now()
        print("SAE E2E start time: =", start)
        print("SAE E2E end time: =", end)
        # diff = start - end
        # diff_minutes = divmod(diff.seconds, 60)
        # print('Time to train: ', diff_minutes[0], 'minutes',
        #       diff_minutes[1], 'seconds')

        print("---" * 25)
        print("----------Layer wise greedy stacked SAE training - All layers----------")
        print("---" * 25)
        start = datetime.now()
        self.__train_DCN_SAE(ps_net_SAE, ps_train_set, device, np_covariates_X_train,
                             np_covariates_Y_train, iter_id, dL,
                             sae_classifier_stacked_all_layer_active,
                             model_path_stacked_all, input_nodes, is_synthetic)

        end = datetime.now()
        print("SAE all layer active start time: =", start)
        print("SAE all layer active end time: =", end)
        # diff = start - end
        # diff_minutes = divmod(diff.seconds, 60)
        # print('Time to train: ', diff_minutes[0], 'minutes',
        #       diff_minutes[1], 'seconds')

        print("---" * 25)
        print("Layer wise greedy stacked SAE training - Current layers")
        print("---" * 25)
        start = datetime.now()
        self.__train_DCN_SAE(ps_net_SAE, ps_train_set, device, np_covariates_X_train,
                             np_covariates_Y_train, iter_id, dL,
                             sae_classifier_stacked_cur_layer_active,
                             model_path_stacked_cur, input_nodes, is_synthetic)

        end = datetime.now()
        print("SAE cur layer active start time: =", start)
        print("SAE all layer active end time: =", end)
        # diff = start - end
        # diff_minutes = divmod(diff.seconds, 60)
        # print('Time to train: ', diff_minutes[0], 'minutes',
        #       diff_minutes[1], 'seconds')

        return sparse_classifier, sae_classifier_stacked_all_layer_active, sae_classifier_stacked_cur_layer_active

    def __train_DCN_SAE(self, ps_net_SAE, ps_train_set,
                        device, np_covariates_X_train,
                        np_covariates_Y_train,
                        iter_id, dL, sparse_classifier,
                        model_path,
                        input_nodes, is_synthetic):
        # eval propensity network using SAE
        ps_score_list_train_SAE = ps_net_SAE.eval(ps_train_set, device, phase="eval",
                                                  sparse_classifier=sparse_classifier)

        # load data for ITE network using SAE
        print("############### DCN Training using SAE ###############")
        data_loader_dict_train_SAE = dL.prepare_tensor_for_DCN(np_covariates_X_train,
                                                               np_covariates_Y_train,
                                                               ps_score_list_train_SAE,
                                                               is_synthetic)

        self.__train_DCN(data_loader_dict_train_SAE,
                         model_path, dL, device, input_nodes)

    
    def __train_DCN(self, data_loader_dict_train, model_path, dL, device, input_nodes):
        tensor_treated_train = self.create_tensors_from_tuple(data_loader_dict_train["treated_data"])
        tensor_control_train = self.create_tensors_from_tuple(data_loader_dict_train["control_data"])

        DCN_train_parameters = {
            "epochs": 100,
            "lr": 0.0001,
            "treated_batch_size": 1,
            "control_batch_size": 1,
            "shuffle": True,
            "treated_set_train": tensor_treated_train,
            "control_set_train": tensor_control_train,
            "model_save_path": model_path,
            "input_nodes": input_nodes
        }

        # train DCN network
        dcn = DCN_network()
        dcn.train(DCN_train_parameters, device)

    @staticmethod
    def create_tensors_from_tuple(group, test_set_flag=False):
        np_df_X = group[0]
        np_ps_score = group[1]
        np_df_Y_f = group[2]
        if test_set_flag:
            np_df_e = group[3]
            tensor = Utils.convert_to_tensor_DCN_test(np_df_X, np_ps_score,
                                                      np_df_Y_f, np_df_e)
        else:
            tensor = Utils.convert_to_tensor_DCN(np_df_X, np_ps_score,
                                                 np_df_Y_f)
        return tensor

    def __test_DCN_SAE(self, iter_id, np_covariates_X_test, np_covariates_Y_test, dL, device,
                       ps_test_set, sparse_classifier, model_path, propensity_score_csv_path,
                       iter_file, is_synthetic, input_nodes):
        # testing using SAE
        ps_net_SAE = Sparse_Propensity_score()
        ps_score_list_SAE = ps_net_SAE.eval(ps_test_set, device, phase="eval",
                                            sparse_classifier=sparse_classifier)
        Utils.write_to_csv(propensity_score_csv_path.format(iter_id), ps_score_list_SAE)

        # load data for ITE network using SAE
        data_loader_dict_SAE = dL.prepare_tensor_for_DCN(np_covariates_X_test,
                                                         np_covariates_Y_test,
                                                         ps_score_list_SAE,
                                                         is_synthetic)
        ate_pred, att_pred, bias_att, atc_pred, policy_value, \
        policy_risk, err_fact \
            = self.__do_test_DCN(data_loader_dict_SAE,
                                 dL, device,
                                 model_path,
                                 input_nodes,
                                 iter_file,
                                 iter_id)
        return ate_pred, att_pred, bias_att, atc_pred, policy_value, \
               policy_risk, err_fact


    def __do_test_DCN(self, data_loader_dict, dL, device, model_path, input_nodes, iter_file, iter_id):
        t_1 = np.ones(data_loader_dict["treated_data"][0].shape[0])

        t_0 = np.zeros(data_loader_dict["control_data"][0].shape[0])

        tensor_treated = \
            Utils.create_tensors_from_tuple_test(data_loader_dict["treated_data"], t_1)
        tensor_control = \
            Utils.create_tensors_from_tuple_test(data_loader_dict["control_data"], t_0)

        DCN_test_parameters = {
            "treated_set": tensor_treated,
            "control_set": tensor_control,
            "model_save_path": model_path
        }

        dcn = DCN_network()
        dcn_pd_models_eval_dict = dcn.eval(DCN_test_parameters, device, input_nodes)

        ate_pred, att_pred, bias_att, atc_pred, policy_value, \
        policy_risk, err_fact = \
            self.__process_evaluated_metric(
                dcn_pd_models_eval_dict["yf_list"],
                dcn_pd_models_eval_dict["e_list"],
                dcn_pd_models_eval_dict["T_list"],
                dcn_pd_models_eval_dict["y1_hat_list"],
                dcn_pd_models_eval_dict["y0_hat_list"],
                dcn_pd_models_eval_dict["ITE_dict_list"],
                dcn_pd_models_eval_dict["predicted_ITE"],
                iter_file,
                iter_id)

        return ate_pred, att_pred, bias_att, atc_pred, policy_value, \
               policy_risk, err_fact

    def __process_evaluated_metric(self, y_f, e, T,
                                   y1_hat, y0_hat,
                                   ite_dict, predicted_ITE_list,
                                   ite_csv_path,
                                   iter_id):
        y1_hat_np = np.array(y1_hat)
        y0_hat_np = np.array(y0_hat)
        e_np = np.array(e)
        t_np = np.array(T)
        np_y_f = np.array(y_f)

        y1_hat_np_b = 1.0 * (y1_hat_np > 0.5)
        y0_hat_np_b = 1.0 * (y0_hat_np > 0.5)

        #err_fact = np.mean(np.abs(y1_hat_np_b - np_y_f))
        #att = np.mean(np_y_f[t_np > 0]) - np.mean(np_y_f[(1 - t_np + e_np) > 1])
        t_np = np.array(T).flatten()
        e_np = np.array(e).flatten()
        np_y_f = np.array(y_f).flatten()
        
        # Ensure masks align with np_y_f
        mask1 = (t_np > 0)
        mask2 = ((1 - t_np + e_np) > 1)
        
        if mask1.shape != np_y_f.shape or mask2.shape != np_y_f.shape:
            raise ValueError(f"Shapes mismatch - np_y_f: {np_y_f.shape}, mask1: {mask1.shape}, mask2: {mask2.shape}")
        
        att = np.mean(np_y_f[mask1]) - np.mean(np_y_f[mask2])


        err_fact = np.mean(np.abs(y1_hat_np_b - np_y_f))


        eff_pred = y0_hat_np - y1_hat_np
        eff_pred[t_np > 0] = -eff_pred[t_np > 0]

        ate_pred = np.mean(eff_pred[e_np > 0])
        atc_pred = np.mean(eff_pred[(1 - t_np + e_np) > 1])

        att_pred = np.mean(eff_pred[(t_np + e_np) > 1])
        bias_att = np.abs(att_pred - att)

        policy_value = self.cal_policy_val(t_np[e_np > 0], np_y_f[e_np > 0],
                                           eff_pred[e_np > 0])

        print("bias_att: " + str(bias_att))
        print("policy_value: " + str(policy_value))
        print("Risk: " + str(1 - policy_value))
        print("atc_pred: " + str(atc_pred))
        print("att_pred: " + str(att_pred))
        print("err_fact: " + str(err_fact))

        Utils.write_to_csv(ite_csv_path.format(iter_id), ite_dict)
        return ate_pred, att_pred, bias_att, atc_pred, policy_value, 1 - policy_value, err_fact

    @staticmethod
    def cal_policy_val(t, yf, eff_pred):
        #  policy_val(t[e>0], yf[e>0], eff_pred[e>0], compute_policy_curve)

        if np.any(np.isnan(eff_pred)):
            return np.nan, np.nan

        policy = eff_pred > 0
        treat_overlap = (policy == t) * (t > 0)
        control_overlap = (policy == t) * (t < 1)

        if np.sum(treat_overlap) == 0:
            treat_value = 0
        else:
            treat_value = np.mean(yf[treat_overlap])

        if np.sum(control_overlap) == 0:
            control_value = 0
        else:
            control_value = np.mean(yf[control_overlap])

        pit = np.mean(policy)
        policy_value = pit * treat_value + (1 - pit) * control_value

        return policy_value
