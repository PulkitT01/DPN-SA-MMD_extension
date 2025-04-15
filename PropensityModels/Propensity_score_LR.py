from sklearn.linear_model import LogisticRegression
import numpy as np

class Propensity_socre_LR:
    @staticmethod
    def train(np_covariates_X_train, np_covariates_Y_train, regularized=False):
        # print(np_covariates_X_train.shape)
        # print(np_covariates_Y_train.shape)
        lr_model = None
        if not regularized:
            lr_model = LogisticRegression(solver='liblinear')
        elif regularized:
            lr_model = LogisticRegression(penalty="l1", solver="liblinear")

        # fit the model with data
        # lr_model.fit(np_covariates_X_train, np_covariates_Y_train.ravel())
        lr_model.fit(
            np_covariates_X_train.cpu().numpy(),  # Move to CPU and convert to NumPy
            np_covariates_Y_train.cpu().numpy().ravel()  # Move to CPU, convert, and flatten
        )

        #proba = lr_model.predict_proba(np_covariates_X_train)[:, -1].tolist()
        # Ensure np_covariates_X_train is on CPU and converted to NumPy
        proba = lr_model.predict_proba(np_covariates_X_train.cpu().numpy())[:, -1].tolist()

        return proba, lr_model

    @staticmethod
    def test(np_covariates_X_test, np_covariates_Y_test, log_reg):
        # print(np_covariates_X_train.shape)
        # print(np_covariates_Y_train.shape)

        # fit the model with data
        # proba = log_reg.predict_proba(np_covariates_X_test)[:, -1].tolist()
        proba = log_reg.predict_proba(np_covariates_X_test.cpu().numpy())[:, -1].tolist()
        return proba

    @staticmethod
    def train_graph(np_covariates_X_train, np_covariates_Y_train, regularized=False):
        print(np_covariates_X_train.shape)
        # print(np_covariates_Y_train.shape)
        lr_model = None
        if not regularized:
            lr_model = LogisticRegression(solver='liblinear')
        elif regularized:
            lr_model = LogisticRegression(penalty="l1", C=1.25, solver="liblinear")

        # fit the model with data
        # lr_model.fit(np_covariates_X_train, np_covariates_Y_train.ravel())
        # treatment = np_covariates_Y_train.ravel()
        lr_model.fit(
            np_covariates_X_train.cpu().numpy(),
            np_covariates_Y_train.cpu().numpy().ravel()
        )
        
        treatment = np_covariates_Y_train.cpu().numpy().ravel()
        print(treatment.shape)
        # treated_index = np.where(treatment == 1)[0]
        # control_index = np.where(treatment == 0)[0]
        # treated_X = np_covariates_X_train[treated_index]
        # control_X = np_covariates_X_train[control_index]
        treated_index = np.where(treatment == 1)[0]
        control_index = np.where(treatment == 0)[0]
        
        treated_X = np_covariates_X_train.cpu().numpy()[treated_index]
        control_X = np_covariates_X_train.cpu().numpy()[control_index]

        print(treated_X.shape)
        print(control_X.shape)
        # proba_treated = lr_model.predict_proba(treated_X)[:, -1].tolist()
        # proba_control = lr_model.predict_proba(control_X)[:, -1].tolist()
        proba_treated = lr_model.predict_proba(treated_X)[:, -1].tolist()
        proba_control = lr_model.predict_proba(control_X)[:, -1].tolist()
        return proba_treated, proba_control

