import numpy as np

def extract_realization(data, r):
    x = data["x"][:, :, r]                       # (672, 25)
    t = data["t"][:, r].reshape(-1, 1)           # (672, 1)
    yf = data["yf"][:, r].reshape(-1, 1)         # (672, 1)
    ycf = data["ycf"][:, r].reshape(-1, 1)       # (672, 1)
    mu0 = data["mu0"][:, r]                      # (672,)
    mu1 = data["mu1"][:, r]                      # (672,)
    ate = np.array([[np.mean(mu1 - mu0)]])       # (1, 1)

    return {
        "x": x,
        "t": t,
        "yf": yf,
        "ycf": ycf,
        "mu0": mu0,
        "mu1": mu1,
        "ate": ate
    }

def save_npz(data_dict, path):
    np.savez(path, **data_dict)
    print(f"Saved: {path}")

def convert_ihdp_train_test(realization=0):
    train_data = np.load("ihdp_npci_1-100.train.npz")
    test_data = np.load("ihdp_npci_1-100.test.npz")

    print(f"Extracting realization {realization} from IHDP dataset...")

    train_real = extract_realization(train_data, realization)
    test_real = extract_realization(test_data, realization)

    print(f"ATE (train): {train_real['ate'].item():.4f}")
    print(f"ATE (test):  {test_real['ate'].item():.4f}")

    save_npz(train_real, f"ihdp_jobs_style.train.npz")
    save_npz(test_real, f"ihdp_jobs_style.test.npz")

if __name__ == "__main__":
    convert_ihdp_train_test(realization=0)