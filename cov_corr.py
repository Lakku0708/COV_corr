import numpy as np
def compute_covariance_matrix(matrix):
    matrix = np.transpose(matrix)
    covariance_matrix = np.cov(matrix, rowvar=False, bias=True)  # rowvar=False -> features in columns
    return covariance_matrix
def compute_correlation_matrix(covariance_matrix):
    std_devs = np.sqrt(np.diag(covariance_matrix))
    correlation_matrix = covariance_matrix / np.outer(std_devs, std_devs)
    np.fill_diagonal(correlation_matrix, 1)
    return correlation_matrix
M, N = 3, 5

matrix = np.random.rand(M, N)
cov_matrix = compute_covariance_matrix(matrix)
print("Covariance Matrix:")
print(cov_matrix)
corr_matrix = compute_correlation_matrix(cov_matrix)
print("\nCorrelation Matrix:")
print(corr_matrix)
