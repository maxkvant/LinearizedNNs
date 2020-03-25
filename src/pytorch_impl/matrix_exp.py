import torch


def matrix_exp(M, device, n_iter=3):
    M = M.double().to(device)

    n = M.size()[0]
    norm = torch.sqrt((M ** 2).sum())
    steps = 0
    while norm > 1e-8:
        M /= 2.
        norm /= 2.
        steps += 1

    series_sum = torch.eye(n).double().to(device)
    prod = M.double().to(device)
    for i in range(1, n_iter):
        series_sum = (series_sum + prod)
        prod = torch.matmul(prod, M) / i

    exp = series_sum
    for _ in range(steps):
        exp = torch.matmul(exp, exp)
    return exp.float()


# compute M^-1 * (exp(M) - E)
def compute_exp_term(M, device, n_iter=3):
    with torch.no_grad():
        M = M.double().to(device)

        n = M.size()[0]
        norm = torch.sqrt((M ** 2).sum())
        steps = 0
        while norm > 1e-8:
            M /= 2.
            norm /= 2.
            steps += 1

        series_sum = torch.zeros([n, n]).double().to(device)
        prod = torch.eye(n).double().to(device)

        # series_sum: E + M / 2 + M^2 / 6 + ...
        for i in range(1, n_iter):
            series_sum = (series_sum + prod)
            prod = torch.matmul(prod, M) / (i + 1)

        # (exp 0) (exp 0) = (exp^2           0)
        # (sum E) (sum E) = (sum * exp + sum E)
        exp = torch.matmul(M, series_sum) + torch.eye(n).to(device)
        for step in range(steps):
            series_sum = (torch.matmul(series_sum, exp) + series_sum) / 2.
            exp = torch.matmul(exp, exp)

        return series_sum.float()
