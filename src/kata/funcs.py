import numpy as np
from numba import njit

# -------------- Kijima degradation model ---------------


@njit(cache=True, nogil=True, fastmath=True, inline="always")
def _dH(k, inv_lambda, age):
    """
    Exact integrated hazard over one unit step (dt=1):
      ΔH = ((age+1)/eta)^k - (age/eta)^k
    """
    return ((age + 1) * inv_lambda) ** k - (age * inv_lambda) ** k


@njit(cache=True, nogil=True, fastmath=True, inline="always")
def step_degrade(absolute_time, s_since_repair, virtual_age, k, inv_lambda):
    age = virtual_age + s_since_repair

    # Failure during the step ?
    dH = _dH(k, inv_lambda, age)
    p = 0.0 if dH <= 0.0 else (1.0 - np.exp(-dH))
    failed = np.random.random() < p

    # Accrue the step
    absolute_time += 1
    s_since_repair += 1

    return (1 if failed else 0), absolute_time, s_since_repair, virtual_age


def repair_kijima_type1(s_since_repair, virtual_age, alpha):
    if (
        alpha < 0
    ):  # Not mathematically sound, but user-defined to refer to a perfect repair (total replacement)
        s_since_repair = 0
        virtual_age = 0
    else:
        virtual_age += s_since_repair * alpha
        s_since_repair = 0
    return s_since_repair, virtual_age


# -------------- Production line model ---------------


@njit(cache=True, nogil=True, fastmath=True)
def step_prod_line(
    ### Mutables
    status: np.ndarray,  # (n_machines) status of the machines, int
    in_buff: np.ndarray,  # (n_machines) input buffer levels, int
    out_buff: np.ndarray,  # (n_machines) output buffer levels, int
    prod_completions: np.ndarray,  # (n_machines) production completions, int
    absolute_times: np.ndarray,  # (n_machines) absolute times, int
    s_since_repairs: np.ndarray,  # (n_machines) time since last repair, int
    virtual_ages: np.ndarray,  # (n_machines) virtual ages, int
    ### Immutables
    prod_rates: np.ndarray,  # (n_machines) production rates of the machines, int
    prod_costs: np.ndarray,  # (n_machines) production costs, int
    in_max_cap: np.ndarray,  # (n_machines) input buffer max capacities, int
    out_max_cap: np.ndarray,  # (n_machines) output buffer max capacities, int
    weibull_ks: np.ndarray,  # (n_machines) Weibull shape parameters, float
    weibull_inv_lambdas: np.ndarray,  # (n_machines) Weibull scale parameters, float
):
    """
    Summary :

    Args:
        param (type): description.

    Returns:
        type: description.

    Raises:
    """
    M = status.shape[0]  # number of machines

    # --- First phase, right-to-left buffer transfer
    for i in range(M - 2, -1, -1):
        if out_buff[i] > 0:
            free_next = in_max_cap[i + 1] - in_buff[i + 1]
            if free_next > 0:
                mv = out_buff[i] if out_buff[i] <= free_next else free_next
                in_buff[i + 1] += mv
                out_buff[i] -= mv

    # Second phase, produce on each machine
    # Semantics :
    # prod_completions < 0: no WIP, start new if in_buf>0
    # prod_completions = 0: finished-but-blocked, try to release to out_buff
    # prod_completions > 0: WIP, continue production

    for i in range(M):
        if status[i] == -1:
            # Machine is under maintenance, nothing progresses
            continue

        elif status[i] == 0:
            # CHeck if there's a product waiting for the out_buffer
            if prod_completions[i] == 0:
                if out_buff[i] < out_max_cap[i]:
                    out_buff[i] += 1
                    prod_completions[i] = -1
            # If the machine is idle, check the input buffer
            if in_buff[i] > 0:
                # Start new production if there's material
                prod_completions[i] = prod_costs[i]
                in_buff[i] -= 1
                # Update status to indicate the machine is now working
                status[i] = 1

        elif status[i] == 1:
            # If the machine is working, continue production
            if prod_completions[i] > prod_rates[i]:
                prod_completions[i] -= prod_rates[i]
            else:
                # Production finished, update status
                if out_buff[i] < out_max_cap[i]:
                    out_buff[i] += 1
                    prod_completions[i] = -1
                else:
                    prod_completions[i] = 0
                status[i] = 0

            # Simulate degradation
            failed, absolute_times[i], s_since_repairs[i], virtual_ages[i] = (
                step_degrade(
                    absolute_times[i],
                    s_since_repairs[i],
                    virtual_ages[i],
                    weibull_ks[i],
                    weibull_inv_lambdas[i],
                )
            )
            if failed:
                status[i] = -1

        return (
            status,
            prod_completions,
            in_buff,
            out_buff,
            absolute_times,
            s_since_repairs,
            virtual_ages,
        )
