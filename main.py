from preprocessgrid import extract_grid, plot_contour, plot_grid
from preprocessgrid import add_halo_cells, calculate_cell_face_areas
from preprocessgrid import calculate_cell_volumes

import numpy as np
import pandas as pd


def initialize_solver(inlet, volume, xsize:int, ysize:int):
    p = inlet['p']
    T = inlet['T']
    gamma = inlet['gamma']
    rho = inlet['rho']
    M = inlet['M']
    c = inlet['c']

    dV = volume['Vol']

    v_inlet = 0
    u_inlet = M * c
    et_inlet = p / (gamma - 1) / rho + 0.5 * (np.power(u_inlet,2) 
                                              + np.power(v_inlet,2))

    # vecQv = np.array([p, u_inlet, v_inlet, T])
    Q = np.array([1, u_inlet, v_inlet, et_inlet])

    # apply inlet conditions to inlet halo, then the rest as initial conditions
    nrows = ysize*xsize
    ncols = Q.size
    vecQ = np.empty(shape=(nrows,ncols))
    for iy in range(ysize):
        for ix in range(xsize): 
            idx = (iy*xsize) + ix
            vecQ[idx] = dV[idx] * rho * Q
    
    return vecQ, Q

def set_boundary_conditions(vecQ:np.array, inlet:dict,
                            lower_flow_bounds_L:list=[-0.5, 0.0],
                            lower_flow_bounds_R:list=[1.0, 1.5], 
                            airfoil_bounds:list=[0.0,1.0]):
    # since P and T are embedded inside rho, rho assumed
    # to stay constant for wall conditions:
    p = inlet['p']
    T = inlet['T']
    gamma = inlet['gamma']
    rho = inlet['rho']
    M = inlet['M']
    c = inlet['c']

    v_inlet = 0
    u_inlet = M * c
    et_inlet = p / (gamma - 1) / rho + 0.5 * (np.power(u_inlet,2) 
                                              + np.power(v_inlet,2))

    # set inviscid, adiabatic upper wall conditions
    idx_start = ysize * (xsize-1)
    idx_end = ysize * xsize
    for i in range(idx_start, idx_end):
        # TODO check if this is right
        # normal velocity component 
        vecQ[i][1] = u_inlet
        # tangent velocity component
        vecQ[i][2] = v_inlet 
        vecQ[i][3] = et_inlet

    # set lower flow boundary
    # TODO
    idx_start = lower_flow_bounds_L[0]
    idx_end = lower_flow_bounds_L[1]
    for i in range(idx_start, idx_end):
        # TODO check if this is right
        vecQ[i][1] = u_inlet
        vecQ[i][2] = v_inlet
        vecQ[i][3] = et_inlet

    # TODO
    idx_start = lower_flow_bounds_R[0]
    idx_end = lower_flow_bounds_R[1]
    for i in range(idx_start, idx_end):
        # TODO check if this is right
        vecQ[i][1] = u_inlet
        vecQ[i][2] = v_inlet
        vecQ[i][3] = et_inlet

    # TODO
    # set adiabatic, slip-wall airfoil wall conditions
    idx_start = airfoil_bounds[0]
    idx_end = airfoil_bounds[1]
    for i in range(idx_start, idx_end):
        # TODO check if this is right
        vecQ[i][1] = u_inlet * 1
        vecQ[i][2] = u_inlet * 1
        vecQ[i][3] = et_inlet

    return vecQ

def interp_MUSCL(Qs, e:float=0, k:float=-1):
    # represents Q_i-2
    Q_i_2 = Qs[0]
    # represents Q_i-1
    Q_i_1 = Qs[1]
    Q_i = Qs[2]
    # represents Q_i+1
    Q_i1 = Qs[3]

    Q_L_half = Q_i_1 + 0.25 * e * ((1-k)*(Q_i_1 - Q_i_2) + (1+k)*(Q_i - Q_i_1))
    Q_R_half = Q_i - 0.25 * e * ((1+k)*(Q_i - Q_i_1) + (1-k)*(Q_i1 - Q_i))

    return Q_L_half, Q_R_half

def calculate_E_fluxes(Q_L_half:np.array, Q:np.array, Q_R_half:np.array,
                        S_xi_x_L:float, S_xi_x_R:float, S_xi_y_L:float, 
                        S_xi_y_R, S_xi_L:float, S_xi_R:float, 
                       sigma:float=1.0, Kp:float=0.25, Ku:float=0.75,
                       beta:float=1/8, R:float=287.0, _k:float=0.0001,
                       Minf:float=2.0):
    # via AUSM+ scheme, evaluate at each cell face
    # NOTE Thes these are written in dimensional form
    # refer to Topic 24.1

    p = Q['p']
    T = Q['T']
    gamma = Q['gamma']
    rho = Q['rho']
    M = Q['M']
    c = Q['c']
    u = Q['u']
    v = Q['v']
    et = p / (gamma - 1) / rho + 0.5 * (np.power(u,2) + np.power(v,2))
    ht = et + p / rho

    # recall vecQv = np.array([p, u, v, T])
    P_L = Q_L_half[0]
    P_R = Q_R_half[0]

    u_L = Q_L_half[1]
    u_R = Q_R_half[1]

    v_L = Q_L_half[2]
    v_R = Q_R_half[2]

    T_L = Q_L_half[3]
    T_R = Q_R_half[3]

    rho_L = P_L/R/T_L
    rho_R = P_R/R/T_R

    et_L = P_L / (gamma - 1) / rho_L + 0.5 * (np.power(u_L,2) + np.power(v_L,2))
    et_R = P_R / (gamma - 1) / rho_R + 0.5 * (np.power(u_R,2) + np.power(v_R,2))

    ht_L = et_L + P_L / rho_L
    ht_R = et_R + P_R / rho_R

    #### E Flux Only ####
    U_xi_L = (u_L * S_xi_x_L + S_xi_y_L * v_L) / S_xi_L
    U_xi_R = (u_R * S_xi_x_R + S_xi_y_R * v_R) / S_xi_R

    ##### Calculate M_1/2 #####
    # assume calorically perfect gas 
    c_star = np.sqrt(2 * ht * (gamma - 1)/(gamma + 1))
    c_L = (c_star*c_star) / np.max([c_star, U_xi_L])
    c_R = (c_star*c_star) / np.max([c_star, -U_xi_R])
    c_half = np.min([c_L, c_R])

    M_L = U_xi_L / c_half
    M_R = U_xi_R / c_half
    M = 0.5 * (M_L * M_L + M_R * M_R)
    Mco = _k * Minf
    # from JCP 214 2006, eq. 14 approximation
    Mo = np.sqrt(np.min(1, np.max(M*M, Mco*Mco)))
    fa = 2 * Mo
    # fa = 1 # dissipation factor, fa=1 the basic AUSM scheme

    # M_half = U_xi_half / c  # c is speed of sound at cell face 

    rho_half = 0.5 * (rho_L + rho_R)
    M_P = -Kp * np.max([1 - sigma*M*M, 0]) *\
        (P_R - P_L)/(rho_half * c_half * c_half)
    
    M_1_plus = 0.5 * (M + np.abs(M))
    M_1_minus= 0.5 * (M - np.abs(M))

    M_2_plus = 0.25 * np.power(M + 1, 2)
    M_2_minus = -0.25 * np.power(M - 1, 2)

    M_plus_m = {True: M_1_plus, False: M_2_plus *\
                 (1 - 16 * beta * M_2_minus)}[M >= 1]
    M_minus_m = {True: M_1_minus, False: M_2_minus *\
                  (1 + 16 * beta * M_2_plus)}[M >= 1]

    M_half = M_plus_m * M_L + M_minus_m * M_R + M_P

    #### Calculate mass flow rate ###
    U_xi_half = c_half * M_half
    rho = {True : rho_L, False : rho_R}[U_xi_half > 0]
    m_dot_half = rho * c * M_half

    #### Calculate Pressure ####
    # must be either left or right side depending on conditon:
    # eps_L = np.array([1, u, v, ht])
    # eps = eps_R   # default is right side 
    # if m_dot > 0: eps = eps_L  # otherwise, switch to left side 

    alpha = (3/16) * (-4 + 5 * fa * fa)
    P_plus_5 = {True: (1/M)*M_1_plus,
                False: M_2_plus * (2 - M) - 16 *\
                      alpha * M * M_2_minus}[np.abs(M) >= 1] 
    P_minus_5 = {True: (1/M)*M_1_minus,
                 False: M_2_minus * (-2 - M) + 16 *\
                   alpha * M * M_2_plus}[np.abs(M) >= 1] 
    P_u = -Ku * P_plus_5 * P_minus_5 * (rho_L + rho_R) *\
          (fa * c_half) * (u_R - u_L)
    p_half = P_plus_5 * M_L * P_L +\
        P_minus_5 * M_R * P_R + P_u

    eps_L = np.array([1, u_L, v_L, ht_L])
    eps_R = np.array([1, u_R, v_R, ht_R])
    E_xi_half = 0.5 * m_dot_half * (eps_L + eps_R) -\
          0.5 * np.abs(m_dot_half) * (eps_L + eps_R) + p_half

    # p = np.array([0, (S_xi_x / S_xi)*p, (S_xi_y / S_xi)*p, 0])
    # E_xi = m_dot * eps + p
    # dissipation coefficient
    # D = M_half # mach number at i-1/2,j cell face (for AUSM)
    # Q_L_half, Q_R_half = interp_MUSCL()
    # E_xi_half = 0.5 * (E_xi * Q_L_half + E_xi * Q_R_half) - 0.5 * np.abs(D) * (Q_R_half - Q_L_half)
    return E_xi_half 

def calculate_F_fluxes(Q_L_half:np.array, Q:np.array, Q_R_half:np.array,
                       S_eta_x_L:float, S_eta_x_R:float, S_eta_y_L:float, 
                       S_eta_y_R, S_eta_L:float, S_eta_R:float,
                       sigma:float=1.0,
                       Kp:float=0.25, Ku:float=0.75, beta:float=1/8,
                       R:float=287.0, _k:float=0.0001, Minf:float=2.0):
    # via AUSM+ scheme, evaluate at each cell face
    # NOTE Thes these are written in dimensional form
    # refer to Topic 24.1

    p = Q['p']
    T = Q['T']
    gamma = Q['gamma']
    rho = Q['rho']
    M = Q['M']
    c = Q['c']
    u = Q['u']
    v = Q['v']
    et = p / (gamma - 1) / rho + 0.5 * (np.power(u,2) + np.power(v,2))
    ht = et + p / rho

    # recall vecQv = np.array([p, u, v, T])
    P_L = Q_L_half[0]
    P_R = Q_R_half[0]

    u_L = Q_L_half[1]
    u_R = Q_R_half[1]

    v_L = Q_L_half[2]
    v_R = Q_R_half[2]

    T_L = Q_L_half[3]
    T_R = Q_R_half[3]

    rho_L = P_L/R/T_L
    rho_R = P_R/R/T_R

    et_L = P_L / (gamma - 1) / rho_L + 0.5 * (np.power(u_L,2) + np.power(v_L,2))
    et_R = P_R / (gamma - 1) / rho_R + 0.5 * (np.power(u_R,2) + np.power(v_R,2))

    ht_L = et_L + P_L / rho_L
    ht_R = et_R + P_R / rho_R

    #### E Flux Only ####
    V_eta_L = (u_L * S_eta_x_L + S_eta_y_L * v_L) / S_eta_L
    V_eta_R = (u_R * S_eta_x_R + S_eta_y_R * v_R) / S_eta_R

    ##### Calculate M_1/2 #####
    # assume calorically perfect gas 
    c_star = np.sqrt(2 * ht * (gamma - 1)/(gamma + 1))
    c_L = (c_star*c_star) / np.max([c_star, V_eta_L])
    c_R = (c_star*c_star) / np.max([c_star, -V_eta_R])
    c_half = np.min([c_L, c_R])

    M_L = V_eta_L / c_half
    M_R = V_eta_R / c_half
    M = 0.5 * (M_L * M_L + M_R * M_R)
    Mco = _k * Minf
    # from JCP 214 2006, eq. 14 approximation
    Mo = np.sqrt(np.min(1, np.max(M*M, Mco*Mco)))
    fa = 2 * Mo
    # fa = 1 # dissipation factor, fa=1 the basic AUSM scheme

    # M_half = U_xi_half / c  # c is speed of sound at cell face 

    rho_half = 0.5 * (rho_L + rho_R)
    M_P = -Kp * np.max([1 - sigma*M*M, 0]) * (P_R - P_L)/(rho_half * c_half * c_half)
    
    M_1_plus = 0.5 * (M + np.abs(M))
    M_1_minus= 0.5 * (M - np.abs(M))

    M_2_plus = 0.25 * np.power(M + 1, 2)
    M_2_minus = -0.25 * np.power(M - 1, 2)

    M_plus_m = {True: M_1_plus, False: M_2_plus * (1 - 16 * beta * M_2_minus)}[M >= 1]
    M_minus_m = {True: M_1_minus, False: M_2_minus * (1 + 16 * beta * M_2_plus)}[M >= 1]

    M_half = M_plus_m * M_L + M_minus_m * M_R + M_P

    #### Calculate mass flow rate ###
    U_xi_half = c_half * M_half
    rho = {True : rho_L, False : rho_R}[U_xi_half > 0]
    m_dot_half = rho * c * M_half

    #### Calculate Pressure ####
    # must be either left or right side depending on conditon:
    # eps_L = np.array([1, u, v, ht])
    # eps = eps_R   # default is right side 
    # if m_dot > 0: eps = eps_L  # otherwise, switch to left side 

    alpha = (3/16) * (-4 + 5 * fa * fa)
    P_plus_5 = {True: (1/M)*M_1_plus,
                False: M_2_plus * (2 - M) - 16 *\
                      alpha * M * M_2_minus}[np.abs(M) >= 1] 
    P_minus_5 = {True: (1/M)*M_1_minus,
                 False: M_2_minus * (-2 - M) + 16 *\
                    alpha * M * M_2_plus}[np.abs(M) >= 1] 
    P_u = -Ku * P_plus_5 * P_minus_5 * (rho_L + rho_R) *\
          (fa * c_half) * (u_R - u_L)
    p_half = P_plus_5 * M_L * P_L +\
          P_minus_5 * M_R * P_R + P_u

    eps_L = np.array([1, u_L, v_L, ht_L])
    eps_R = np.array([1, u_R, v_R, ht_R])
    F_eta_half = 0.5 * m_dot_half * (eps_L + eps_R) -\
          0.5 * np.abs(m_dot_half) * (eps_L + eps_R) + p_half

    # p = np.array([0, (S_xi_x / S_xi)*p, (S_xi_y / S_xi)*p, 0])
    # E_xi = m_dot * eps + p
    # dissipation coefficient
    # D = M_half # mach number at i-1/2,j cell face (for AUSM)
    # Q_L_half, Q_R_half = interp_MUSCL()
    # E_xi_half = 0.5 * (E_xi * Q_L_half + E_xi * Q_R_half) -\
    # 0.5 * np.abs(D) * (Q_R_half - Q_L_half)
    return F_eta_half 

def get_local_time_step(CFL_max:float, flow:dict, area:pd.DataFrame, 
                        xsize:int, ysize:int, shrink_factor:float=1.0):

    c = flow['c']
    u = flow['u']
    v = flow['v']

    xi_x = area['xi_x']
    xi_y = area['xi_y']
    eta_x = area['eta_x']
    eta_y = area['eta_y']

    # TODO is this acceptable? 
    _v = CFL_max
    VNN_max = CFL_max

    xi = np.power(xi_x,2) + np.power(xi_y,2)
    eta = np.power(eta_x,2) + np.power(eta_y,2)

    U_xi = np.sqrt(np.power((eta_y * u)/xi, 2) +
                    np.power((eta_x * v)/xi,2))
    V_eta = np.sqrt(np.power((xi_y * u)/eta, 2) +
                     np.power((xi_x * v)/eta,2))

    U_xi_term = (U_xi + c) * np.sqrt(xi)
    vn_eta_term = (V_eta + c) * np.sqrt(eta)
    v_xi_term = 1 / (_v * xi)
    v_eta_term = 1 / (_v * eta)

    # obtain the time step at each face center (locally)
    dt = np.zeros(xsize*ysize)
    for iy in range(ysize):
        for ix in range(xsize): 
            currIdx = (iy*xsize) + ix
            CFL_term = CFL_max * np.min([U_xi_term[currIdx],
                                          vn_eta_term[currIdx]])
            VNN_term = VNN_max * np.min([v_xi_term[currIdx], 
                                         v_eta_term[currIdx]])
            dt[currIdx] = np.min([CFL_term, VNN_term])

    dt = dt * shrink_factor
    return dt

if __name__ == "__main__":

    # TODO List:
    # 1. viscous fluxes (topic 24.1) - only for 65x65 grid
    # 2. TVD, MUSCL scheme 
    # 3. assess how the other grids do 

    debug_flag = True
    show_plot_flag = False

    grid_file = "g65x49u-1.dat"
    # grid_file = 'g33x25u.dat'
    # grid_file = 'g65x65s.dat'

    CFL_max = 0.6
    solver_hard_stop_itr = 100
    eps = 0.1

    # 1st order accurate
    e = 0
    # 2nd or 3rd order accurate
    # e = 1

    # 2nd order full upwind
    k = -1
    # 2nd order upwind biased (Fromm scheme)
    # k = 0
    # 3rd order full upwind
    # k = 1/3
    # 3rd order upwind biased (QUICK scheme)
    # k = 1/2
    # 2nd order central
    # k = 1

    # Inlet Conditions
    inlet = {
        'M': 2.0,
        'p': 101325,
        'T': 300,
        'c': 347.2,
        'R': 287.,
        'gamma': 1.4,
        'Cp': 1005, 
        'rho': 101325 / 287. / 300.,
        'u': 347.2 * 2.0,
        'v': 0.0,
    } 

    grid_coords, grid_size = extract_grid(grid_file)
    print(f"Grid size: {grid_size[0]}, {grid_size[1]}")
    if debug_flag: plot_grid(grid_coords=grid_coords, show_plot=show_plot_flag,)

    new_grid_coords, new_grid_size =\
        add_halo_cells(grid_coords=grid_coords, grid_size=grid_size)
    if debug_flag: plot_grid(grid_coords=new_grid_coords, show_plot=show_plot_flag,)

    vol = calculate_cell_volumes(new_grid_coords, new_grid_size)
    vol = vol[vol['Vol'] > 0.0] # to format the colorbar scaling
    if debug_flag: plot_contour(grid_coords=vol, hue='Vol', show_plot=show_plot_flag,
                                save_plot_as=r'plots\cell_volume.png',
                                colorbar_label='Cell Volume')

    area, xsize, ysize = calculate_cell_face_areas(grid_coords, new_grid_size)
    area = area[area['Area_eta'] > 0.0]
    area = area[area['Area_xi'] > 0.0]
    if debug_flag: plot_contour(area, hue='Area_eta', show_plot=show_plot_flag,
                                save_plot_as=r'plots\cell_face_area_eta.png',
                                colorbar_label=r'Cell Face Area in $\eta$ Direction')
    if debug_flag: plot_contour(area, hue='Area_xi', show_plot=show_plot_flag,
                                save_plot_as=r'plots\cell_face_area_xi.png',
                                colorbar_label=r'Cell Face Area in $\xi$ Direction')

    area_xi = area['Area_xi']
    area_eta = area['Area_eta']
    eta_x = area['eta_x']
    eta_y = area['eta_y']
    xi_x = area['xi_x']
    xi_y = area['xi_y']

    # select local time step from finite-volume coordinates (Topic 24.3 pg 1-4)
    dt = get_local_time_step(CFL_max, inlet, area, xsize, ysize)

    # construct fluxes at each cell face and evaluate the system of equations 
    # for each cell (in Topic 19) - apply local time stepping (Topic 26(6))
    # this is where AUSM comes into play... (page 5 in Topic 28)
    # initialize the loop
    Qs, Q = initialize_solver(inlet, vol, xsize, ysize)

    # convergence rate tracking
    L2_norm_sq = np.zeros(solver_hard_stop_itr)
    Linf_norm = np.zeros(solver_hard_stop_itr)
    L2_norm_sq_curr = 1.0 
    i = 0
    while L2_norm_sq_curr > eps and i < solver_hard_stop_itr:
        L2_norm_sq_prev = 0
        Linf_norm_curr = 0

        Qs = set_boundary_conditions(Qs, inlet)
        # must solve for each cell...
        for iy in range(1,ysize-1):
            for ix in range(1,xsize-1): 
                # at i-1/2,j and i+1/2,j for each iteration
                idx = (iy * xsize) + ix
                Q_prev = Qs[idx]
                _dt = dt[idx]

                Q_L_half, Q_R_half = interp_MUSCL(Q_prev, e, k)

                S_xi = area_xi[idx]
                S_xi_L  = area_xi[idx-1]
                S_xi_R = area_xi[idx+1]
                S_xi_x_L = eta_y[idx-1]
                S_xi_x_R = eta_y[idx+1]
                S_xi_y_L = eta_x[idx-1]
                S_xi_y_R = eta_x[idx+1]
                E_xi = calculate_E_fluxes(Q_L_half, Q_prev, Q_R_half,
                                          S_xi_x_L, S_xi_x_R, 
                                          S_xi_y_L, S_xi_y_R, 
                                          S_xi_L, S_xi_R)
                
                S_eta = area_eta[idx]
                S_xi_L  = area_eta[idx-1]
                S_xi_R = area_eta[idx+1]
                S_eta_x_L = xi_y[idx-1]
                S_eta_x_R = xi_y[idx+1]
                S_eta_y_L = xi_x[idx-1]
                S_eta_y_R = xi_x[idx+1]
                F_eta = calculate_F_fluxes(Q_L_half, Q_prev, Q_R_half,
                                            S_eta_x_L, S_eta_x_R, 
                                            S_eta_y_L, S_eta_y_R,
                                            S_eta)

                # TODO enforce inlet and exit plane conditions
                # exit plane: 1st order extrapolation - Q_q+1_nx,j = Q_q_nx-1,j
                Q_next = Q_prev - _dt * (E_xi * S_xi + F_eta * S_eta)

                # TODO normalize by Qref (ref inflow pressure, temperature, and velocity magnitude)
                L2_norm_sq_curr = L2_norm_sq_prev + np.power(Q_next - Q_prev,2)
                L2_norm_sq_prev = L2_norm_sq_curr
                Linf_norm_curr = np.max(Linf_norm_curr, np.abs(Q_prev - Q_next))

                printout = f'iter: {idx:5} \t||\t L2 Norm Sq: {L2_norm_sq_curr:.7f}'
                print(printout)

                Qs[idx] = Q_next

        L2_norm_sq[i] = L2_norm_sq_curr
        Linf_norm[i] = Linf_norm_curr
        i = i + 1

        # TODO record the latest flow properties to plot
        if L2_norm_sq_curr >= eps: 
            pass
