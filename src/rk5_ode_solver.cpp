#include "../include/rk5_ode_solver.h"

atg_scs::Rk5OdeSolver::Rk5OdeSolver() {
    m_stage = m_nextStage = RkStage::Undefined;
}

atg_scs::Rk5OdeSolver::~Rk5OdeSolver() {
    m_initialState.destroy();
    m_accumulator.destroy();
    for (int i = 0; i < 6; i++) {
        m_k[i].destroy();
    }
}

void atg_scs::Rk5OdeSolver::start(SystemState *initial, double dt) {
    OdeSolver::start(initial, dt);

    m_initialState.copy(initial);
    m_accumulator.copy(initial);

    // Initialize memory for each of the 7 intermediate stages
    for (int i = 0; i < 6; i++) {
        m_k[i].copy(initial); 
    }

    m_stage = RkStage::Stage_1;
}

bool atg_scs::Rk5OdeSolver::step(SystemState *system) {
    // Using 6-stage 5th order Runge-Kutta method (RK5)

    switch (m_stage) {
        case RkStage::Stage_1:
            system->dt = 0.0;
            break;
        case RkStage::Stage_2:
            // c2 = 1/5
            for (int i = 0; i < system->n; ++i) {
                system->v_theta[i] =
                    m_initialState.v_theta[i] + m_dt * (m_k[0].a_theta[i] * 1.0/5.0);
                system->theta[i] =
                    m_initialState.theta[i] + m_dt * (m_k[0].v_theta[i] * 1.0/5.0);
                system->v_x[i] =
                    m_initialState.v_x[i] + m_dt * (m_k[0].a_x[i] * 1.0/5.0);
                system->v_y[i] =
                    m_initialState.v_y[i] + m_dt * (m_k[0].a_y[i] * 1.0/5.0);
                system->p_x[i] =
                    m_initialState.p_x[i] + m_dt * (m_k[0].v_x[i] * 1.0/5.0);
                system->p_y[i] =
                    m_initialState.p_y[i] + m_dt * (m_k[0].v_y[i] * 1.0/5.0);
            }
            system->dt = m_dt / 5.0;
            break;
        case RkStage::Stage_3:
            // c3 = 3/10, a31 = 3/40, a32 = 9/40
            for (int i = 0; i < system->n; ++i) {
                system->v_theta[i] =
                    m_initialState.v_theta[i] + m_dt * (
                        m_k[0].a_theta[i] * 3.0/40.0 +
                        m_k[1].a_theta[i] * 9.0/40.0);
                system->theta[i] =
                    m_initialState.theta[i] + m_dt * (
                        m_k[0].v_theta[i] * 3.0/40.0 +
                        m_k[1].v_theta[i] * 9.0/40.0);
                system->v_x[i] =
                    m_initialState.v_x[i] + m_dt * (
                        m_k[0].a_x[i] * 3.0/40.0 +
                        m_k[1].a_x[i] * 9.0/40.0);
                system->v_y[i] =
                    m_initialState.v_y[i] + m_dt * (
                        m_k[0].a_y[i] * 3.0/40.0 +
                        m_k[1].a_y[i] * 9.0/40.0);
                system->p_x[i] =
                    m_initialState.p_x[i] + m_dt * (
                        m_k[0].v_x[i] * 3.0/40.0 +
                        m_k[1].v_x[i] * 9.0/40.0);
                system->p_y[i] =
                    m_initialState.p_y[i] + m_dt * (
                        m_k[0].v_y[i] * 3.0/40.0 +
                        m_k[1].v_y[i] * 9.0/40.0);
            }
            system->dt = m_dt * 3.0 / 10.0;
            break;
        case RkStage::Stage_4:
            // c4 = 4/5, a41 = 44/45, a42 = -56/15, a43 = 32/9
            for (int i = 0; i < system->n; ++i) {
                double dv = (
                    44.0/45.0 * m_k[0].a_theta[i] +
                    -56.0/15.0 * m_k[1].a_theta[i] +
                    32.0/9.0 * m_k[2].a_theta[i]);

                double dtheta = (
                    44.0/45.0 * m_k[0].v_theta[i] +
                    -56.0/15.0 * m_k[1].v_theta[i] +
                    32.0/9.0 * m_k[2].v_theta[i]);

                double dvx = (
                    44.0/45.0 * m_k[0].a_x[i] +
                    -56.0/15.0 * m_k[1].a_x[i] +
                    32.0/9.0 * m_k[2].a_x[i]);

                double dvy = (
                    44.0/45.0 * m_k[0].a_y[i] +
                    -56.0/15.0 * m_k[1].a_y[i] +
                    32.0/9.0 * m_k[2].a_y[i]);

                double dpx = (
                    44.0/45.0 * m_k[0].v_x[i] +
                    -56.0/15.0 * m_k[1].v_x[i] +
                    32.0/9.0 * m_k[2].v_x[i]);

                double dpy = (
                    44.0/45.0 * m_k[0].v_y[i] +
                    -56.0/15.0 * m_k[1].v_y[i] +
                    32.0/9.0 * m_k[2].v_y[i]);

                system->v_theta[i] = m_initialState.v_theta[i] + m_dt * dv;
                system->theta[i] = m_initialState.theta[i] + m_dt * dtheta;
                system->v_x[i] = m_initialState.v_x[i] + m_dt * dvx;
                system->v_y[i] = m_initialState.v_y[i] + m_dt * dvy;
                system->p_x[i] = m_initialState.p_x[i] + m_dt * dpx;
                system->p_y[i] = m_initialState.p_y[i] + m_dt * dpy;
            }
            system->dt = m_dt * 4.0 / 5.0;
            break;
        case RkStage::Stage_5:
            // c5 = 8/9, a51 = 19372/6561, a52 = -25360/2187, a53 = 64448/6561, a54 = -212/729
            for (int i = 0; i < system->n; ++i) {
                double dv = (
                    19372.0/6561.0 * m_k[0].a_theta[i] +
                    -25360.0/2187.0 * m_k[1].a_theta[i] +
                    64448.0/6561.0 * m_k[2].a_theta[i] +
                    -212.0/729.0 * m_k[3].a_theta[i]);

                double dtheta = (
                    19372.0/6561.0 * m_k[0].v_theta[i] +
                    -25360.0/2187.0 * m_k[1].v_theta[i] +
                    64448.0/6561.0 * m_k[2].v_theta[i] +
                    -212.0/729.0 * m_k[3].v_theta[i]);

                double dvx = (
                    19372.0/6561.0 * m_k[0].a_x[i] +
                    -25360.0/2187.0 * m_k[1].a_x[i] +
                    64448.0/6561.0 * m_k[2].a_x[i] +
                    -212.0/729.0 * m_k[3].a_x[i]);

                double dvy = (
                    19372.0/6561.0 * m_k[0].a_y[i] +
                    -25360.0/2187.0 * m_k[1].a_y[i] +
                    64448.0/6561.0 * m_k[2].a_y[i] +
                    -212.0/729.0 * m_k[3].a_y[i]);

                double dpx = (
                    19372.0/6561.0 * m_k[0].v_x[i] +
                    -25360.0/2187.0 * m_k[1].v_x[i] +
                    64448.0/6561.0 * m_k[2].v_x[i] +
                    -212.0/729.0 * m_k[3].v_x[i]);

                double dpy = (
                    19372.0/6561.0 * m_k[0].v_y[i] +
                    -25360.0/2187.0 * m_k[1].v_y[i] +
                    64448.0/6561.0 * m_k[2].v_y[i] +
                    -212.0/729.0 * m_k[3].v_y[i]);

                system->v_theta[i] = m_initialState.v_theta[i] + m_dt * dv;
                system->theta[i] = m_initialState.theta[i] + m_dt * dtheta;
                system->v_x[i] = m_initialState.v_x[i] + m_dt * dvx;
                system->v_y[i] = m_initialState.v_y[i] + m_dt * dvy;
                system->p_x[i] = m_initialState.p_x[i] + m_dt * dpx;
                system->p_y[i] = m_initialState.p_y[i] + m_dt * dpy;
            }
            system->dt = m_dt * 8.0 / 9.0;
            break;
        case RkStage::Stage_6:
            // c6 = 1, a61 = 9017/3168, a62 = -355/33, a63 = 46732/5247, a64 = 49/176, a65 = -5103/18656
            for (int i = 0; i < system->n; ++i) {
                double dv = (
                    9017.0/3168.0 * m_k[0].a_theta[i] +
                    -355.0/33.0 * m_k[1].a_theta[i] +
                    46732.0/5247.0 * m_k[2].a_theta[i] +
                    49.0/176.0 * m_k[3].a_theta[i] +
                    -5103.0/18656.0 * m_k[4].a_theta[i]);

                double dtheta = (
                    9017.0/3168.0 * m_k[0].v_theta[i] +
                    -355.0/33.0 * m_k[1].v_theta[i] +
                    46732.0/5247.0 * m_k[2].v_theta[i] +
                    49.0/176.0 * m_k[3].v_theta[i] +
                    -5103.0/18656.0 * m_k[4].v_theta[i]);

                double dvx = (
                    9017.0/3168.0 * m_k[0].a_x[i] +
                    -355.0/33.0 * m_k[1].a_x[i] +
                    46732.0/5247.0 * m_k[2].a_x[i] +
                    49.0/176.0 * m_k[3].a_x[i] +
                    -5103.0/18656.0 * m_k[4].a_x[i]);

                double dvy = (
                    9017.0/3168.0 * m_k[0].a_y[i] +
                    -355.0/33.0 * m_k[1].a_y[i] +
                    46732.0/5247.0 * m_k[2].a_y[i] +
                    49.0/176.0 * m_k[3].a_y[i] +
                    -5103.0/18656.0 * m_k[4].a_y[i]);

                double dpx = (
                    9017.0/3168.0 * m_k[0].v_x[i] +
                    -355.0/33.0 * m_k[1].v_x[i] +
                    46732.0/5247.0 * m_k[2].v_x[i] +
                    49.0/176.0 * m_k[3].v_x[i] +
                    -5103.0/18656.0 * m_k[4].v_x[i]);

                double dpy = (
                    9017.0/3168.0 * m_k[0].v_y[i] +
                    -355.0/33.0 * m_k[1].v_y[i] +
                    46732.0/5247.0 * m_k[2].v_y[i] +
                    49.0/176.0 * m_k[3].v_y[i] +
                    -5103.0/18656.0 * m_k[4].v_y[i]);

                system->v_theta[i] = m_initialState.v_theta[i] + m_dt * dv;
                system->theta[i] = m_initialState.theta[i] + m_dt * dtheta;
                system->v_x[i] = m_initialState.v_x[i] + m_dt * dvx;
                system->v_y[i] = m_initialState.v_y[i] + m_dt * dvy;
                system->p_x[i] = m_initialState.p_x[i] + m_dt * dpx;
                system->p_y[i] = m_initialState.p_y[i] + m_dt * dpy;
            }
            system->dt = m_dt;
            break;
        case RkStage::Stage_7:
            // c7 = 1, a71 = 35/384, a72 = 0, a73 = 500/1113, a74 = 125/192, a75 = -2187/6784, a76 = 11/84
            for (int i = 0; i < system->n; ++i) {
                double dv = (
                    35.0/384.0 * m_k[0].a_theta[i] +
                    500.0/1113.0 * m_k[2].a_theta[i] +
                    125.0/192.0 * m_k[3].a_theta[i] +
                    -2187.0/6784.0 * m_k[4].a_theta[i] +
                    11.0/84.0 * m_k[5].a_theta[i]);

                double dtheta = (
                    35.0/384.0 * m_k[0].v_theta[i] +
                    500.0/1113.0 * m_k[2].v_theta[i] +
                    125.0/192.0 * m_k[3].v_theta[i] +
                    -2187.0/6784.0 * m_k[4].v_theta[i] +
                    11.0/84.0 * m_k[5].v_theta[i]);

                double dvx = (
                    35.0/384.0 * m_k[0].a_x[i] +
                    500.0/1113.0 * m_k[2].a_x[i] +
                    125.0/192.0 * m_k[3].a_x[i] +
                    -2187.0/6784.0 * m_k[4].a_x[i] +
                    11.0/84.0 * m_k[5].a_x[i]);

                double dvy = (
                    35.0/384.0 * m_k[0].a_y[i] +
                    500.0/1113.0 * m_k[2].a_y[i] +
                    125.0/192.0 * m_k[3].a_y[i] +
                    -2187.0/6784.0 * m_k[4].a_y[i] +
                    11.0/84.0 * m_k[5].a_y[i]);

                double dpx = (
                    35.0/384.0 * m_k[0].v_x[i] +
                    500.0/1113.0 * m_k[2].v_x[i] +
                    125.0/192.0 * m_k[3].v_x[i] +
                    -2187.0/6784.0 * m_k[4].v_x[i] +
                    11.0/84.0 * m_k[5].v_x[i]);

                double dpy = (
                    35.0/384.0 * m_k[0].v_y[i] +
                    500.0/1113.0 * m_k[2].v_y[i] +
                    125.0/192.0 * m_k[3].v_y[i] +
                    -2187.0/6784.0 * m_k[4].v_y[i] +
                    11.0/84.0 * m_k[5].v_y[i]);

                system->v_theta[i] = m_initialState.v_theta[i] + m_dt * dv;
                system->theta[i] = m_initialState.theta[i] + m_dt * dtheta;
                system->v_x[i] = m_initialState.v_x[i] + m_dt * dvx;
                system->v_y[i] = m_initialState.v_y[i] + m_dt * dvy;
                system->p_x[i] = m_initialState.p_x[i] + m_dt * dpx;
                system->p_y[i] = m_initialState.p_y[i] + m_dt * dpy;
            }
            system->dt = m_dt;
            break;
        default:
            break;
    }

    m_nextStage = getNextStage(m_stage);

    return m_nextStage == RkStage::Complete;
}

void atg_scs::Rk5OdeSolver::solve(SystemState *system) {
    int kIndex = (int)m_stage - 1;
    if (kIndex >= 0 && kIndex < 6) {
        m_k[kIndex].copy(system);
    }

    // RK5 weights from the last row of Butcher tableau
    double stageWeight = 0.0;
    switch (m_stage) {
        case RkStage::Stage_1: stageWeight = 5179.0 / 57600.0; break;
        case RkStage::Stage_2: stageWeight = 0.0; break;
        case RkStage::Stage_3: stageWeight = 7571.0 / 16695.0; break;
        case RkStage::Stage_4: stageWeight = 393.0 / 640.0; break;
        case RkStage::Stage_5: stageWeight = -92097.0 / 339200.0; break;
        case RkStage::Stage_6: stageWeight = 187.0 / 2100.0; break;
        case RkStage::Stage_7: stageWeight = 1.0 / 40.0; break;
        default: stageWeight = 0.0;
    }
    
    const double stepscale = m_dt * stageWeight;

    if (stageWeight != 0.0) {
        for (int i = 0; i < system->n; ++i) {
            m_accumulator.v_theta[i] += system->a_theta[i] * stepscale;
            m_accumulator.theta[i] += system->v_theta[i] * stepscale;
            m_accumulator.v_x[i] += system->a_x[i] * stepscale;
            m_accumulator.v_y[i] += system->a_y[i] * stepscale;
            m_accumulator.p_x[i] += system->v_x[i] * stepscale;
            m_accumulator.p_y[i] += system->v_y[i] * stepscale;
        }

        for (int i = 0; i < system->n_c; ++i) {
            m_accumulator.r_x[i] += system->r_x[i] * stepscale;
            m_accumulator.r_y[i] += system->r_y[i] * stepscale;
            m_accumulator.r_t[i] += system->r_t[i] * stepscale;
        }
    }

    if (m_stage == RkStage::Stage_7) {
        for (int i = 0; i < system->n; ++i) {
            system->v_theta[i] = m_accumulator.v_theta[i];
            system->theta[i] = m_accumulator.theta[i];
            system->v_x[i] = m_accumulator.v_x[i];
            system->v_y[i] = m_accumulator.v_y[i];
            system->p_x[i] = m_accumulator.p_x[i];
            system->p_y[i] = m_accumulator.p_y[i];
        }

        for (int i = 0; i < system->n_c; ++i) {
            system->r_x[i] = m_accumulator.r_x[i];
            system->r_y[i] = m_accumulator.r_y[i];
            system->r_t[i] = m_accumulator.r_t[i];
        }
    }

    m_stage = m_nextStage;
}

void atg_scs::Rk5OdeSolver::end() {
    OdeSolver::end();

    m_stage = m_nextStage = RkStage::Undefined;
}

atg_scs::Rk5OdeSolver::RkStage atg_scs::Rk5OdeSolver::getNextStage(RkStage stage) {
    switch (stage) {
        case RkStage::Stage_1: return RkStage::Stage_2;
        case RkStage::Stage_2: return RkStage::Stage_3;
        case RkStage::Stage_3: return RkStage::Stage_4;
        case RkStage::Stage_4: return RkStage::Stage_5;
        case RkStage::Stage_5: return RkStage::Stage_6;
        case RkStage::Stage_6: return RkStage::Stage_7;
        case RkStage::Stage_7: return RkStage::Complete;
        default: return RkStage::Undefined;
    }
}