#include "../include/rk6_ode_solver.h"

atg_scs::Rk6OdeSolver::Rk6OdeSolver() {
    m_stage = m_nextStage = RkStage::Undefined;
}

atg_scs::Rk6OdeSolver::~Rk6OdeSolver() {
    m_initialState.destroy();
    m_accumulator.destroy();
    for (int i = 0; i < 6; ++i) {
        m_k[i].destroy();
    }
}

void atg_scs::Rk6OdeSolver::start(SystemState *initial, double dt) {
    OdeSolver::start(initial, dt);

    m_initialState.copy(initial);
    m_accumulator.copy(initial);

    // Initialize memory for each of the 6 intermediate stages
    for (int i = 0; i < 6; ++i) {
        m_k[i].copy(initial); 
    }

    m_stage = RkStage::Stage_1;
}

bool atg_scs::Rk6OdeSolver::step(SystemState *state) {
    // Using 7-stage 6th order Runge-Kutta method
    switch (m_stage) {
        case RkStage::Stage_1:
            state->dt = 0.0;
            break;
        case RkStage::Stage_2:
            for (int i = 0; i < state->n; ++i) {
                state->v_theta[i] =
                    m_initialState.v_theta[i] + m_dt * (m_k[0].a_theta[i] * 1.0/3.0);
                state->theta[i] =
                    m_initialState.theta[i] + m_dt * (m_k[0].v_theta[i] * 1.0/3.0);
                state->v_x[i] =
                    m_initialState.v_x[i] + m_dt * (m_k[0].a_x[i] * 1.0/3.0);
                state->v_y[i] =
                    m_initialState.v_y[i] + m_dt * (m_k[0].a_y[i] * 1.0/3.0);
                state->p_x[i] =
                    m_initialState.p_x[i] + m_dt * (m_k[0].v_x[i] * 1.0/3.0);
                state->p_y[i] =
                    m_initialState.p_y[i] + m_dt * (m_k[0].v_y[i] * 1.0/3.0);
            }
            state->dt = m_dt / 3.0;
            break;
        case RkStage::Stage_3:
            for (int i = 0; i < state->n; ++i) {
                state->v_theta[i] =
                    m_initialState.v_theta[i] + m_dt * (m_k[1].a_theta[i] * 2.0/3.0);
                state->theta[i] =
                    m_initialState.theta[i] + m_dt * (m_k[1].v_theta[i] * 2.0/3.0);
                state->v_x[i] =
                    m_initialState.v_x[i] + m_dt * (m_k[1].a_x[i] * 2.0/3.0);
                state->v_y[i] =
                    m_initialState.v_y[i] + m_dt * (m_k[1].a_y[i] * 2.0/3.0);
                state->p_x[i] =
                    m_initialState.p_x[i] + m_dt * (m_k[1].v_x[i] * 2.0/3.0);
                state->p_y[i] =
                    m_initialState.p_y[i] + m_dt * (m_k[1].v_y[i] * 2.0/3.0);
            }
            state->dt = m_dt * 2.0 / 3.0;
            break;
        case RkStage::Stage_4:
            for (int i = 0; i < state->n; ++i) {
                double dv = (1.0/12.0 * m_k[0].a_theta[i] 
                   + 4.0/12.0 * m_k[1].a_theta[i] 
                   - 1.0/12.0 * m_k[2].a_theta[i]);

                double dtheta = (1.0/12.0 * m_k[0].v_theta[i] 
                    + 4.0/12.0 * m_k[1].v_theta[i] 
                    - 1.0/12.0 * m_k[2].v_theta[i]);
                double dvx = (1.0/12.0 * m_k[0].a_x[i] 
                    + 4.0/12.0 * m_k[1].a_x[i] 
                    - 1.0/12.0 * m_k[2].a_x[i]);
                double dvy = (1.0/12.0 * m_k[0].a_y[i] 
                    + 4.0/12.0 * m_k[1].a_y[i] 
                    - 1.0/12.0 * m_k[2].a_y[i]);
                double dpx = (1.0/12.0 * m_k[0].v_x[i] 
                    + 4.0/12.0 * m_k[1].v_x[i] 
                    - 1.0/12.0 * m_k[2].v_x[i]);
                double dpy = (1.0/12.0 * m_k[0].v_y[i] 
                    + 4.0/12.0 * m_k[1].v_y[i] 
                    - 1.0/12.0 * m_k[2].v_y[i]);
                state->v_theta[i] =
                    m_initialState.v_theta[i] + m_dt * dv;
                state->theta[i] =
                    m_initialState.theta[i] + m_dt * dtheta;
                state->v_x[i] =
                    m_initialState.v_x[i] + m_dt * dvx;
                state->v_y[i] =
                    m_initialState.v_y[i] + m_dt * dvy;
                state->p_x[i] =
                    m_initialState.p_x[i] + m_dt * dpx;
                state->p_y[i] =
                    m_initialState.p_y[i] + m_dt * dpy;
            }
            state->dt = m_dt / 3.0;
            break;
        case RkStage::Stage_5:
            for (int i = 0; i < state->n; ++i) {
                double dv = (-1.0/ 16.0   * m_k[0].a_theta[i] 
                   + 9.0/8.0   * m_k[1].a_theta[i] 
                   - 3.0/16.0  * m_k[2].a_theta[i] 
                   -3.0/8.0    * m_k[3].a_theta[i]);

                double dtheta = (- 1.0/16.0   * m_k[0].v_theta[i] 
                    + 9.0/ 8.0   * m_k[1].v_theta[i]
                    - 3.0/16.0   * m_k[2].v_theta[i] 
                    -3.0/8.0    * m_k[3].v_theta[i]);
                double dvx = (-1.0/ 16.0   * m_k[0].a_x[i] 
                    + 9.0/ 8.0   * m_k[1].a_x[i] 
                    - 3.0/16.0   * m_k[2].a_x[i] 
                    -3.0/8.0    * m_k[3].a_x[i]);
                double dvy = (-1.0/ 16.0   * m_k[0].a_y[i] 
                    + 9.0/ 8.0   * m_k[1].a_y[i] 
                    - 3.0/16.0   * m_k[2].a_y[i] 
                    -3.0/8.0    * m_k[3].a_y[i]);
                double dpx = (-1.0/ 16.0   * m_k[0].v_x[i] 
                    + 9.0/ 8.0   * m_k[1].v_x[i] 
                    - 3.0/16.0   * m_k[2].v_x[i] 
                    -3.0/8.0    * m_k[3].v_x[i]);
                double dpy = (-1.0/ 16.0   * m_k[0].v_y[i] 
                    + 9.0/ 8.0   * m_k[1].v_y[i] 
                    - 3.0/16.0   * m_k[2].v_y[i] 
                    -3.0/8.0    * m_k[3].v_y[i]);
                state->v_theta[i] =
                    m_initialState.v_theta[i] + m_dt * dv;
                state->theta[i] =
                    m_initialState.theta[i] + m_dt * dtheta;
                state->v_x[i] =
                    m_initialState.v_x[i] + m_dt * dvx;
                state->v_y[i] =
                    m_initialState.v_y[i] + m_dt * dvy;
                state->p_x[i] =
                    m_initialState.p_x[i] + m_dt * dpx;
                state->p_y[i] =
                    m_initialState.p_y[i] + m_dt * dpy;
            }
            state->dt = m_dt * 5.0 / 6.0;
            break;
        case RkStage::Stage_6:
            for (int i = 0; i < state->n; ++i) {
                double dv = ( 
                   9.0/8.0  * m_k[1].a_theta[i] 
                   -3.0/8.0  * m_k[2].a_theta[i] 
                   -3.0/4.0  * m_k[3].a_theta[i] 
                   + 1.0/2.0  * m_k[4].a_theta[i]);
                double dtheta = (
                   9.0/8.0  * m_k[1].v_theta[i] 
                   -3.0/8.0  * m_k[2].v_theta[i] 
                   -3.0/4.0  * m_k[3].v_theta[i] 
                   + 1.0/2.0  * m_k[4].v_theta[i]);
                double dvx = ( 
                   9.0/8.0  * m_k[1].a_x[i] 
                   -3.0/8.0  * m_k[2].a_x[i] 
                   -3.0/4.0  * m_k[3].a_x[i] 
                   + 1.0/2.0  * m_k[4].a_x[i]);
                double dvy = ( 
                   9.0/8.0  * m_k[1].a_y[i] 
                   -3.0/8.0  * m_k[2].a_y[i] 
                   -3.0/4.0  * m_k[3].a_y[i] 
                   + 1.0/2.0  * m_k[4].a_y[i]);
                double dpx = ( 
                   9.0/8.0  * m_k[1].v_x[i] 
                   -3.0/8.0  * m_k[2].v_x[i] 
                   -3.0/4.0  * m_k[3].v_x[i] 
                   + 1.0/2.0  * m_k[4].v_x[i]);
                double dpy = ( 
                   9.0/8.0  * m_k[1].v_y[i] 
                   -3.0/8.0  * m_k[2].v_y[i] 
                   -3.0/4.0  * m_k[3].v_y[i] 
                   + 1.0/2.0  * m_k[4].v_y[i]);
                state->v_theta[i] =
                    m_initialState.v_theta[i] + m_dt * dv;
                state->theta[i] =
                    m_initialState.theta[i] + m_dt * dtheta;
                state->v_x[i] =
                    m_initialState.v_x[i] + m_dt * dvx;
                state->v_y[i] =
                    m_initialState.v_y[i] + m_dt * dvy;
                state->p_x[i] =
                    m_initialState.p_x[i] + m_dt * dpx;
                state->p_y[i] =
                    m_initialState.p_y[i] + m_dt * dpy;
            }
            state->dt = m_dt;
            break;
        case RkStage::Stage_7:
            for (int i = 0; i < state->n; ++i) {
                double dv = (9.0/44.0  * m_k[0].a_theta[i] 
                   - 9.0/11.0 * m_k[1].a_theta[i] 
                   + 63.0/44.0  * m_k[2].a_theta[i] 
                   + 18.0/11.0  * m_k[3].a_theta[i] 
                   - 16.0/11.0  * m_k[5].a_theta[i]);

                double dtheta = (9.0/44.0  * m_k[0].v_theta[i] 
                   - 9.0/11.0 * m_k[1].v_theta[i] 
                   + 63.0/44.0  * m_k[2].v_theta[i] 
                   + 18.0/11.0  * m_k[3].v_theta[i]  
                   - 16.0/11.0  * m_k[5].v_theta[i]);

                double dvx = (9.0/44.0  * m_k[0].a_x[i] 
                   - 9.0/11.0 * m_k[1].a_x[i] 
                   + 63.0/44.0  * m_k[2].a_x[i] 
                   + 18.0/11.0  * m_k[3].a_x[i] 
                   - 16.0/11.0  * m_k[5].a_x[i]);

                double dvy = (9.0/44.0  * m_k[0].a_y[i] 
                   - 9.0/11.0 * m_k[1].a_y[i] 
                   + 63.0/44.0  * m_k[2].a_y[i] 
                   + 18.0/11.0  * m_k[3].a_y[i] 
                   - 16.0/11.0  * m_k[5].a_y[i]);

                double dpx = (9.0/44.0  * m_k[0].v_x[i] 
                   - 9.0/11.0 * m_k[1].v_x[i] 
                   + 63.0/44.0  * m_k[2].v_x[i] 
                   + 18.0/11.0  * m_k[3].v_x[i] 
                   - 16.0/11.0  * m_k[5].v_x[i]);

                double dpy = (9.0/44.0  * m_k[0].v_y[i] 
                   - 9.0/11.0 * m_k[1].v_y[i] 
                   + 63.0/44.0  * m_k[2].v_y[i] 
                   + 18.0/11.0  * m_k[3].v_y[i] 
                   - 16.0/11.0  * m_k[5].v_y[i]);
                

                state->v_theta[i] =
                    m_initialState.v_theta[i] + m_dt * dv;
                state->theta[i] =
                    m_initialState.theta[i] + m_dt * dtheta;
                state->v_x[i] =
                    m_initialState.v_x[i] + m_dt * dvx;
                state->v_y[i] =
                    m_initialState.v_y[i] + m_dt * dvy;
                state->p_x[i] =
                    m_initialState.p_x[i] + m_dt * dpx;
                state->p_y[i] =
                    m_initialState.p_y[i] + m_dt * dpy;
            }
            state->dt = m_dt;
            break;
        default:
            break;
    }

    m_nextStage = getNextStage(m_stage);

    return m_nextStage == RkStage::Complete;
}

void atg_scs::Rk6OdeSolver::solve(SystemState *system) {
    int kIndex = (int)m_stage - 1;
    if (kIndex >= 0 && kIndex < 6) {
        m_k[kIndex].copy(system);
    }

    double stageWeight = 0.0;
    switch (m_stage) {
        case RkStage::Stage_1: stageWeight = 11.0 / 120.0; break;
        case RkStage::Stage_2: stageWeight = 0.0; break;
        case RkStage::Stage_3: stageWeight = 27.0 / 40.0; break;
        case RkStage::Stage_4: stageWeight = 27.0 / 40.0; break;
        case RkStage::Stage_5: stageWeight = -4.0 / 15.0; break;
        case RkStage::Stage_6: stageWeight = -4.0 / 15.0; break;
        case RkStage::Stage_7: stageWeight = 11.0 / 120.0; break;
    default: stageWeight = 0.0;
    }
    const double stepscale = m_dt * stageWeight;

    if (stageWeight>0.0){

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

void atg_scs::Rk6OdeSolver::end() {
    OdeSolver::end();

    m_stage = m_nextStage = RkStage::Undefined;
}

atg_scs::Rk6OdeSolver::RkStage atg_scs::Rk6OdeSolver::getNextStage(RkStage stage) {
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