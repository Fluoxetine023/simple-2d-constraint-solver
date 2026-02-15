#include "../include/Improved_Euler_ode_solver.h"

atg_scs::ImprovedEulerOdeSolver::ImprovedEulerOdeSolver() {
    m_stage = m_nextStage = Stage::Undefined;
}

atg_scs::ImprovedEulerOdeSolver::~ImprovedEulerOdeSolver() {
    m_initialState.destroy();
    m_accumulator.destroy();
    m_predictedState.destroy();
}

void atg_scs::ImprovedEulerOdeSolver::start(SystemState *initial, double dt) {
    OdeSolver::start(initial, dt);

    m_initialState.copy(initial);
    m_accumulator.copy(initial);
    m_predictedState.copy(initial);

    m_stage = Stage::Stage_1;
}

bool atg_scs::ImprovedEulerOdeSolver::step(SystemState *system) {
    switch (m_stage) {
        case Stage::Stage_1:
            // --- PREDICTOR STEP ---
            // Store initial state
            m_initialState.copy(system);

            // Temporary estimate: y_pred = y_n + dt * f(y_n)
            for (int i = 0; i < system->n; ++i) {
                system->v_theta[i] = m_initialState.v_theta[i] +
                                     m_dt * m_initialState.a_theta[i];
                system->theta[i] = m_initialState.theta[i] +
                                   m_dt * m_initialState.v_theta[i];
                system->v_x[i] =
                        m_initialState.v_x[i] + m_dt * m_initialState.a_x[i];
                system->v_y[i] =
                        m_initialState.v_y[i] + m_dt * m_initialState.a_y[i];
                system->p_x[i] =
                        m_initialState.p_x[i] + m_dt * m_initialState.v_x[i];
                system->p_y[i] =
                        m_initialState.p_y[i] + m_dt * m_initialState.v_y[i];
            }

            // Store predicted state for use in corrector
            m_predictedState.copy(system);

            system->dt = m_dt;
            m_nextStage = Stage::Stage_2;
            break;

        case Stage::Stage_2:
            // --- CORRECTOR STEP ---
            // After physics computation, system now contains derivatives at predicted state
            // We average the initial derivative and this new derivative.
            for (int i = 0; i < system->n; ++i) {
                // Heun's Formula: y_next = y_n + (dt/2) * [f(y_n) + f(y_pred)]

                system->v_theta[i] = m_initialState.v_theta[i] +
                                     (m_dt / 2.0) * (m_initialState.a_theta[i] +
                                                     system->a_theta[i]);
                system->theta[i] = m_initialState.theta[i] +
                                   (m_dt / 2.0) * (m_initialState.v_theta[i] +
                                                   m_predictedState.v_theta[i]);
                system->v_x[i] =
                        m_initialState.v_x[i] +
                        (m_dt / 2.0) * (m_initialState.a_x[i] + system->a_x[i]);
                system->v_y[i] =
                        m_initialState.v_y[i] +
                        (m_dt / 2.0) * (m_initialState.a_y[i] + system->a_y[i]);
                system->p_x[i] = m_initialState.p_x[i] +
                                 (m_dt / 2.0) * (m_initialState.v_x[i] +
                                                 m_predictedState.v_x[i]);
                system->p_y[i] = m_initialState.p_y[i] +
                                 (m_dt / 2.0) * (m_initialState.v_y[i] +
                                                 m_predictedState.v_y[i]);
            }

            m_nextStage = Stage::Complete;
            m_stage = m_nextStage;
            return true;// Step complete

        case Stage::Complete:
            m_stage = Stage::Stage_1;
            return false;

        default:
            return false;
    }

    m_stage = m_nextStage;
    return false;
}

void atg_scs::ImprovedEulerOdeSolver::solve(SystemState *system) {
    // This method can be used for accumulator-based implementations
    // For basic Improved Euler, the work is done in step()
}

void atg_scs::ImprovedEulerOdeSolver::end() {
    OdeSolver::end();
    m_stage = m_nextStage = Stage::Undefined;
}

atg_scs::ImprovedEulerOdeSolver::Stage
atg_scs::ImprovedEulerOdeSolver::getNextStage(Stage stage) {
    switch (stage) {
        case Stage::Stage_1:
            return Stage::Stage_2;
        case Stage::Stage_2:
            return Stage::Complete;
        case Stage::Complete:
            return Stage::Stage_1;
        default:
            return Stage::Undefined;
    }
}