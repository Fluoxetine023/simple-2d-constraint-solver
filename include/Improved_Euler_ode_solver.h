#ifndef ATG_SIMPLE_2D_CONSTRAINT_SOLVER_IMPROVED_EULER_ODE_SOLVER_H
#define ATG_SIMPLE_2D_CONSTRAINT_SOLVER_IMPROVED_EULER_ODE_SOLVER_H

#include "ode_solver.h"

namespace atg_scs {
class ImprovedEulerOdeSolver : public OdeSolver {
public:
    enum class Stage { Stage_1, Stage_2, Complete, Undefined };

public:
    ImprovedEulerOdeSolver();
    virtual ~ImprovedEulerOdeSolver();

    virtual void start(SystemState *initial, double dt) override;
    virtual bool step(SystemState *system) override;
    virtual void solve(SystemState *system) override;
    virtual void end() override;

protected:
    static Stage getNextStage(Stage stage);

protected:
    Stage m_stage;
    Stage m_nextStage;

    SystemState m_initialState;
    SystemState m_accumulator;
    SystemState m_predictedState;// Added to store predicted state
};

} /* namespace atg_scs */

#endif /* ATG_SIMPLE_2D_CONSTRAINT_SOLVER_IMPROVED_EULER_ODE_SOLVER_H */