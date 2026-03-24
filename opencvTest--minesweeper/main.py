from image_recognizer import State
from CSP import *
from loguru import logger
from log import init_loguru


def main():
    init_loguru()
    state = State()
    state.state_init()
    logger.info(state.grid)
    state.state_refresh({(3, 4):(423, 423)})
    while True:

        positions=state.grid
        builder = GetVariables(positions)
        variables=builder.variables
        constraints=builder.constraints
        csp=CSP(variables)
        safe_block=csp.get_safe_and_mine(constraints,positions)
        logger.info(safe_block)
        logger.info(state.grid)
        state.state_refresh(safe_block) 

if __name__=="__main__":
    main()