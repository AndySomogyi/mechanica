/*
 * engine_advance.h
 *
 *  Created on: Jan 2, 2021
 *      Author: andy
 */

#ifndef SRC_MDCORE_SRC_ENGINE_ADVANCE_H_
#define SRC_MDCORE_SRC_ENGINE_ADVANCE_H_

/**
 * @brief Update the particle velocities and positions, re-shuffle if
 *      appropriate.
 * @param e The #engine on which to run.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */
CAPI_FUNC(int) engine_advance ( struct engine *e );

#endif /* SRC_MDCORE_SRC_ENGINE_ADVANCE_H_ */
