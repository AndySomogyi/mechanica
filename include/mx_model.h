/*
 * mx_model.h
 *
 *  Created on: Apr 2, 2017
 *      Author: andy
 */

#ifndef INCLUDE_MX_MODEL_H_
#define INCLUDE_MX_MODEL_H_

/**
 * The model objects.
 *
 * Models are compied with either Mx_CompileStringFlags or Mx_CompileString.
 */
typedef struct MxModel MxModel;


mx_real MxModel_ReadScalar(const MxModel *model, const MxSymbol *field, const mx_real *xyz);

HRESULT MxModel_GetParticleScalars(const MxModel* model, const MxSymbol *field,
		uint32_t partStart, uint32_t partEnd, size_t stride, void* buffer);

HRESULT MxModel_GetParticleAttrs(const MxModel *model, );







#endif /* INCLUDE_MX_MODEL_H_ */
