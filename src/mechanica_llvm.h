/*
 * cayman_llvm.h
 *
 *  Created on: Jul 9, 2015
 *      Author: andy
 *
 * The LLVM API is in a ****CONSTANT**** state of flux,
 * it is the most irritating aspect of LLVM, they
 * CANNOT SEEM TO COME UP WITH A STABLE API!
 *
 * This file is here to help manage the changes in one location
 * in LLVM as new version arise.
 */

#ifndef _INCLUDED_LLVM_H_
#define _INCLUDED_LLVM_H_

#include "llvm/Analysis/Passes.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/LambdaResolver.h"
#include "llvm/ExecutionEngine/Orc/LazyEmittingLayer.h"
#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Transforms/Scalar.h"


#endif /* SRC_CAYMAN_LLVM_H_ */
