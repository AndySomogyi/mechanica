/*
 * ScalarField.h
 *
 *  Created on: Aug 10, 2017
 *      Author: andy
 */

#ifndef SRC_SCALARFIELD_H_
#define SRC_SCALARFIELD_H_

#include "mechanica_private.h"

struct ScalarFieldDesc {

};

#if 0



/**
 * In mathematics and physics, a scalar field associates a scalar value to every
 * point in a space. The scalar may either be a mathematical number or a physical
 * quantity. Mechanica incorporates this idea and allows a scalar field to be
 * associated with any *region* of space. Mechanica elements define a spatial
 * region. Scalar fields are required to be coordinate-independent, meaning
 * that any two observers using the same units will agree on the value of the
 * scalar field at the same absolute point in space (or spacetime) regardless
 * of their respective points of origin. Examples used in physics include
 * the temperature distribution throughout space, the pressure distribution
 * in a fluid. Most commonly, scalar fields will be used to attach a set of
 * chemical solutes (species) to regions of space.
 *
 * The ScalarField interface provides a way to access scalar field values.
 *
 * Initial Conditions???  What is the best way to associate initial conditions
 * with scalar fields, should init conditions be attached to the object type,
 * or the object instance?
 */
struct  ScalarField
{
public:

    /**
     * Loads the initial conditions into the current model state.
     *
     * Initial conditions may have been updated at any time externally.
     */
    virtual void reset() = 0;



    /************************ Floating Species Section ****************************/
    #if (1) /**********************************************************************/
    /******************************************************************************/

    /**
     * dependent species are defined by rules and the only way to change them
     * is by changing the values on which they depend.
     */
    virtual int getNumDepFloatingSpecies() = 0;

    /**
     * total number of floating species.
     */
    virtual int getNumFloatingSpecies() = 0;

    virtual int getFloatingSpeciesIndex(const std::string& eid) = 0;
    virtual std::string getFloatingSpeciesId(int index) = 0;

    /**
     * independent species do are not defined by rules, they typically participate
     * in reactions and can have their values set at any time.
     */
    virtual int getNumIndFloatingSpecies() = 0;

    /**
     * get the floating species amounts
     *
     * @param[in] len the length of the indx and values arrays.
     * @param[in] indx an array of length len of boundary species to return.
     * @param[out] values an array of at least length len which will store the
     *                returned boundary species amounts.
     */
    virtual int getFloatingSpeciesAmounts(int len, int const *indx,
            MxReal *values) = 0;

    virtual int setFloatingSpeciesAmounts(int len, int const *indx,
            const MxReal *values) = 0;

    virtual int getFloatingSpeciesAmountRates(int len, int const *indx,
            MxReal *values) = 0;


    virtual int getFloatingSpeciesConcentrationRates(int len, int const *indx,
                MxReal *values) = 0;

    /**
     * get the floating species concentrations
     *
     * @param[in] len the length of the indx and values arrays.
     * @param[in] indx an array of length len of boundary species to return.
     * @param[out] values an array of at least length len which will store the
     *                returned boundary species amounts.
     */
    virtual int getFloatingSpeciesConcentrations(int len, int const *indx,
            MxReal *values) = 0;

    /**
     * set the floating species concentrations
     *
     * @param[in] len the length of the indx and values arrays.
     * @param[in] indx an array of length len of boundary species to return.
     * @param[in] values an array of at least length len which will store the
     *                returned boundary species amounts.
     */
    virtual int setFloatingSpeciesConcentrations(int len, int const *indx,
            MxReal const *values) = 0;

    /**
     * Set the initial concentrations of the floating species.
     *
     * Takes the same indices as the other floating species methods.
     *
     * Note, if a floating species has an initial assignment rule,
     * than the initial conditions value can only be set by
     * updating the values on which it depends, it can not be set
     * directly.
     */
    virtual int setFloatingSpeciesInitConcentrations(int len, int const *indx,
                MxReal const *values) = 0;

    /**
     * Get the initial concentrations of the floating species,
     * uses the same indexing as the other floating species methods.
     */
    virtual int getFloatingSpeciesInitConcentrations(int len, int const *indx,
                    MxReal *values) = 0;

    /**
     * Set the initial amounts of the floating species.
     *
     * Takes the same indices as the other floating species methods.
     *
     * Note, if a floating species has an initial assignment rule,
     * than the initial conditions value can only be set by
     * updating the values on which it depends, it can not be set
     * directly.
     */
    virtual int setFloatingSpeciesInitAmounts(int len, int const *indx,
                MxReal const *values) = 0;

    /**
     * Get the initial amounts of the floating species,
     * uses the same indexing as the other floating species methods.
     */
    virtual int getFloatingSpeciesInitAmounts(int len, int const *indx,
                    MxReal *values) = 0;

    /************************ End Floating Species Section ************************/
    #endif /***********************************************************************/
    /******************************************************************************/



    /************************ Boundary Species Section ****************************/
    #if (1) /**********************************************************************/
    /******************************************************************************/


    /**
     * get the number of boundary species.
     */
    virtual int getNumBoundarySpecies() = 0;
    virtual int getBoundarySpeciesIndex(const std::string &eid) = 0;
    virtual std::string getBoundarySpeciesId(int index) = 0;

    /**
     * get the boundary species amounts
     *
     * @param[in] len the length of the indx and values arrays.
     * @param[in] indx an array of length len of boundary species to return.
     * @param[out] values an array of at least length len which will store the
     *                returned boundary species amounts.
     */
    virtual int getBoundarySpeciesAmounts(int len, int const *indx,
            MxReal *values) = 0;


    /**
     * get the boundary species concentrations
     *
     * @param[in] len the length of the indx and values arrays.
     * @param[in] indx an array of length len of boundary species to return.
     * @param[out] values an array of at least length len which will store the
     *                returned boundary species amounts.
     */
    virtual int getBoundarySpeciesConcentrations(int len, int const *indx,
            MxReal *values) = 0;

    /**
     * set the boundary species concentrations
     *
     * @param[in] len the length of the indx and values arrays.
     * @param[in] indx an array of length len of boundary species to return.
     * @param[in] values an array of at least length len which will store the
     *                returned boundary species amounts.
     */
    virtual int setBoundarySpeciesConcentrations(int len, int const *indx,
            MxReal const *values) = 0;


    /************************ End Boundary Species Section ************************/
    #endif /***********************************************************************/
    /******************************************************************************/


    /************************ Global Parameters Section ***************************/
    #if (1) /**********************************************************************/
    /******************************************************************************/

    /**
     * get the number of global parameters
     */
    virtual int getNumGlobalParameters() = 0;

    /**
     * index of the global parameter id, -1 if it does not exist.
     */
    virtual int getGlobalParameterIndex(const std::string& eid) = 0;

    /**
     * id of the indexed global parameter.
     */
    virtual std::string getGlobalParameterId(int index) = 0;

    /**
     * get the global parameter values
     *
     * @param[in] len the length of the indx and values arrays.
     * @param[in] indx an array of length len of boundary species to return.
     * @param[out] values an array of at least length len which will store the
     *                returned boundary species amounts.
     */
    virtual int getGlobalParameterValues(int len, int const *indx,
            MxReal *values) = 0;

    virtual int setGlobalParameterValues(int len, int const *indx,
            const MxReal *values) = 0;


    /************************ Global Parameters Species Section *******************/
    #endif /***********************************************************************/
    /******************************************************************************/


    /**
     * Compartments
     * ============
     *
     * Scalar fields in mechanica are attached to a region of space, which is by
     * definition a single compartment.
     *
     * Mechanica only supports attaching a SBML model with a single compartment
     * to a element type. SBML models with multiple compartments must be attached
     * to multiple element instances.
     */




    /************************ Selection Ids Species Section ***********************/
    #if (1) /**********************************************************************/
    /******************************************************************************/

    /**
     * populates a given list with all the ids that this class can accept.
     *
     * @param ids: a list of strings that will be filled by this class.
     * @param types: the types of ids that are requested. Can be set to
     * 0xffffffff to request all the ids that this class supports.
     * This should by a bitwise OR of the filelds in SelectionRecord::SelectionType
     */
    virtual void getIds(int types, std::list<std::string> &ids) = 0;

    /**
     * returns a bit field of the ids that this class supports.
     */
    virtual int getSupportedIdTypes() = 0;

    /**
     * gets the value for the given id string. The string must be a SelectionRecord
     * string that is accepted by this class.
     */
    virtual MxReal getValue(const std::string& id) = 0;

    /**
     * sets the value coresponding to the given selection stringl
     */
    virtual void setValue(const std::string& id, MxReal value) = 0;


    /************************ End Selection Ids Species Section *******************/
    #endif /***********************************************************************/
    /******************************************************************************/

    /**
     * allocate a block of memory and copy the stochiometric values into it,
     * and return it.
     *
     * The caller is responsible for freeing the memory that is referenced by data.
     *
     * @param[out] rows will hold the number of rows in the matrix.
     * @param[out] cols will hold the number of columns in the matrix.
     * @param[out] data a pointer which will hold a newly allocated memory block.
     */
    virtual int getStoichiometryMatrix(int* rows, int* cols, MxReal** data) = 0;

    /**
     * Get the current stiochiometry value for the given species / reaction.
     *
     * If either are not valid, NaN is returned.
     */
    virtual MxReal getStoichiometry(int speciesIndex, int reactionIndex) = 0;


    virtual int getNumConservedMoieties() = 0;
    virtual int getConservedMoietyIndex(const std::string& eid) = 0;
    virtual std::string getConservedMoietyId(int index) = 0;
    virtual int getConservedMoietyValues(int len, int const *indx, MxReal *values) = 0;
    virtual int setConservedMoietyValues(int len, int const *indx,
            const MxReal *values) = 0;

    virtual int getNumRateRules() = 0;

    /**
     * get the number of reactions the model has
     */
    virtual int getNumReactions() = 0;

    /**
     * get the index of a named reaction
     * @returns >= 0 on success, < 0 on failure.
     */
    virtual int getReactionIndex(const std::string& eid) = 0;

    /**
     * get the name of the specified reaction
     */
    virtual std::string getReactionId(int index) = 0;

    /**
     * get the vector of reaction rates.
     *
     * @param len: the length of the suplied buffer, must be >= reaction rates size.
     * @param indx: pointer to index array. If NULL, then it is ignored and the
     * reaction rates are copied directly into the suplied buffer.
     * @param values: pointer to user suplied buffer where rates will be stored.
     */
    virtual int getReactionRates(int len, int const *indx,
                MxReal *values) = 0;

    /**
     * get the 'values' i.e. the what the rate rule integrates to, and
     * store it in the given array.
     *
     * The length of rateRuleValues obviously must be the number of
     * rate rules we have.
     */
    virtual void getRateRuleValues(MxReal *rateRuleValues) = 0;

    /**
     * get the id of an element of the state vector.
     */
    virtual std::string getStateVectorId(int index) = 0;

    /**
     * The state vector is a vector of elements that are defined by
     * differential equations (rate rules) or independent floating species
     * are defined by reactions.
     *
     * To get the ids of the state vector elements, use getStateVectorId.
     *
     * copies the internal model state vector into the provided
     * buffer.
     *
     * @param[out] stateVector a buffer to copy the state vector into, if NULL,
     *         return the size required.
     *
     * @return the number of items coppied into the provided buffer, if
     *         stateVector is NULL, returns the length of the state vector.
     */
    virtual int getStateVector(MxReal *stateVector) = 0;

    /**
     * sets the internal model state to the provided packed state vector.
     *
     * @param[in] an array which holds the packed state vector, must be
     *         at least the size returned by getStateVector.
     *
     * @return the number of items copied from the state vector, negative
     *         on failure.
     */
    virtual int setStateVector(const MxReal *stateVector) = 0;

    /**
     * the state vector y is the rate rule values and floating species
     * concentrations concatenated. y is of length numFloatingSpecies + numRateRules.
     *
     * The state vector is packed such that the first n raterule elements are the
     * values of the rate rules, and the last n floatingspecies are the floating
     * species values.
     *
     * @param[in] time current simulator time
     * @param[in] y state vector, must be either null, or have a size of that
     *         speciefied by getStateVector. If y is null, then the model is
     *         evaluated using its current state. If y is not null, then the
     *         y is considered the state vector.
     * @param[out] dydt calculated rate of change of the state vector, if null,
     *         it is ignored.
     */
    virtual void getStateVectorRate(MxReal time, const MxReal *y, MxReal* dydt=0) = 0;

    virtual void testConstraints() = 0;

    virtual std::string getInfo() = 0;

    virtual void print(std::ostream &stream) = 0;

    /******************************* Events Section *******************************/
    #if (1) /**********************************************************************/
    /******************************************************************************/

    virtual int getNumEvents() = 0;

    /**
     * get the event status, false if the even is not triggered, true if it is.
     *
     * The reason this returns an unsigned char instead of a bool array is this
     * array is typically stuffed into an std::vector, and std::vector<bool> is
     * well, weird as it's actually implemented as a bitfield, and can not be
     * used as a C array.
     *
     * So, on every modern system I'm aware of, bool is an unsigned char, so
     * use that data type here.
     */
    virtual int getEventTriggers(int len, const int *indx, unsigned char *values) = 0;


    /**
     * Itterate through all of the current and pending events and apply them. If any
     * events trigger a state change which triggers any additional events, these
     * are applied as well. After this method finishes, all events are processed.
     *
     * @param timeEnd: model time when the event occured.
     * @param previousEventStatus: array of previous event triggered states.
     * @param initialState (optional): initial state vector, may be NULL, in which
     * the current state is used.
     * @param finalState (optional): final state vector, where the final state is
     * coppied to. May be NULL, in which case, ignored.
     */
    virtual int applyEvents(MxReal timeEnd, const unsigned char* previousEventStatus,
                const MxReal *initialState, MxReal* finalState) = 0;


    /**
     * evaluate the event 'roots' -- when events transition form triggered - non-triggered
     * or triggered to non-triggered state.
     *
     * Simplest method is to return 1 for triggered, -1 for not-triggered, so long
     * as there is a zero crossing.
     *
     * @param time[in] current time
     * @param y[in] the state vector
     * @param gdot[out] result event roots, this is of length numEvents.
     */
    virtual void getEventRoots(MxReal time, const MxReal* y, MxReal* gdot) = 0;

    virtual MxReal getNextPendingEventTime(bool pop) = 0;

    virtual int getPendingEventSize() = 0;

    virtual void resetEvents() = 0;

    /**
     * need a virtual destructor as object implementing this interface
     * can be deleted directly, i.e.
     * ExecutableModel *p = createModel(...);
     * delete p;
     */
    virtual ~ScalarField() {};

    /******************************* Events Section *******************************/
     #endif /**********************************************************************/
    /******************************************************************************/


    /**
     * Gets the index for an event id.
     * If there is no event with this id, returns -1.
     */
    virtual int getEventIndex(const std::string& eid) = 0;
    virtual std::string getEventId(int index) = 0;


    /**
     * Get the amount rate of change for the i'th floating species
     * given a reaction rates vector.
     *
     * TODO: This should be merged with getFloatingSpeciesAmountRates, but that will
     * break interface, will do in next point release.
     *
     * TODO: If the conversion factor changes in between getting the
     * reaction rates vector via getReactionRates
     *
     * @param index: index of the desired floating species rate.
     * @param reactionRates: pointer to buffer of reaction rates.
     */
    virtual MxReal getFloatingSpeciesAmountRate(int index,
            const MxReal *reactionRates) = 0;

    /**
     * reset the model according to a bitfield specified by the
     * SelectionRecord::SelectionType values.
     */
    virtual void reset(int options) = 0;



};

#endif




#endif /* SRC_SCALARFIELD_H_ */
