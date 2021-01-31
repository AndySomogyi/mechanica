/*
 * SpaceQueue.hpp
 *
 *  Created on: Jan 29, 2021
 *      Author: andy
 */
#pragma once
#ifndef SRC_MDCORE_SRC_SPACEQUEUE_HPP_
#define SRC_MDCORE_SRC_SPACEQUEUE_HPP_

#include <platform.h>

struct space;
struct space_cell;

struct SpaceTask {
    uint32_t id;
    uint64_t primeId;

    /**
     * product of the prime ids of the tasks that this task depends on.
     */
    uint64_t dependsIds;
};



/**
 * * a task can only depend on previous tasks, in that the tasks that a
 *   task depends on must be inserted earlier in the queue.
 */
class SpaceQueue
{
public:
    SpaceQueue();


    // should be first
    SpaceTask& createSortTask(int cell);

    // depends on sort for each task
    SpaceTask& createPairDensityTask(int cell_i, int cell_j);

    // depends on sort
    SpaceTask& createSelfDensityTask(int cell);

    // depends on self density for each cell
    SpaceTask& createPairForceTask(int cell_i, int cell_j);

    // depends on density
    SpaceTask& createForceTask(int cell);

    uint64_t completed_tasks;

    uint64_t current_tasks;

    uint64_t current_cells;




};

#endif /* SRC_MDCORE_SRC_SPACEQUEUE_HPP_ */
