/*
 * Best Offset prefetcher as implemented by Pierre Michaud.
 *
 * Reference: https://github.com/Quangmire/ChampSim/blob/master/prefetcher/bo.h
 */

#ifndef BOP_ORIG_HELPER_H
#define BOP_ORIG_HELPER_H

#include <algorithm>
#include <map>
#include <stdio.h>
#include <stdlib.h>
#include "cache.h"
#include "prefetcher.h"

namespace knob 
{
    extern vector<int32_t> bop_candidates;
    extern int32_t bop_default_candidate;
    extern uint32_t bop_max_rounds;
    extern uint32_t bop_max_score;
    extern uint32_t bop_low_score;
    extern uint32_t bop_bad_score;
    extern uint32_t bop_pref_degree;
}

// Submission ID: 3

// Paper title: A Best-Offset Prefetcher

// Author: Pierre Michaud

// (Modified to be a LLC prefetcher by Akanksha Jain)
// (Modified to work in Pythia prefetcher framework by Carson Molder)
// Prefetch Throttling is disabled since MSH info is not available
//######################################################################################
//                             PREFETCHER PARAMETERS 
//######################################################################################

// Because prefetch cannot cross 4KB-page boundaries, there is no need to consider offsets
// greater than 63. However, with pages larger than 4KB, it would be beneficial to consider
// larger offsets.

#define MAX_N_CANDIDATES 512
#define RRINDEX 6
#define RRTAG 12
#define DELAYQSIZE 15
#define DELAY 60
#define TIME_BITS 12
//#define LLC_RATE_MAX 255
//#define GAUGE_MAX 8191
//#define MSHR_THRESHOLD_MAX (LLC_MSHR_SIZE-4)
//#define MSHR_THRESHOLD_MIN 2
//#define LOW_SCORE 20
//#define BAD_SCORE ((knob_small_llc)? 10 : 1)
//#define BAD_SCORE 10
//#define BANDWIDTH ((knob_low_bandwidth)? 64 : 16)
//######################################################################################
//                               PREFETCHER STATE
//######################################################################################

int prefetch_offset;   // 7 bits (6-bit value + 1 sign bit)

// Recent Requests (RR) table: 2 banks, 64 entries per bank, RRTAG bits per entry
int recent_request[2][1<<RRINDEX]; // 2x64x12 = 1536 bits

// 1 prefetch bit per L2 cache line : 256x8 = 2048 bits 
// TODO: Make agnostic to cache level.
// const int NUM_SET = LLC_SET;
// const int NUM_WAY = LLC_WAY;
int** prefetch_bit; //int prefetch_bit[NUM_SET][NUM_WAY]; 
int num_set;
int num_way;


struct offsets_scores {
  int score[MAX_N_CANDIDATES];  // log2 SCORE_MAX = 5 bits per entry
  int max_score;                // log2 SCORE_MAX = 5 bits
  int best_offset;              // 7 bits (6-bit value + 1 sign bit)
  int round;                    // log2 ROUND_MAX = 7 bits
  int p;                        // log2 NOFFSETS = 6 bits
} os;                           // 46x5+5+7+7+6 = 255 bits


struct delay_queue {
  int lineaddr[DELAYQSIZE]; // RRINDEX+RTAG = 18 bits
  int cycle[DELAYQSIZE];    // TIME_BITS = 12 bits
  int valid[DELAYQSIZE];    // 1 bit 
  int tail;                 // log2 DELAYQSIZE = 4 bits
  int head;                 // log2 DELAYQSIZE = 4 bits
} dq;                       // 15x(18+12+1)+4+4 = 473 bits


struct prefetch_throttle {
  int mshr_threshold;     // log2 L2C_MSHR_SIZE = 4 bits
  int prefetch_score;     // log2 SCORE_MAX = 5 bits
  int llc_rate;           // log2 LLC_RATE_MAX = 8 bits
  int llc_rate_gauge;     // log2 GAUGE_MAX = 13 bits
  int last_cycle;         // TIME_BITS = 12 bits
} pt;                     // 4+5+8+13+12 = 42 bits

// Total prefetcher state: 7 + 1536 + 2048 + 255 + 473 + 42 = 4361 bits

// Statistics tracking (For final output)
struct stats_tracker {
  std::map<int32_t, uint32_t> candidate_chosen;
  std::map<int32_t, uint32_t> candidate_prefetched;
} stats;




//######################################################################################
//                            SOME MACROS & DEFINITIONS
//######################################################################################

#define LOGLINE 6

#define SAMEPAGE(lineaddr1,lineaddr2) ((((lineaddr1) ^ (lineaddr2)) >> 6) == 0)

#define INCREMENT(x,n) {x++; if (x==(n)) x=0;}

#define TRUNCATE(x,nbits) (((x) & ((1<<(nbits))-1)))

typedef long long t_addr;



//######################################################################################
//                            RECENT REQUESTS TABLE (RR)
//######################################################################################

void rr_init()
{
  int i;
  for (i=0; i<(1<<RRINDEX); i++) {
    recent_request[0][i] = 0;
    recent_request[1][i] = 0;
  }
}


int rr_tag(t_addr lineaddr)
{
  return TRUNCATE(lineaddr>>RRINDEX,RRTAG);
}


int rr_index_left(t_addr lineaddr)
{
  return TRUNCATE(lineaddr^(lineaddr>>RRINDEX),RRINDEX);
}


int rr_index_right(t_addr lineaddr)
{
  return TRUNCATE(lineaddr^(lineaddr>>(2*RRINDEX)),RRINDEX);
}


void rr_insert_left(t_addr lineaddr)
{
  int i = rr_index_left(lineaddr);
  recent_request[0][i] = rr_tag(lineaddr);
}


void rr_insert_right(t_addr lineaddr)
{
  int i = rr_index_right(lineaddr);
  recent_request[1][i] = rr_tag(lineaddr);
}


int rr_hit(t_addr lineaddr)
{
  int i = rr_index_left(lineaddr);
  int j = rr_index_right(lineaddr);
  int tag = rr_tag(lineaddr);
  return (recent_request[0][i] == tag) || (recent_request[1][j] == tag);
}



//######################################################################################
//                               DELAY QUEUE (DQ)
//######################################################################################

// Without the delay queue, the prefetcher would always try to select an offset value
// large enough for having timely prefetches. However, sometimes, a small offset yields
// late prefetches but greater prefetch accuracy and better performance. The delay queue
// is an imperfect solution to this problem.

// This implementation of the delay queue is specific to the DPC2 simulator, as the DPC2
// prefetcher can act only at certain clock cycles. In a real processor, the delay queue
// implementation can be simpler.


void dq_init()
{
  int i;
  for (i=0; i<DELAYQSIZE; i++) {
    dq.lineaddr[i] = 0;
    dq.cycle[i] = 0;
    dq.valid[i] = 0;
  }
  dq.tail = 0;
  dq.head = 0;
}


void dq_push(t_addr lineaddr)
{
  // enqueue one line address
  if (dq.valid[dq.tail]) {
    // delay queue is full
    // dequeue the oldest entry and write the "left" bank of the RR table
    rr_insert_left(dq.lineaddr[dq.head]);
    INCREMENT(dq.head,DELAYQSIZE);
  }
  dq.lineaddr[dq.tail] = TRUNCATE(lineaddr,RRINDEX+RRTAG);
  dq.cycle[dq.tail] = TRUNCATE(current_core_cycle[0],TIME_BITS);
  dq.valid[dq.tail] = 1;
  INCREMENT(dq.tail,DELAYQSIZE);
}


int dq_ready()
{
  // tells whether or not the oldest entry is ready to be dequeued
  if (! dq.valid[dq.head]) {
    // delay queue is empty
    return 0;
  }
    // TODO: Change to per-core cycle
  int cycle = TRUNCATE(current_core_cycle[0],TIME_BITS);
  int issuecycle = dq.cycle[dq.head];
  int readycycle = TRUNCATE(issuecycle+DELAY,TIME_BITS);
  if (readycycle >= issuecycle) {
    return (cycle < issuecycle) || (cycle >= readycycle);
  } else {
    return (cycle < issuecycle) && (cycle >= readycycle);
  }
}


void dq_pop()
{
  // dequeue the entries that are ready to be dequeued,
  // and do a write in the "left" bank of the RR table for each of them
  int i;
  for (i=0; i<DELAYQSIZE; i++) {
    if (! dq_ready()) {
      break;
    }
    rr_insert_left(dq.lineaddr[dq.head]);
    dq.valid[dq.head] = 0;
    INCREMENT(dq.head,DELAYQSIZE);
  }
}



//######################################################################################
//                               PREFETCH THROTTLE (PT)
//######################################################################################

// The following prefetch throttling method is specific to the DPC2 simulator, as other
// parts of the microarchitecture (requests schedulers, cache replacement policy,
// LLC hit/miss information,...) can be neither modified nor observed. Consequently,
// we ignore hardware implementation considerations here.

/*
void pt_init()
{
  pt.mshr_threshold = MSHR_THRESHOLD_MAX;
  pt.prefetch_score = knob::bop_max_score;
  pt.llc_rate = 0;
  pt.llc_rate_gauge = GAUGE_MAX/2;
  pt.last_cycle = 0;
}


// The pt_update_mshr_threshold function is for adjusting the MSHR threshold
// (a prefetch request is dropped when the MSHR occupancy exceeds the threshold)

void pt_update_mshr_threshold()
{
  if ((pt.prefetch_score > knob::bop_low_score) || (pt.llc_rate > (2*BANDWIDTH))) {
    // prefetch accuracy not too bad, or low bandwidth requirement
    // ==> maximum prefetch aggressiveness
    pt.mshr_threshold = MSHR_THRESHOLD_MAX;
  } else if (pt.llc_rate < BANDWIDTH) {
    // LLC access rate exceeds memory bandwidth, implying that there are some LLC hits.
    // If there are more LLC misses than hits, perhaps memory bandwidth saturates.
    // If there are more LLC hits than misses, the MSHR is probably not stressed.
    // So we set the MSHR threshold low.
    pt.mshr_threshold = MSHR_THRESHOLD_MIN;
  } else {
    // in-between situation: we set the MSHR threshold proportionally to the (inverse) LLC rate
    pt.mshr_threshold = MSHR_THRESHOLD_MIN + (MSHR_THRESHOLD_MAX-MSHR_THRESHOLD_MIN) * (double) (pt.llc_rate - BANDWIDTH) / BANDWIDTH;
  }
}


// The pt_llc_access function estimates the average time between consecutive LLC accesses.
// It is called on every LLC access.

void pt_llc_access()
{
  // update the gauge
  int cycle = TRUNCATE(current_core_cycle[0],TIME_BITS);
  int dt = TRUNCATE(cycle - pt.last_cycle,TIME_BITS);
  pt.last_cycle = cycle;
  pt.llc_rate_gauge += dt - pt.llc_rate;

  // if the gauge reaches its upper limit, increment the rate counter
  // if the gauge reaches its lower limit, decrement the rate counter
  // otherwise leave the rate counter unchanged
  if (pt.llc_rate_gauge > GAUGE_MAX) {
    pt.llc_rate_gauge = GAUGE_MAX;
    if (pt.llc_rate < LLC_RATE_MAX) {
      pt.llc_rate++;
      pt_update_mshr_threshold();
    }
  } else if (pt.llc_rate_gauge < 0) {
    pt.llc_rate_gauge = 0;
    if (pt.llc_rate > 0) {
      pt.llc_rate--;
      pt_update_mshr_threshold();
    }
  }
}
*/

//######################################################################################
//                               OFFSETS SCORES (OS)
//######################################################################################

// A method for determining the best offset value

void os_reset()
{
  uint32_t i;
  for (i = 0; i < knob::bop_candidates.size(); i++) {
    os.score[i] = 0;
  }
  os.max_score = 0;
  os.best_offset = 0;
  os.round = 0;
  os.p = 0;
}


// The os_learn_best_offset function tests one offset at a time, trying to 
// determine if the current line would have been successfully prefetched with
// that offset

void os_learn_best_offset(t_addr lineaddr)
{
  int testoffset = knob::bop_candidates[os.p];
  t_addr testlineaddr = lineaddr - testoffset;

  if (SAMEPAGE(lineaddr,testlineaddr) && rr_hit(testlineaddr)) {
    // the current line would likely have been prefetched successfully with that
    // offset ==> increment the score 
    os.score[os.p]++;
    if (os.score[os.p] >= os.max_score) {
      os.max_score = os.score[os.p];
      os.best_offset = testoffset;
    }
  }

  if (os.p == ((int)knob::bop_candidates.size() - 1)) {
    // one round finished
    os.round++;

    if ((os.max_score == (int)knob::bop_max_score) 
        || (os.round == (int)knob::bop_max_rounds)) {
      // learning phase is finished, update the prefetch offset
      prefetch_offset = 
        (os.best_offset != 0) ? os.best_offset : knob::bop_default_candidate;
      pt.prefetch_score = os.max_score;
      //pt_update_mshr_threshold();

      if (os.max_score <= (int)knob::bop_bad_score) {
        // prefetch accuracy is likely to be very low ==> turn the prefetch off 
        prefetch_offset = 0;
      }
      // new learning phase starts
      stats.candidate_chosen[prefetch_offset]++;
      os_reset();
      return;
    }
  }
  INCREMENT(os.p, (int)knob::bop_candidates.size()); // prepare to test the next offset
}


//######################################################################################
//                               DPC2 INTERFACE
//######################################################################################


void bo_prefetcher_initialize(uint8_t cache_type) {
    // Candidate list size remain within fixed array size of os.score.
    assert((int)knob::bop_candidates.size() <= MAX_N_CANDIDATES); 
    // Default offset must be in the candidate list.
    assert(std::find(
      knob::bop_candidates.begin(), knob::bop_candidates.end(),
      knob::bop_default_candidate) != knob::bop_candidates.end());

    prefetch_offset = knob::bop_default_candidate;
    rr_init();
    os_reset();
    dq_init();
    //pt_init();

    if (cache_type == IS_LLC) {
      num_set = LLC_SET;
      num_way = LLC_WAY;
    } else if (cache_type == IS_L2C) {
      num_set = L2C_SET;
      num_way = L2C_WAY;
    } else if (cache_type == IS_L1D) {
      num_set = L1D_SET;
      num_way = L1D_WAY;
    } else {
      assert(0); // Cache level must be L1D, L2, or LLC.
    }

    // Initialize prefetch_bit, a [num_set][num_way] 2D table.
    int i, j;
    prefetch_bit = (int**)malloc(num_set * sizeof(int*));
    for (i = 0; i < num_set; i++) {
        prefetch_bit[i] = (int*)malloc(num_way * sizeof(int));
        for (j = 0; j < num_way; j++) {
            prefetch_bit[i][j] = 0;
        }
    }
}

void bo_prefetcher_operate(uint64_t addr, uint64_t ip, uint8_t cache_hit, 
                           uint8_t type, int set, int way, 
                           vector<uint64_t>& prefetch_candidates) {
    t_addr lineaddr = addr >> LOGLINE;

    int s = set;
    int w = way;
    int hit = (w < num_way);
    int prefetched = 0;
    assert(prefetch_candidates.size() == 0);

    if (hit) {
        // read the prefetch bit, and reset it
        prefetched = prefetch_bit[s][w];
        prefetch_bit[s][w] = 0;
    } 
    else {
        //pt_llc_access();
    }

    dq_pop();

    //int prefetch_issued = 0;

    if (! hit || prefetched ) {
        os_learn_best_offset(lineaddr);

        int offset = prefetch_offset;
        if (offset == 0) {
            // The prefetcher is currently turned off.
            // Just push the line address into the delay queue for best-offset learning.
            dq_push(lineaddr);
            //prefetch_issued = 0; 
        }
        /*else if (! SAMEPAGE(lineaddr,lineaddr+offset)) {
            // crossing the page boundary, no prefetch request issued
            prefetch_issued = 0; 
        }*/
        else
        {
            dq_push(lineaddr);
            for(uint32_t i = 1; i <= knob::bop_pref_degree; i++) {
                if (pt.prefetch_score > (int)knob::bop_low_score) {
                    prefetch_candidates.push_back((lineaddr+i*offset)<<LOGLINE);
                    stats.candidate_prefetched[offset]++;
                    //prefetch_issued += cache->prefetch_line(ip ,lineaddr<<LOGLINE,(lineaddr+i*offset)<<LOGLINE, FILL_LLC, 0);
                }
            }
        }
    }
}


void bo_prefetcher_cache_fill(uint64_t addr, uint32_t set, uint32_t way, 
                              uint8_t prefetch, uint64_t evicted_addr)
{
    
    // In this version of the DPC2 simulator, the "prefetch" boolean passed
    // as input here is not reset whenever a demand request hits in the L2
    // MSHR on an in-flight prefetch request. Fortunately, this is the information
    // we need for updating the RR table for best-offset learning.
    // However, the prefetch bit stored in the L2 is not completely accurate
    // (though hopefully this does not impact performance too much).
    // In a real hardware implementation of the BO prefetcher, we would distinguish
    // "prefetched" and "demand-requested", which are independent informations.

    t_addr lineaddr = addr >> LOGLINE;

    // write the prefetch bit 
    int s = set;
    int w = way;
    prefetch_bit[s][w] = prefetch;

    // write the "right" bank of the RR table
    t_addr baselineaddr;
    if (prefetch || (prefetch_offset == 0)) {
        baselineaddr = lineaddr - prefetch_offset;
        if (SAMEPAGE(lineaddr,baselineaddr)) {
            rr_insert_right(baselineaddr);
        }
    }
}


void bo_prefetcher_final_stats() {
  for (auto candidate : knob::bop_candidates) {
    std::cout << std::dec
              << "bop.offset.index_" << candidate << ".chosen " << stats.candidate_chosen[candidate] << std::endl
              << "bop.offset.index_" << candidate << ".prefetched " << stats.candidate_prefetched[candidate] << std::endl;
  }
  std::cout << endl;
}

#endif // BOP_ORIG_HELPER_H
